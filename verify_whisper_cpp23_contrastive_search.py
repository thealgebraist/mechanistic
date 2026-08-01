#!/usr/bin/env python3
"""Verify typed C++23 contrastive-search hidden-state transport."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import (
    GenerationConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    logging,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL = ROOT / "work/whisper_tiny_en"
AUDIO = ROOT / "work/whisper_sample.wav"
CPP = ROOT / "work/whisper_graph_cpp23"
REFERENCE = ROOT / "work/contrastive_search_reference"
REFERENCE_REVISION = "89ece6d21c47e6187e86d45d98fd495feadb33cb"
REFERENCE_HASH = "ea33addf7128014a238f3210abba3df7f8343acfb3880383f38ce5b34881c3d3"

CASES = [
    ("canonical", 4, 0.60, 32),
    ("narrow_low_penalty", 2, 0.20, 32),
    ("wide_strong_penalty", 8, 0.80, 18),
    ("five_candidates", 5, 0.45, 24),
    ("pure_degeneration", 3, 1.00, 8),
    ("eos_termination", 4, 0.30, 448),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


reference_source = REFERENCE / "custom_generate/generate.py"
if not reference_source.exists():
    hf_hub_download(
        "transformers-community/contrastive-search",
        "custom_generate/generate.py",
        revision=REFERENCE_REVISION,
        local_dir=REFERENCE,
    )
if sha256(reference_source) != REFERENCE_HASH:
    raise RuntimeError(f"contrastive reference hash mismatch: {reference_source}")
sys.path.insert(0, str(REFERENCE))
from custom_generate.generate import generate as reference_generate  # noqa: E402

logging.set_verbosity_error()
processor = WhisperProcessor.from_pretrained(MODEL, local_files_only=True)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL, local_files_only=True, dtype=torch.float32
).eval()
with wave.open(str(AUDIO), "rb") as source:
    pcm = (
        np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
        .astype(np.float32)
        / 32768.0
    )
inputs = processor(
    pcm, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
)
prefix = torch.tensor([[50257, 50362]], dtype=torch.long)


def configuration(candidates: int, penalty: float, maximum: int) -> GenerationConfig:
    config = GenerationConfig.from_model_config(model.config)
    config.max_length = maximum
    config.top_k = candidates
    config.penalty_alpha = penalty
    config.eos_token_id = 50256
    config.pad_token_id = 50256
    config.decoder_start_token_id = 50257
    config.suppress_tokens = model.generation_config.suppress_tokens
    config.begin_suppress_tokens = model.generation_config.begin_suppress_tokens
    config.low_memory = True
    return config


@torch.inference_mode()
def first_decision(candidates: int, penalty: float) -> dict:
    initial = model(
        input_features=inputs.input_features,
        attention_mask=inputs.attention_mask,
        decoder_input_ids=prefix,
        output_hidden_states=True,
        use_cache=True,
        return_dict=True,
    )
    context = initial.decoder_hidden_states[-1][0].float()
    logits = initial.logits[0, -1].float().clone()
    for token in model.generation_config.suppress_tokens:
        logits[token] = -torch.inf
    for token in model.generation_config.begin_suppress_tokens:
        logits[token] = -torch.inf
    probabilities = torch.softmax(logits, dim=-1)
    top_probabilities, top_tokens = torch.topk(probabilities, candidates)
    degenerations = []
    for token in top_tokens:
        candidate_ids = torch.cat([prefix, token.reshape(1, 1)], dim=1)
        candidate = model(
            input_features=inputs.input_features,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=candidate_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = candidate.decoder_hidden_states[-1][0, -1].float()
        cosine = torch.nn.functional.cosine_similarity(
            context, hidden.unsqueeze(0), dim=-1
        )
        degenerations.append(cosine.max())
    degenerations = torch.stack(degenerations)
    scores = (1.0 - penalty) * top_probabilities - penalty * degenerations
    return {
        "tokens": top_tokens.cpu().tolist(),
        "probabilities": top_probabilities.cpu().tolist(),
        "degenerations": degenerations.cpu().tolist(),
        "scores": scores.cpu().tolist(),
        "selected_rank": int(scores.argmax()),
    }


pattern = re.compile(
    r"WHISPER_CPP23_CONTRASTIVE_SEARCH candidates=(\d+) penalty=([^ ]+) "
    r"tokens=([^ ]+) candidate_branches=(\d+) cosine_edges=(\d+) "
    r"maximum_cosine=([^ ]+) first_candidates=([^ ]+) "
    r"first_probabilities=([^ ]+) first_degenerations=([^ ]+) "
    r"first_scores=([^ ]+) first_selected_rank=(\d+) "
    r"graph_nodes_visited=(\d+)"
)


def floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",")]


rows = []
for name, candidates, penalty, maximum in CASES:
    config = configuration(candidates, penalty, maximum)
    with torch.inference_mode():
        expected = reference_generate(
            model,
            inputs=inputs.input_features,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=prefix,
            generation_config=config,
        ).tolist()[0]
    decision = first_decision(candidates, penalty)
    command = [
        str(CPP),
        "--contrastive-search",
        str(MODEL / "model.safetensors"),
        str(OUT / "whisper_cpp23_tensor_manifest.tsv"),
        str(AUDIO),
        str(OUT / "whisper_cpp23_hann_f32.bin"),
        str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
        str(OUT / "whisper_cpp23_token_manifest.tsv"),
        str(OUT / "whisper_cpp23_token_bytes.bin"),
        str(candidates),
        str(penalty),
        str(maximum),
    ]
    line = subprocess.check_output(command, text=True).strip()
    match = pattern.fullmatch(line)
    assert match, line
    actual = [int(token) for token in match.group(3).split(",")]
    actual_candidates = [int(token) for token in match.group(7).split(",")]
    actual_probabilities = floats(match.group(8))
    actual_degenerations = floats(match.group(9))
    actual_scores = floats(match.group(10))
    probability_error = max(
        abs(a - b)
        for a, b in zip(actual_probabilities, decision["probabilities"])
    )
    degeneration_error = max(
        abs(a - b)
        for a, b in zip(actual_degenerations, decision["degenerations"])
    )
    score_error = max(
        abs(a - b) for a, b in zip(actual_scores, decision["scores"])
    )
    assert int(match.group(1)) == candidates
    assert abs(float(match.group(2)) - penalty) < 2e-6
    assert actual == expected, (name, actual, expected)
    assert actual_candidates == decision["tokens"], (name, actual_candidates, decision)
    assert probability_error < 1e-4, (name, probability_error)
    assert degeneration_error < 2e-5, (name, degeneration_error)
    assert score_error < 1e-4, (name, score_error)
    assert int(match.group(11)) == decision["selected_rank"]
    assert int(match.group(12)) == 74
    assert int(match.group(4)) == candidates * (len(actual) - len(prefix[0]))
    assert int(match.group(5)) > 0
    rows.append(
        {
            "case": name,
            "top_k": candidates,
            "penalty_alpha": penalty,
            "maximum_positions": maximum,
            "tokens": actual,
            "reference_tokens_exact": True,
            "first_candidate_tokens_exact": True,
            "first_selected_rank": int(match.group(11)),
            "maximum_first_probability_error": probability_error,
            "maximum_first_degeneration_error": degeneration_error,
            "maximum_first_contrastive_score_error": score_error,
            "candidate_branches": int(match.group(4)),
            "cosine_edges": int(match.group(5)),
            "maximum_observed_cosine": float(match.group(6)),
            "graph_nodes_visited": int(match.group(12)),
            "constructor": "Selection.ContrastiveSearch",
        }
    )

artifact = {
    "certificate": "WHISPER_CPP23_CONTRASTIVE_SEARCH_1",
    "reference_repository": "transformers-community/contrastive-search",
    "reference_revision": REFERENCE_REVISION,
    "reference_generate_sha256": REFERENCE_HASH,
    "case_count": len(rows),
    "all_token_sequences_exact": all(row["reference_tokens_exact"] for row in rows),
    "all_first_candidate_tokens_exact": all(
        row["first_candidate_tokens_exact"] for row in rows
    ),
    "worst_first_probability_error": max(
        row["maximum_first_probability_error"] for row in rows
    ),
    "worst_first_degeneration_error": max(
        row["maximum_first_degeneration_error"] for row in rows
    ),
    "worst_first_contrastive_score_error": max(
        row["maximum_first_contrastive_score_error"] for row in rows
    ),
    "all_graph_nodes_visited": all(row["graph_nodes_visited"] == 74 for row in rows),
    "cases": rows,
    "scope": (
        "Batch-one encoder-decoder contrastive search with explicit top-k candidate "
        "branches, copied K/V state, final-hidden-state cosine edges, confidence-minus-"
        "degeneration ranking, EOS termination, and low-memory sequential execution."
    ),
}
OUT.mkdir(exist_ok=True)
(OUT / "whisper_cpp23_contrastive_search.json").write_text(
    json.dumps(artifact, indent=2) + "\n"
)
table = "\n".join(
    f"| `{row['case']}` | {row['top_k']} | {row['penalty_alpha']} | "
    f"{row['maximum_positions']} | {len(row['tokens'])} | "
    f"{row['maximum_first_probability_error']:.6g} | "
    f"{row['maximum_first_degeneration_error']:.6g} | "
    f"{row['maximum_first_contrastive_score_error']:.6g} | yes |"
    for row in rows
)
(OUT / "WHISPER_CPP23_CONTRASTIVE_SEARCH.md").write_text(
    f"""# Explicit C++23 contrastive-search graph

Each decoding position expands a typed `ContrastiveSearch` state into `top_k` candidate K/V branches. Every candidate's final 384-dimensional decoder hidden state is connected to every prior context hidden state by a cosine edge. The transition score is `(1-alpha) * model_probability - alpha * maximum_context_cosine`; only the winning branch is retained.

The oracle is source-pinned revision `{REFERENCE_REVISION}` of `transformers-community/contrastive-search`, with SHA-256 `{REFERENCE_HASH}`. The C++ path uses the reference's low-memory sequential semantics, while making candidate branches and hidden-state edges explicit.

| case | top-k | alpha | max positions | output tokens | first probability error | first cosine-penalty error | first score error | complete tokens exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

All `{len(rows)}` complete token sequences match the pinned reference exactly. The first decision in every case also has the exact candidate-token set and selected rank. Worst first-decision errors are `{artifact['worst_first_probability_error']:.9g}` for probability, `{artifact['worst_first_degeneration_error']:.9g}` for degeneration penalty, and `{artifact['worst_first_contrastive_score_error']:.9g}` for the combined score.

This is a finite execution certificate under the declared C++23/Accelerate binary32 ABI, not a backend-independent proof for every waveform.
"""
)
print(
    json.dumps(
        {
            key: artifact[key]
            for key in (
                "certificate",
                "case_count",
                "all_token_sequences_exact",
                "all_first_candidate_tokens_exact",
                "worst_first_probability_error",
                "worst_first_degeneration_error",
                "worst_first_contrastive_score_error",
                "all_graph_nodes_visited",
            )
        },
        indent=2,
    )
)
