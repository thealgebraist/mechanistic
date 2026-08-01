#!/usr/bin/env python3
"""Verify explicit diverse-group beam state against the pinned HF module."""
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
REFERENCE = ROOT / "work/group_beam_reference_4_57"
REFERENCE_REVISION = "1a281620f7c5fa711c6a44d61c42a4e3a9c2098b"
REFERENCE_FILES = {
    "custom_generate/generate.py":
        "07cb918df0a9298b89debb926b672bf8fd688cc2e66ba97a756cd04c12d02b42",
    "custom_generate/beam_search.py":
        "b55a2e9c65c357391eb78cc37bf0ecaeb89034b4dc2789a2d9c7516f4dd803a4",
}

CASES = [
    ("four_beams_two_groups", 4, 2, 4, 8, 1.0, 0.5, False, "heuristic"),
    ("strong_diversity", 4, 2, 4, 10, 0.7, 2.0, False, "heuristic"),
    ("six_beams_three_groups", 6, 3, 6, 8, 1.0, 1.0, False, "heuristic"),
    ("eos_finalization", 4, 2, 4, 448, 1.0, 0.5, False, "heuristic"),
    ("all_finished_stopping", 4, 2, 2, 448, 0.7, 0.5, True, "all-finished"),
    ("canonical_stopping", 4, 2, 2, 448, 1.0, 0.5, "never", "canonical"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


for filename, expected_hash in REFERENCE_FILES.items():
    required = REFERENCE / filename
    if not required.exists():
        hf_hub_download(
            "transformers-community/group-beam-search",
            filename,
            revision=REFERENCE_REVISION,
            local_dir=REFERENCE,
        )
    if sha256(required) != expected_hash:
        raise RuntimeError(f"grouped-beam reference hash mismatch: {required}")

sys.path.insert(0, str(REFERENCE))
from custom_generate.generate import generate as reference_group_generate  # noqa: E402

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
decoder_prefix = torch.tensor([[50257, 50362]], dtype=torch.long)


def reference_configuration(
    beams: int,
    groups: int,
    returned: int,
    maximum: int,
    length_penalty: float,
    diversity_penalty: float,
    stopping,
) -> GenerationConfig:
    config = GenerationConfig.from_model_config(model.config)
    config.num_beams = beams
    config.num_beam_groups = groups
    config.num_return_sequences = returned
    config.max_length = maximum
    config.eos_token_id = 50256
    config.pad_token_id = 50256
    config.decoder_start_token_id = 50257
    config.return_dict_in_generate = True
    config.output_scores = True
    config.suppress_tokens = model.generation_config.suppress_tokens
    config.begin_suppress_tokens = model.generation_config.begin_suppress_tokens
    config.length_penalty = length_penalty
    config.diversity_penalty = diversity_penalty
    config.early_stopping = stopping
    return config


rows = []
for (
    name,
    beams,
    groups,
    returned,
    maximum,
    length_penalty,
    diversity_penalty,
    stopping,
    cpp_stopping,
) in CASES:
    with torch.inference_mode():
        expected = reference_group_generate(
            model,
            inputs=inputs.input_features,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=decoder_prefix,
            generation_config=reference_configuration(
                beams,
                groups,
                returned,
                maximum,
                length_penalty,
                diversity_penalty,
                stopping,
            ),
        )
    command = [
        str(CPP),
        "--group-beam-search",
        str(MODEL / "model.safetensors"),
        str(OUT / "whisper_cpp23_tensor_manifest.tsv"),
        str(AUDIO),
        str(OUT / "whisper_cpp23_hann_f32.bin"),
        str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
        str(OUT / "whisper_cpp23_token_manifest.tsv"),
        str(OUT / "whisper_cpp23_token_bytes.bin"),
        str(beams),
        str(groups),
        str(returned),
        str(maximum),
        str(length_penalty),
        str(diversity_penalty),
        cpp_stopping,
    ]
    line = subprocess.check_output(command, text=True).strip()
    match = re.fullmatch(
        r"WHISPER_CPP23_GROUP_BEAM_SEARCH beams=(\d+) groups=(\d+) "
        r"returned=(\d+) sequences=([^ ]+) scores=([^ ]+) "
        r"expanded_candidates=(\d+) cache_branches=(\d+) "
        r"graph_nodes_visited=(\d+)",
        line,
    )
    assert match, line
    actual_sequences = [
        [int(token) for token in sequence.split(",")]
        for sequence in match.group(4).split(";")
    ]
    actual_scores = [float(score) for score in match.group(5).split(",")]
    expected_sequences = expected.sequences.tolist()
    expected_scores = expected.sequences_scores.tolist()
    score_errors = [
        abs(actual - reference)
        for actual, reference in zip(actual_scores, expected_scores)
    ]
    assert int(match.group(1)) == beams
    assert int(match.group(2)) == groups
    assert int(match.group(3)) == returned
    assert actual_sequences == expected_sequences, (
        name,
        actual_sequences,
        expected_sequences,
    )
    assert len(actual_scores) == len(expected_scores) == returned
    assert max(score_errors) < 3e-4, (
        name,
        actual_scores,
        expected_scores,
        score_errors,
    )
    assert int(match.group(6)) > 0
    assert int(match.group(7)) > 0
    assert int(match.group(8)) == 74
    rows.append(
        {
            "case": name,
            "num_beams": beams,
            "num_beam_groups": groups,
            "num_return_sequences": returned,
            "maximum_positions": maximum,
            "length_penalty": length_penalty,
            "diversity_penalty": diversity_penalty,
            "early_stopping": stopping,
            "sequences": actual_sequences,
            "cpp23_sequence_scores": actual_scores,
            "reference_sequence_scores": expected_scores,
            "maximum_sequence_score_error": max(score_errors),
            "all_ranked_sequences_exact": True,
            "expanded_candidates": int(match.group(6)),
            "cache_branches": int(match.group(7)),
            "graph_nodes_visited": int(match.group(8)),
            "constructor": "Selection.DiverseGroupBeamSearch",
        }
    )

artifact = {
    "certificate": "WHISPER_CPP23_DIVERSE_GROUP_BEAM_SEARCH_1",
    "reference_repository": "transformers-community/group-beam-search",
    "reference_revision": REFERENCE_REVISION,
    "reference_generate_sha256": sha256(
        REFERENCE / "custom_generate/generate.py"
    ),
    "reference_beam_scorer_sha256": sha256(
        REFERENCE / "custom_generate/beam_search.py"
    ),
    "case_count": len(rows),
    "all_ranked_sequences_exact": all(
        row["all_ranked_sequences_exact"] for row in rows
    ),
    "worst_sequence_score_error": max(
        row["maximum_sequence_score_error"] for row in rows
    ),
    "all_graph_nodes_visited": all(row["graph_nodes_visited"] == 74 for row in rows),
    "cases": rows,
    "scope": (
        "Batch-one deterministic diverse group beam search with explicit "
        "per-group live/completed states, Hamming token transport, copied K/V "
        "branches, global final ranking, and three stopping modes."
    ),
}
(OUT / "whisper_cpp23_group_beam_search.json").write_text(
    json.dumps(artifact, indent=2) + "\n"
)
table = "\n".join(
    f"| `{row['case']}` | {row['num_beams']} | {row['num_beam_groups']} | "
    f"{row['num_return_sequences']} | {row['maximum_positions']} | "
    f"{row['diversity_penalty']} | {row['maximum_sequence_score_error']:.6g} | yes |"
    for row in rows
)
(OUT / "WHISPER_CPP23_GROUP_BEAM_SEARCH.md").write_text(
    f"""# Explicit C++23 diverse-group beam graph

The C++ graph partitions live hypotheses into typed groups. Each group owns branchable token/K/V states and a completed-hypothesis set. At every position, groups advance in order; a Hamming transport edge subtracts the configured penalty for tokens selected by earlier groups at that same position. Completion is tracked per group, then all group hypotheses are globally ranked.

The oracle is revision `{REFERENCE_REVISION}` of `transformers-community/group-beam-search`, whose compatible 4.57-era `generate.py` hash is `{artifact['reference_generate_sha256']}`. Repository head is intentionally not used because its Transformers-v5 cache call is incompatible with the pinned 4.57.3 model runtime.

| case | beams | groups | returned | max positions | diversity | max score error | ranked sequences exact |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

All `{len(rows)}` complete ranked sequence sets match exactly. Worst normalized score error is `{artifact['worst_sequence_score_error']:.9g}`. This finite certificate covers deterministic batch-one grouped beam search; constrained and sampled beam variants remain separate algorithms.
"""
)
print(
    json.dumps(
        {
            key: artifact[key]
            for key in (
                "certificate",
                "case_count",
                "all_ranked_sequences_exact",
                "worst_sequence_score_error",
                "all_graph_nodes_visited",
            )
        },
        indent=2,
    )
)
