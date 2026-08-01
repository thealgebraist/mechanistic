#!/usr/bin/env python3
"""Verify explicit constrained-beam graph against the pinned HF module."""
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
REFERENCE = ROOT / "work/constrained_beam_reference_4_57"
REFERENCE_REPOSITORY = "transformers-community/constrained-beam-search"
REFERENCE_REVISION = "57fb32700aa9933f2e5077030f479d4931e56267"
REFERENCE_FILES = {
    "custom_generate/generate.py":
        "176b15cece1977680c24e5847b63185f74c96a2cc25be79d1976f7bd185415bf",
    "custom_generate/beam_search.py":
        "247baefee037b9cebda29140abcc298ed2d4fdb32b79a40550a06b004a03bda5",
    "custom_generate/beam_constraints.py":
        "72da73e8f601167c895974bbb6064f1be7d14f33247f6a8ea0acc6c162b62704",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


for filename, expected_hash in REFERENCE_FILES.items():
    required = REFERENCE / filename
    if not required.exists():
        hf_hub_download(
            REFERENCE_REPOSITORY,
            filename,
            revision=REFERENCE_REVISION,
            local_dir=REFERENCE,
        )
    if sha256(required) != expected_hash:
        raise RuntimeError(f"constrained-beam reference hash mismatch: {required}")

sys.path.insert(0, str(REFERENCE))
from custom_generate.beam_constraints import (  # noqa: E402
    DisjunctiveConstraint,
    PhrasalConstraint,
)
from custom_generate.generate import generate as reference_generate  # noqa: E402

logging.set_verbosity_error()

CASES = [
    {
        "name": "single_token_force_words",
        "beams": 4,
        "returned": 4,
        "maximum": 12,
        "penalty": 1.0,
        "stopping": False,
        "cpp_stopping": "heuristic",
        "force_words_ids": [[25996]],
        "cpp_constraints": "p:25996",
        "requirements": [[[25996]]],
    },
    {
        "name": "multi_token_phrase",
        "beams": 4,
        "returned": 4,
        "maximum": 18,
        "penalty": 1.0,
        "stopping": False,
        "cpp_stopping": "heuristic",
        "force_words_ids": [[1976, 37052]],
        "cpp_constraints": "p:1976,37052",
        "requirements": [[[1976, 37052]]],
    },
    {
        "name": "two_required_phrases",
        "beams": 6,
        "returned": 4,
        "maximum": 18,
        "penalty": 0.7,
        "stopping": False,
        "cpp_stopping": "heuristic",
        "force_words_ids": [[25996], [17180]],
        "cpp_constraints": "p:25996;p:17180",
        "requirements": [[[25996]], [[17180]]],
    },
    {
        "name": "two_required_phrases_completed",
        "beams": 6,
        "returned": 1,
        "maximum": 30,
        "penalty": 0.7,
        "stopping": False,
        "cpp_stopping": "heuristic",
        "force_words_ids": [[25996], [17180]],
        "cpp_constraints": "p:25996;p:17180",
        "requirements": [[[25996]], [[17180]]],
    },
    {
        "name": "single_token_disjunction",
        "beams": 4,
        "returned": 4,
        "maximum": 18,
        "penalty": 1.0,
        "stopping": False,
        "cpp_stopping": "heuristic",
        "force_words_ids": [[[25996], [17180]]],
        "cpp_constraints": "d:25996|17180",
        "requirements": [[[25996], [17180]]],
    },
    {
        "name": "multi_token_disjunction",
        "beams": 4,
        "returned": 1,
        "maximum": 18,
        "penalty": 1.0,
        "stopping": False,
        "cpp_stopping": "heuristic",
        "force_words_ids": [[[1976, 37052], [3504, 6097]]],
        "cpp_constraints": "d:1976,37052|3504,6097",
        "requirements": [[[1976, 37052], [3504, 6097]]],
    },
    {
        "name": "direct_constraint_objects",
        "beams": 4,
        "returned": 4,
        "maximum": 18,
        "penalty": 1.0,
        "stopping": False,
        "cpp_stopping": "heuristic",
        "constraints": [PhrasalConstraint([10912])],
        "cpp_constraints": "p:10912",
        "requirements": [[[10912]]],
    },
    {
        "name": "natural_phrase_eos_finalization",
        "beams": 4,
        "returned": 1,
        "maximum": 448,
        "penalty": 1.0,
        "stopping": "never",
        "cpp_stopping": "canonical",
        "constraints": [PhrasalConstraint([21443])],
        "cpp_constraints": "p:21443",
        "requirements": [[[21443]]],
    },
]

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


def contains(sequence: list[int], phrase: list[int]) -> bool:
    return any(
        sequence[start : start + len(phrase)] == phrase
        for start in range(len(sequence) - len(phrase) + 1)
    )


rows = []
for case in CASES:
    config = GenerationConfig.from_model_config(model.config)
    config.num_beams = case["beams"]
    config.num_return_sequences = case["returned"]
    config.max_length = case["maximum"]
    config.eos_token_id = 50256
    config.pad_token_id = 50256
    config.decoder_start_token_id = 50257
    config.return_dict_in_generate = True
    config.output_scores = True
    config.suppress_tokens = model.generation_config.suppress_tokens
    config.begin_suppress_tokens = model.generation_config.begin_suppress_tokens
    config.length_penalty = case["penalty"]
    config.early_stopping = case["stopping"]
    if "force_words_ids" in case:
        config.force_words_ids = case["force_words_ids"]
    else:
        config.constraints = case["constraints"]
    with torch.inference_mode():
        expected = reference_generate(
            model,
            inputs=inputs.input_features,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=decoder_prefix,
            generation_config=config,
        )

    command = [
        str(CPP),
        "--constrained-beam-search",
        str(MODEL / "model.safetensors"),
        str(OUT / "whisper_cpp23_tensor_manifest.tsv"),
        str(AUDIO),
        str(OUT / "whisper_cpp23_hann_f32.bin"),
        str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
        str(OUT / "whisper_cpp23_token_manifest.tsv"),
        str(OUT / "whisper_cpp23_token_bytes.bin"),
        str(case["beams"]),
        str(case["returned"]),
        str(case["maximum"]),
        str(case["penalty"]),
        case["cpp_stopping"],
        case["cpp_constraints"],
    ]
    line = subprocess.check_output(command, text=True).strip()
    match = re.fullmatch(
        r"WHISPER_CPP23_CONSTRAINED_BEAM_SEARCH beams=(\d+) returned=(\d+) "
        r"sequences=([^ ]+) scores=([^ ]+) expanded_candidates=(\d+) "
        r"cache_branches=(\d+) graph_nodes_visited=(\d+)",
        line,
    )
    assert match, line
    actual_sequences = [
        [int(token) for token in sequence.split(",")]
        for sequence in match.group(3).split(";")
    ]
    actual_scores = [float(score) for score in match.group(4).split(",")]
    expected_sequences = expected.sequences.tolist()
    expected_scores = expected.sequences_scores.tolist()
    score_errors = [
        abs(actual - reference)
        for actual, reference in zip(actual_scores, expected_scores)
    ]
    assert actual_sequences == expected_sequences, (
        case["name"],
        actual_sequences,
        expected_sequences,
    )
    assert len(actual_scores) == len(expected_scores) == case["returned"]
    assert max(score_errors) < 3e-4, (
        case["name"],
        actual_scores,
        expected_scores,
        score_errors,
    )
    assert int(match.group(5)) > 0
    assert int(match.group(6)) > 0
    assert int(match.group(7)) == 74
    satisfaction = [
        all(any(contains(sequence, phrase) for phrase in disjunction)
            for disjunction in case["requirements"])
        for sequence in actual_sequences
    ]
    rows.append(
        {
            "case": case["name"],
            "num_beams": case["beams"],
            "num_return_sequences": case["returned"],
            "maximum_positions": case["maximum"],
            "length_penalty": case["penalty"],
            "early_stopping": case["stopping"],
            "constraint_encoding": case["cpp_constraints"],
            "sequences": actual_sequences,
            "constraint_satisfied": satisfaction,
            "cpp23_sequence_scores": actual_scores,
            "reference_sequence_scores": expected_scores,
            "maximum_sequence_score_error": max(score_errors),
            "all_ranked_sequences_exact": True,
            "expanded_candidates": int(match.group(5)),
            "cache_branches": int(match.group(6)),
            "graph_nodes_visited": int(match.group(7)),
            "constructor": "Selection.ConstrainedBeamSearch",
        }
    )

artifact = {
    "certificate": "WHISPER_CPP23_CONSTRAINED_BEAM_SEARCH_1",
    "reference_repository": REFERENCE_REPOSITORY,
    "reference_revision": REFERENCE_REVISION,
    "reference_source_sha256": {
        filename: sha256(REFERENCE / filename) for filename in REFERENCE_FILES
    },
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
        "Batch-one deterministic constrained beam search with phrase and "
        "disjunctive-trie ADTs, replayed progress states, progress-bank "
        "interleaving, EOS eligibility, fallback finalization, and copied K/V states."
    ),
}
(OUT / "whisper_cpp23_constrained_beam_search.json").write_text(
    json.dumps(artifact, indent=2) + "\n"
)
table = "\n".join(
    f"| `{row['case']}` | {row['num_beams']} | {row['num_return_sequences']} | "
    f"{row['maximum_positions']} | `{row['constraint_encoding']}` | "
    f"{sum(row['constraint_satisfied'])}/{len(row['constraint_satisfied'])} | "
    f"{row['maximum_sequence_score_error']:.6g} | yes |"
    for row in rows
)
(OUT / "WHISPER_CPP23_CONSTRAINED_BEAM_SEARCH.md").write_text(
    f"""# Explicit C++23 constrained-beam graph

Positive constraints are algebraic values: `ForcedPhrase` or `ForcedDisjunction`. Every beam carries an implicit replayable constraint state consisting of completed constraints, one optional in-progress machine, pending constraints, a progress bank, and the next tokens that advance it. Ordinary top-probability edges and constraint-advance edges are merged, deduplicated, bank-interleaved, and then used to branch token and K/V state.

The oracle is revision `{REFERENCE_REVISION}` of `{REFERENCE_REPOSITORY}`. All three Python source files are hash-checked before execution.

| case | beams | returned | max positions | constraint | satisfying outputs | max score error | ranked tensor exact |
|---|---:|---:|---:|---|---:|---:|---:|
{table}

All `{len(rows)}` ranked tensors match exactly, including the reference fallback at short bounds where some returned beams cannot yet satisfy every constraint. Worst normalized score error is `{artifact['worst_sequence_score_error']:.9g}`. This finite certificate covers phrase constraints, multiple required phrases, disjunctions, direct constraint objects, maximum-length fallback, and canonical EOS finalization.
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
