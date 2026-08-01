#!/usr/bin/env python3
"""Verify explicit C++23 beam-state transport against GenerationMixin."""

from __future__ import annotations

import json
import re
import subprocess
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import (
    GenerationConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    logging,
)
from transformers.generation.utils import GenerationMixin

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL = ROOT / "work/whisper_tiny_en"
AUDIO = ROOT / "work/whisper_sample.wav"
CPP = ROOT / "work/whisper_graph_cpp23"
logging.set_verbosity_error()

CASES = [
    ("two_beam_ranked_frontier", 2, 2, 8, 1.0, False, "heuristic"),
    ("four_beam_ranked_frontier", 4, 4, 8, 1.0, False, "heuristic"),
    ("two_beam_eos_finalization", 2, 1, 448, 1.0, False, "heuristic"),
    ("all_finished_early_stop", 2, 1, 448, 0.7, True, "all-finished"),
    ("canonical_never_stop", 2, 1, 448, 1.0, "never", "canonical"),
]

processor = WhisperProcessor.from_pretrained(MODEL, local_files_only=True)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL, local_files_only=True, dtype=torch.float32
).eval()
with wave.open(str(AUDIO), "rb") as source:
    pcm = (
        np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(
            np.float32
        )
        / 32768.0
    )
inputs = processor(
    pcm, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
)
decoder_prefix = torch.tensor([[50257, 50362]], dtype=torch.long)


def reference_configuration(
    beams: int, returned: int, maximum: int, penalty: float, stopping
) -> GenerationConfig:
    config = GenerationConfig.from_model_config(model.config)
    config.num_beams = beams
    config.num_return_sequences = returned
    config.max_length = maximum
    config.eos_token_id = 50256
    config.pad_token_id = 50256
    config.decoder_start_token_id = 50257
    config.return_dict_in_generate = True
    config.output_scores = True
    config.suppress_tokens = model.generation_config.suppress_tokens
    config.begin_suppress_tokens = model.generation_config.begin_suppress_tokens
    config.length_penalty = penalty
    config.early_stopping = stopping
    return config


rows = []
for name, beams, returned, maximum, penalty, stopping, cpp_stopping in CASES:
    with torch.inference_mode():
        expected = GenerationMixin.generate(
            model,
            inputs=inputs.input_features,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=decoder_prefix,
            generation_config=reference_configuration(
                beams, returned, maximum, penalty, stopping
            ),
        )
    command = [
        str(CPP),
        "--beam-search",
        str(MODEL / "model.safetensors"),
        str(OUT / "whisper_cpp23_tensor_manifest.tsv"),
        str(AUDIO),
        str(OUT / "whisper_cpp23_hann_f32.bin"),
        str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
        str(OUT / "whisper_cpp23_token_manifest.tsv"),
        str(OUT / "whisper_cpp23_token_bytes.bin"),
        str(beams),
        str(returned),
        str(maximum),
        str(penalty),
        cpp_stopping,
    ]
    line = subprocess.check_output(command, text=True).strip()
    match = re.fullmatch(
        r"WHISPER_CPP23_BEAM_SEARCH beams=(\d+) returned=(\d+) sequences=([^ ]+) scores=([^ ]+) expanded_candidates=(\d+) cache_branches=(\d+) graph_nodes_visited=(\d+)",
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
    assert int(match.group(1)) == beams and int(match.group(2)) == returned
    assert actual_sequences == expected_sequences, (
        name,
        actual_sequences,
        expected_sequences,
    )
    assert len(actual_scores) == len(expected_scores) == returned
    assert max(score_errors) < 2e-4, (
        name,
        actual_scores,
        expected_scores,
        score_errors,
    )
    assert (
        int(match.group(5)) > 0
        and int(match.group(6)) > 0
        and int(match.group(7)) == 74
    )
    rows.append(
        {
            "case": name,
            "num_beams": beams,
            "num_return_sequences": returned,
            "maximum_positions": maximum,
            "length_penalty": penalty,
            "early_stopping": stopping,
            "sequences": actual_sequences,
            "cpp23_sequence_scores": actual_scores,
            "transformers_sequence_scores": expected_scores,
            "maximum_sequence_score_error": max(score_errors),
            "all_ranked_sequences_exact": True,
            "expanded_candidates": int(match.group(5)),
            "cache_branches": int(match.group(6)),
            "graph_nodes_visited": 74,
            "constructor": "Selection.StandardBeamSearch",
        }
    )

artifact = {
    "certificate": "WHISPER_CPP23_STANDARD_BEAM_SEARCH_1",
    "case_count": len(rows),
    "all_ranked_sequences_exact": all(
        row["all_ranked_sequences_exact"] for row in rows
    ),
    "worst_sequence_score_error": max(
        row["maximum_sequence_score_error"] for row in rows
    ),
    "all_graph_nodes_visited": all(row["graph_nodes_visited"] == 74 for row in rows),
    "cases": rows,
    "scope": "Batch-one deterministic standard beam search with explicit copied K/V states, top-2B continuation frontier, EOS/max-length finalization, length penalties, and three early-stopping modes.",
}
OUT.mkdir(exist_ok=True)
(OUT / "whisper_cpp23_beam_search.json").write_text(
    json.dumps(artifact, indent=2) + "\n"
)
table = "\n".join(
    f"| `{row['case']}` | {row['num_beams']} | {row['num_return_sequences']} | {row['maximum_positions']} | {row['length_penalty']} | {row['cache_branches']} | {row['maximum_sequence_score_error']:.6g} | yes |"
    for row in rows
)
(OUT / "WHISPER_CPP23_BEAM_SEARCH.md").write_text(
    f"""# Explicit C++23 beam-state graph

Each live beam is represented by a token stack, cumulative log probability, parent lineage, and a branchable four-layer self/cross-attention cache. At each step the graph computes log-softmax mass, applies Whisper policy masks, constructs the global beam×vocabulary frontier, keeps the top `2B` candidates, separates completed from live states, copies only selected caches, and ranks completed hypotheses by length-normalized score.

| case | beams | returned | max positions | length penalty | cache branches | max score error | ranked sequences exact |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

All ranked token sequences match the pinned Transformers GenerationMixin beam implementation exactly. Worst normalized sequence-score error is `{artifact["worst_sequence_score_error"]:.9g}`. The four-beam finite frontier contains genuinely different alternatives, so this is not merely replaying greedy output. Scope is standard, deterministic, batch-one beam search; grouped, constrained, and sampled beam variants remain separate work.
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
