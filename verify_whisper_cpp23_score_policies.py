#!/usr/bin/env python3
"""Verify deterministic GenerationConfig score policies on real Whisper audio."""

from __future__ import annotations

import json
import re
import subprocess
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, logging

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL = ROOT / "work/whisper_tiny_en"
AUDIO = ROOT / "work/whisper_sample.wav"
CPP = ROOT / "work/whisper_graph_cpp23"
logging.set_verbosity_error()

CASES = [
    (
        "single_token_sequence_bias",
        {"sequence_bias": [[[1770], -100.0]]},
        ["1770:-100", "-", "-", "0", "-", "0", "448"],
        "SequenceBiasPolicy.AdditiveSequenceBias",
    ),
    (
        "contextual_sequence_bias",
        {"sequence_bias": [[[50362, 1770], -100.0]]},
        ["50362,1770:-100", "-", "-", "0", "-", "0", "448"],
        "SequenceBiasPolicy.AdditiveSequenceBias",
    ),
    (
        "forced_bos_model_noop",
        {"forced_bos_token_id": 464},
        ["-", "464", "-", "0", "-", "0", "448"],
        "ForcedBeginningPolicy.ForcedBeginningToken",
    ),
    (
        "forced_eos_at_maximum",
        {"forced_eos_token_id": 50256, "max_length": 8},
        ["-", "-", "50256", "0", "-", "0", "8"],
        "ForcedEndingPolicy.ForcedEndingTokens",
    ),
    (
        "exponential_eos_decay",
        {"exponential_decay_length_penalty": (0, 2.0)},
        ["-", "-", "-", "0", "0,2", "0", "448"],
        "ExponentialEosPolicy.ExponentialEosDecay",
    ),
    (
        "repair_invalid_finite_model_path",
        {"remove_invalid_values": True},
        ["-", "-", "-", "1", "-", "0", "448"],
        "InvalidLogitPolicy.RepairInvalidLogits",
    ),
    (
        "renormalize_logits",
        {"renormalize_logits": True},
        ["-", "-", "-", "0", "-", "1", "448"],
        "LogitNormalizationPolicy.NormalizeLogProbabilities",
    ),
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

rows = []
for name, generation_kwargs, cpp_arguments, constructor in CASES:
    with torch.inference_mode():
        expected = model.generate(
            inputs.input_features,
            attention_mask=inputs.attention_mask,
            **generation_kwargs,
        )[0].tolist()
    command = [
        str(CPP),
        "--transcribe-score-policies",
        str(MODEL / "model.safetensors"),
        str(OUT / "whisper_cpp23_tensor_manifest.tsv"),
        str(AUDIO),
        str(OUT / "whisper_cpp23_hann_f32.bin"),
        str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
        str(OUT / "whisper_cpp23_token_manifest.tsv"),
        str(OUT / "whisper_cpp23_token_bytes.bin"),
        *cpp_arguments,
    ]
    line = subprocess.check_output(command, text=True).strip()
    match = re.fullmatch(
        r"WHISPER_CPP23_SCORE_POLICIES tokens=([0-9,]*) terminated_by_eos=([01]) graph_nodes_visited=(\d+)",
        line,
    )
    assert match, line
    actual = (
        []
        if not match.group(1)
        else [int(value) for value in match.group(1).split(",")]
    )
    assert actual == expected, (name, actual, expected)
    assert match.group(2) == "1" and int(match.group(3)) == 74
    rows.append(
        {
            "case": name,
            "generation_kwargs": generation_kwargs,
            "generated_token_ids": actual,
            "token_count": len(actual),
            "exact_transformers_token_match": True,
            "terminated_by_eos": True,
            "graph_nodes_visited": 74,
            "constructor": constructor,
        }
    )

artifact = {
    "certificate": "WHISPER_CPP23_DETERMINISTIC_SCORE_POLICIES_1",
    "case_count": len(rows),
    "all_token_sequences_exact": all(
        row["exact_transformers_token_match"] for row in rows
    ),
    "all_graph_nodes_visited": all(row["graph_nodes_visited"] == 74 for row in rows),
    "cases": rows,
    "scope": "Real-audio greedy generation under sequence bias, forced boundaries, exponential EOS decay, invalid-value repair on the finite model path, and final logit normalization.",
}
OUT.mkdir(exist_ok=True)
(OUT / "whisper_cpp23_score_policies.json").write_text(
    json.dumps(artifact, indent=2) + "\n"
)
table = "\n".join(
    f"| `{row['case']}` | `{row['constructor']}` | {row['token_count']} | yes |"
    for row in rows
)
(OUT / "WHISPER_CPP23_SCORE_POLICIES.md").write_text(
    f"""# Deterministic score-policy lowering

Seven named C++23 policy constructors were exercised on the real Whisper sample and compared with the corresponding Transformers GenerationConfig override.

| case | C++23 ADT constructor | output tokens | exact Transformers sequence |
|---|---|---:|---:|
{table}

The contextual sequence-bias case verifies suffix-sensitive state rather than only a vocabulary-wide token bias. Forced EOS and exponential decay alter termination at different positions. `forced_bos_token_id` is an intentional model-path no-op because Whisper Tiny English begins with two prepared decoder tokens, so Transformers' `cur_len == 1` condition is false; the C++ constructor retains the same conditional semantics. Invalid-value repair is executable but is an identity on this checkpoint because all raw model logits are finite. This is finite behavioral evidence, not universal floating-point equivalence.
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
                "all_graph_nodes_visited",
            )
        },
        indent=2,
    )
)
