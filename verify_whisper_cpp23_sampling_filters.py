#!/usr/bin/env python3
"""Compare complete first-step sampling distributions with Transformers."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
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
        "temperature",
        {"temperature": 0.7, "top_k": 0},
        [".7", "-", "-", "-", "-", "-", "-"],
    ),
    ("top_k", {"temperature": 1.0, "top_k": 20}, ["1", "20", "-", "-", "-", "-", "-"]),
    (
        "top_p",
        {"temperature": 1.0, "top_k": 0, "top_p": 0.9},
        ["1", "-", ".9", "-", "-", "-", "-"],
    ),
    (
        "min_p",
        {"temperature": 1.0, "top_k": 0, "min_p": 0.02},
        ["1", "-", "-", ".02", "-", "-", "-"],
    ),
    (
        "typical_p",
        {"temperature": 1.0, "top_k": 0, "typical_p": 0.9},
        ["1", "-", "-", "-", ".9", "-", "-"],
    ),
    (
        "epsilon",
        {"temperature": 1.0, "top_k": 0, "epsilon_cutoff": 0.0005},
        ["1", "-", "-", "-", "-", ".0005", "-"],
    ),
    (
        "eta",
        {"temperature": 1.0, "top_k": 0, "eta_cutoff": 0.001},
        ["1", "-", "-", "-", "-", "-", ".001"],
    ),
    (
        "composed",
        {
            "temperature": 1.2,
            "top_k": 100,
            "top_p": 0.98,
            "min_p": 0.0001,
            "typical_p": 0.99,
            "epsilon_cutoff": 1e-7,
            "eta_cutoff": 1e-7,
        },
        ["1.2", "100", ".98", ".0001", ".99", "1e-7", "1e-7"],
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
with tempfile.TemporaryDirectory(prefix="whisper-sampling-") as temporary:
    temporary = Path(temporary)
    for name, generation_kwargs, cpp_arguments in CASES:
        torch.manual_seed(1)
        with torch.inference_mode():
            output = model.generate(
                inputs.input_features,
                attention_mask=inputs.attention_mask,
                do_sample=True,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                force_unique_generate_call=True,
                **generation_kwargs,
            )
        expected = torch.softmax(output.scores[0][0], dim=-1).cpu().numpy()
        mass_path = temporary / f"{name}.bin"
        command = [
            str(CPP),
            "--sampling-mass",
            str(MODEL / "model.safetensors"),
            str(OUT / "whisper_cpp23_tensor_manifest.tsv"),
            str(AUDIO),
            str(OUT / "whisper_cpp23_hann_f32.bin"),
            str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
            str(OUT / "whisper_cpp23_token_manifest.tsv"),
            str(OUT / "whisper_cpp23_token_bytes.bin"),
            *cpp_arguments,
            str(mass_path),
        ]
        line = subprocess.check_output(command, text=True).strip()
        match = re.fullmatch(
            r"WHISPER_CPP23_SAMPLING_MASS support=(\d+) selected=(\d+) sum=([0-9.eE+-]+) graph_nodes_visited=(\d+)",
            line,
        )
        assert match, line
        actual = np.fromfile(mass_path, dtype="<f4")
        assert actual.shape == expected.shape == (model.config.vocab_size,)
        expected_support = expected > 0.0
        actual_support = actual > 0.0
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        maximum = float(difference.max())
        l1 = float(difference.sum())
        support_count = int(actual_support.sum())
        assert np.array_equal(actual_support, expected_support), name
        assert int(match.group(1)) == support_count
        assert int(match.group(2)) == int(expected.argmax()) == int(actual.argmax())
        assert abs(float(match.group(3)) - 1.0) < 1e-6
        assert int(match.group(4)) == 73
        assert maximum < 2e-5 and l1 < 3e-5, (name, maximum, l1)
        rows.append(
            {
                "case": name,
                "generation_kwargs": generation_kwargs,
                "support_count": support_count,
                "selected_token": int(actual.argmax()),
                "support_exact": True,
                "max_absolute_probability_error": maximum,
                "l1_probability_error": l1,
                "cpp23_probability_sum": float(actual.astype(np.float64).sum()),
                "graph_nodes_visited": 73,
            }
        )

artifact = {
    "certificate": "WHISPER_CPP23_SAMPLING_FILTER_DISTRIBUTIONS_1",
    "vocabulary_size": model.config.vocab_size,
    "case_count": len(rows),
    "all_supports_exact": all(row["support_exact"] for row in rows),
    "all_argmax_tokens_exact": True,
    "worst_max_absolute_probability_error": max(
        row["max_absolute_probability_error"] for row in rows
    ),
    "worst_l1_probability_error": max(row["l1_probability_error"] for row in rows),
    "cases": rows,
    "scope": "Complete first-step categorical distributions after Whisper suppression and Transformers-order sampling filters; this deliberately does not compare RNG streams.",
}
OUT.mkdir(exist_ok=True)
(OUT / "whisper_cpp23_sampling_filters.json").write_text(
    json.dumps(artifact, indent=2) + "\n"
)

table = "\n".join(
    f"| `{row['case']}` | {row['support_count']} | {row['selected_token']} | {row['max_absolute_probability_error']:.6g} | {row['l1_probability_error']:.6g} | yes |"
    for row in rows
)
(OUT / "WHISPER_CPP23_SAMPLING_FILTERS.md").write_text(
    f"""# C++23 sampling-filter probability equivalence

The runtime lowers temperature, top-k, nucleus/top-p, min-p, typical-p, epsilon, and eta filters into named C++23 probability transforms. Each row compares the complete `{model.config.vocab_size}`-token first-step categorical distribution with Transformers after the same Whisper suppression policy. Comparing distributions avoids conflating policy equivalence with unrelated PyTorch/C++ random-number generators.

| case | surviving support | argmax token | maximum probability error | L1 probability error | support exact |
|---|---:|---:|---:|---:|---:|
{table}

All supports and argmax tokens match exactly. Worst maximum probability error is `{artifact["worst_max_absolute_probability_error"]:.9g}` and worst L1 error is `{artifact["worst_l1_probability_error"]:.9g}`. This is a concrete finite equivalence check for one real Whisper state, not a universal floating-point theorem.
"""
)
print(
    json.dumps(
        {
            key: artifact[key]
            for key in (
                "certificate",
                "case_count",
                "all_supports_exact",
                "all_argmax_tokens_exact",
                "worst_max_absolute_probability_error",
                "worst_l1_probability_error",
            )
        },
        indent=2,
    )
)
