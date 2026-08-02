#!/usr/bin/env python3
"""Verify strict post-token max_time semantics and monotonic C++ execution."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import subprocess
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.generation.stopping_criteria import MaxTimeCriteria

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL = ROOT / "work/whisper_tiny_en"
AUDIO = ROOT / "work/whisper_sample.wav"
CPP = ROOT / "work/whisper_graph_cpp23"
TRANSITION = re.compile(
    r"WHISPER_CPP23_DEADLINE_TRANSITION maximum_seconds=([^ ]+) "
    r"elapsed_seconds=([^ ]+) decision=([A-Z_]+)"
)
SEARCH = re.compile(
    r"WHISPER_CPP23_DEADLINE_SEARCH tokens=([0-9,]+) "
    r"terminated_by_deadline=([01]) deadline_checks=(\d+) "
    r"cache_positions=(\d+) graph_nodes_visited=(\d+)"
)


def cpp_transition(maximum: float, elapsed: float) -> str:
    line = subprocess.check_output(
        [str(CPP), "--deadline-transition", str(maximum), str(elapsed)],
        text=True,
    ).strip()
    match = TRANSITION.fullmatch(line)
    assert match, line
    return match.group(3)


transition_rows = []
for maximum, elapsed in (
    (1.0, 0.0),
    (1.0, 1.0),
    (1.0, 1.000001),
    (0.0, 0.0),
    (0.0, 0.000001),
    (-1.0, 0.0),
):
    criterion = MaxTimeCriteria(maximum, initial_timestamp=100.0)
    with patch(
        "transformers.generation.stopping_criteria.time.time",
        return_value=100.0 + elapsed,
    ):
        python_stop = bool(criterion(torch.zeros((1, 1), dtype=torch.long), None)[0])
    expected = "STOP_AFTER" if python_stop else "CONTINUE_AT_OR_BEFORE"
    actual = cpp_transition(maximum, elapsed)
    assert actual == expected
    transition_rows.append(
        {
            "maximum_seconds": maximum,
            "elapsed_seconds": elapsed,
            "python_stop": python_stop,
            "cpp23_decision": actual,
            "exact": True,
        }
    )

processor = WhisperProcessor.from_pretrained(MODEL, local_files_only=True)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL, local_files_only=True, dtype=torch.float32
).eval()
with wave.open(str(AUDIO), "rb") as audio:
    pcm = (
        np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
        .astype(np.float32)
        / 32768.0
    )
inputs = processor(
    pcm, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
)
common = [
    str(MODEL / "model.safetensors"),
    str(OUT / "whisper_cpp23_tensor_manifest.tsv"),
    str(AUDIO),
    str(OUT / "whisper_cpp23_hann_f32.bin"),
    str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
    "6",
]
generation_rows = []
for maximum in (-1.0, 0.0, 1.0e-9):
    with torch.inference_mode():
        python_tokens = model.generate(
            inputs.input_features,
            attention_mask=inputs.attention_mask,
            max_new_tokens=4,
            max_time=maximum,
        )[0].tolist()
    line = subprocess.check_output(
        [str(CPP), "--deadline-search", *common, str(maximum)], text=True
    ).strip()
    match = SEARCH.fullmatch(line)
    assert match, line
    cpp_tokens = [int(token) for token in match.group(1).split(",")]
    assert cpp_tokens == python_tokens == [1770]
    assert [int(match.group(index)) for index in range(2, 6)] == [1, 1, 2, 74]
    generation_rows.append(
        {
            "max_time_seconds": maximum,
            "python_tokens": python_tokens,
            "cpp23_tokens": cpp_tokens,
            "terminated_by_deadline": True,
            "deadline_checks": 1,
            "cache_positions": 2,
            "graph_nodes_visited": 74,
            "tokens_exact": True,
        }
    )

invalid = subprocess.run(
    [str(CPP), "--deadline-transition", "1", "-0.1"],
    text=True,
    capture_output=True,
)
assert invalid.returncode != 0 and "generation deadline domain" in invalid.stderr

source = inspect.getsource(MaxTimeCriteria)
artifact = {
    "certificate": "WHISPER_CPP23_MONOTONIC_DEADLINE_1",
    "transformers_version": "4.57.3",
    "max_time_criteria_sha256": hashlib.sha256(source.encode()).hexdigest(),
    "transition_case_count": len(transition_rows),
    "generation_case_count": len(generation_rows),
    "all_transition_decisions_exact": True,
    "all_generation_tokens_exact": True,
    "all_graph_nodes_visited": True,
    "negative_elapsed_rejected": True,
    "clock_contract": "C++ std::chrono::steady_clock; injected elapsed time for boundary tests",
    "transition_cases": transition_rows,
    "generation_cases": generation_rows,
    "scope": (
        "Finite strict-boundary correspondence plus three real-audio greedy runs. "
        "The C++ deadline is checked after token selection, uses a monotonic clock, "
        "and currently certifies the converted greedy generation path."
    ),
}
OUT.mkdir(exist_ok=True)
(OUT / "whisper_cpp23_deadline.json").write_text(
    json.dumps(artifact, indent=2) + "\n"
)
(OUT / "WHISPER_CPP23_DEADLINE.md").write_text(
    f"""# Monotonic generation deadline

Transformers 4.57.3 evaluates `MaxTimeCriteria` after each selected token and stops only when `elapsed > max_time`. The C++23 ADT preserves that strict boundary while replacing wall-clock time with `std::chrono::steady_clock` so clock adjustments cannot extend or shorten generation.

All {len(transition_rows)} injected boundary cases agree exactly with the pinned Python criterion, including equality, just-over-deadline, zero, and negative-limit cases. All {len(generation_rows)} real-audio expired-deadline runs produce the same single token (`1770`), check the deadline once, retain two prefix cache positions, and visit all 74 graph nodes.

This certificate covers finite deadlines on the greedy path. It does not claim deadline scheduling for the separately implemented beam, constrained, contrastive, or prompt-lookup algorithms.
"""
)
print(
    json.dumps(
        {
            key: artifact[key]
            for key in (
                "certificate",
                "transition_case_count",
                "generation_case_count",
                "all_transition_decisions_exact",
                "all_generation_tokens_exact",
                "all_graph_nodes_visited",
                "negative_elapsed_rejected",
            )
        },
        indent=2,
    )
)
