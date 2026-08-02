#!/usr/bin/env python3
"""Verify return_legacy_cache as a pure public-output projection."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL = ROOT / "work/whisper_tiny_en"
AUDIO = ROOT / "work/whisper_sample.wav"
CPP = ROOT / "work/whisper_graph_cpp23"
PATTERN = re.compile(
    r"WHISPER_CPP23_GENERATION_CACHE representation=([^ ]+) "
    r"tokens=([0-9,]+) layers=(\d+) self_positions=(\d+) "
    r"cross_positions=(\d+) cache_floats=(\d+) graph_nodes_visited=(\d+)"
)


def legacy_array(cache) -> np.ndarray:
    legacy = cache if isinstance(cache, tuple) else cache.to_legacy_cache()
    assert len(legacy) == 4 and all(len(layer) == 4 for layer in legacy)
    return np.concatenate(
        [
            tensor[0]
            .permute(1, 0, 2)
            .contiguous()
            .detach()
            .cpu()
            .numpy()
            .astype("<f4", copy=False)
            .reshape(-1)
            for layer in legacy
            for tensor in layer
        ]
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

python_outputs = {}
for value in (None, False, True):
    with torch.inference_mode():
        output = model.generate(
            inputs.input_features,
            attention_mask=inputs.attention_mask,
            max_new_tokens=2,
            use_cache=True,
            return_dict_in_generate=True,
            return_legacy_cache=value,
        )
    python_outputs[value] = output

assert type(python_outputs[None].past_key_values).__name__ == "EncoderDecoderCache"
assert type(python_outputs[False].past_key_values).__name__ == "EncoderDecoderCache"
assert isinstance(python_outputs[True].past_key_values, tuple)
expected_object = legacy_array(python_outputs[False].past_key_values)
expected_legacy = legacy_array(python_outputs[True].past_key_values)
assert np.array_equal(expected_object, expected_legacy)
expected_tokens = python_outputs[True].sequences[0].tolist()

common = [
    str(MODEL / "model.safetensors"),
    str(OUT / "whisper_cpp23_tensor_manifest.tsv"),
    str(AUDIO),
    str(OUT / "whisper_cpp23_hann_f32.bin"),
    str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
    "4",
]
cpp_rows = []
with tempfile.TemporaryDirectory(prefix="whisper-cache-projection-") as temporary:
    temporary = Path(temporary)
    arrays = {}
    for encoded, expected_name, expected in (
        ("object", "EncoderDecoderCache", expected_object),
        ("legacy", "legacy_four_tuple_per_layer", expected_legacy),
    ):
        path = temporary / f"{encoded}.bin"
        line = subprocess.check_output(
            [str(CPP), "--generation-cache-projection", *common, encoded, str(path)],
            text=True,
        ).strip()
        match = PATTERN.fullmatch(line)
        assert match, line
        actual = np.fromfile(path, dtype="<f4")
        arrays[encoded] = actual
        maximum = float(
            np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))
        )
        assert match.group(1) == expected_name
        assert [int(token) for token in match.group(2).split(",")] == expected_tokens
        assert [int(match.group(i)) for i in range(3, 8)] == [
            4,
            3,
            1500,
            expected.size,
            74,
        ]
        assert actual.size == expected.size and maximum < 3.0e-3
        cpp_rows.append(
            {
                "representation": expected_name,
                "tokens": expected_tokens,
                "layers": 4,
                "self_positions": 3,
                "cross_positions": 1500,
                "cache_floats": int(actual.size),
                "max_absolute_cache_error": maximum,
                "graph_nodes_visited": 74,
            }
        )
    assert np.array_equal(arrays["object"], arrays["legacy"])
    invalid = subprocess.run(
        [
            str(CPP),
            "--generation-cache-projection",
            *common,
            "unknown",
            str(temporary / "invalid.bin"),
        ],
        text=True,
        capture_output=True,
    )
    assert invalid.returncode != 0 and "generation cache representation" in invalid.stderr

with torch.inference_mode():
    tensor_only = model.generate(
        inputs.input_features,
        attention_mask=inputs.attention_mask,
        max_new_tokens=2,
        return_dict_in_generate=False,
        return_legacy_cache=True,
    )
assert isinstance(tensor_only, torch.Tensor)

artifact = {
    "certificate": "WHISPER_CPP23_GENERATION_CACHE_PROJECTION_1",
    "transformers_version": "4.57.3",
    "python_none_representation": "EncoderDecoderCache",
    "python_false_representation": "EncoderDecoderCache",
    "python_true_representation": "legacy_four_tuple_per_layer",
    "tensor_only_has_no_public_cache": True,
    "sequences_exact": True,
    "python_projection_values_bitwise_equal": True,
    "cpp23_projection_values_bitwise_equal": True,
    "invalid_representation_rejected": True,
    "worst_max_absolute_cache_error": max(
        row["max_absolute_cache_error"] for row in cpp_rows
    ),
    "cases": cpp_rows,
    "scope": (
        "Finite two-token generation on one recording. return_legacy_cache is "
        "verified as a container projection over identical four-layer self/cross tensors."
    ),
}
OUT.mkdir(exist_ok=True)
(OUT / "whisper_cpp23_cache_projection.json").write_text(
    json.dumps(artifact, indent=2) + "\n"
)
(OUT / "WHISPER_CPP23_CACHE_PROJECTION.md").write_text(
    f"""# Generation cache output projection

Transformers 4.57.3 returns an `EncoderDecoderCache` when `return_legacy_cache` is `None` or `False`, and a four-tensor tuple per decoder layer when it is `True`. The C++23 ADT represents both public forms over one internal cache recurrence.

Both C++23 projections contain {cpp_rows[0]['cache_floats']:,} binary32 values across four layers, three self-attention positions, and 1,500 cross-attention positions. Their bytes are identical. Compared with the corresponding PyTorch tensors, the worst maximum absolute error is `{artifact['worst_max_absolute_cache_error']:.9g}`. Generated token sequences match exactly, and an unknown representation is rejected.

This proves a finite output-container correspondence; it does not change or broaden the numerical backend proof boundary.
"""
)
print(
    json.dumps(
        {
            key: artifact[key]
            for key in (
                "certificate",
                "sequences_exact",
                "python_projection_values_bitwise_equal",
                "cpp23_projection_values_bitwise_equal",
                "worst_max_absolute_cache_error",
                "invalid_representation_rejected",
            )
        },
        indent=2,
    )
)
