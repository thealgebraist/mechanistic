#!/usr/bin/env python3
"""Verify one C++23 audio batch against one true Transformers batch call."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL = ROOT / "work/whisper_tiny_en"
CPP = ROOT / "work/whisper_graph_cpp23"
AUDIO = [ROOT / "work/whisper_sample.wav"] + [
    OUT / "whisper_cpp23_multiaudio" / f"1272-128104-{index:04d}.wav"
    for index in (1, 2, 3, 10)
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_pcm(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        assert source.getframerate() == 16000
        assert source.getnchannels() == 1
        assert source.getsampwidth() == 2
        return (
            np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
            .astype(np.float32)
            / 32768.0
        )


for path in AUDIO:
    if not path.exists():
        raise RuntimeError(
            f"missing {path}; run verify_whisper_cpp23_multiaudio.py first"
        )

processor = WhisperProcessor.from_pretrained(MODEL, local_files_only=True)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL, local_files_only=True, dtype=torch.float32
).eval()
pcm = [read_pcm(path) for path in AUDIO]
inputs = processor(
    pcm,
    sampling_rate=16000,
    return_tensors="pt",
    return_attention_mask=True,
)
assert tuple(inputs.input_features.shape) == (len(AUDIO), 80, 3000)

start = time.perf_counter()
with torch.inference_mode():
    generated = model.generate(
        inputs.input_features,
        attention_mask=inputs.attention_mask,
        max_new_tokens=128,
    )
transformers_seconds = time.perf_counter() - start

pad_token = int(model.generation_config.pad_token_id)
expected_tokens: list[list[int]] = []
for row in generated.tolist():
    while row and row[-1] == pad_token:
        row.pop()
    expected_tokens.append(row)
expected_text = [
    text.strip()
    for text in processor.batch_decode(generated, skip_special_tokens=True)
]

command = [
    str(CPP),
    "--transcribe-batch",
    str(MODEL / "model.safetensors"),
    str(OUT / "whisper_cpp23_tensor_manifest.tsv"),
    str(OUT / "whisper_cpp23_hann_f32.bin"),
    str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
    str(OUT / "whisper_cpp23_token_manifest.tsv"),
    str(OUT / "whisper_cpp23_token_bytes.bin"),
    *(str(path) for path in AUDIO),
]
start = time.perf_counter()
completed = subprocess.run(
    command,
    cwd=ROOT,
    env={**os.environ, "WHISPER_VERIFY_RECOMPUTE": "1"},
    text=True,
    capture_output=True,
    check=True,
)
cpp23_seconds = time.perf_counter() - start

item_pattern = re.compile(
    r"^WHISPER_CPP23_BATCH_ITEM item=(\d+) tokens=([0-9,]*) "
    r"cache_logit_error=([0-9.eE+-]+) graph_nodes_visited=(\d+) "
    r"text=(\"(?:\\.|[^\"])*\")$"
)
items = []
for line in completed.stdout.splitlines():
    match = item_pattern.fullmatch(line)
    if not match:
        continue
    index = int(match.group(1))
    tokens = [int(value) for value in match.group(2).split(",") if value]
    text = json.loads(match.group(5))
    assert index == len(items)
    assert tokens == expected_tokens[index], (index, tokens, expected_tokens[index])
    assert text == expected_text[index], (index, text, expected_text[index])
    assert int(match.group(4)) == 74
    items.append(
        {
            "index": index,
            "audio": str(AUDIO[index].relative_to(ROOT)),
            "audio_sha256": sha256(AUDIO[index]),
            "samples": len(pcm[index]),
            "attention_mask_frames": int(inputs.attention_mask[index].sum()),
            "token_ids": tokens,
            "transcript": text,
            "cache_logit_error": float(match.group(3)),
            "graph_nodes_visited": int(match.group(4)),
            "exact_joint_batch_token_match": True,
            "exact_joint_batch_text_match": True,
        }
    )

summary_pattern = re.compile(
    r"^WHISPER_CPP23_BATCH_OK items=(\d+) shared_checkpoint=(\d+) "
    r"isolated_item_state=(\d+) peak_rss_bytes=(\d+)$"
)
summary = next(
    (
        summary_pattern.fullmatch(line)
        for line in completed.stdout.splitlines()
        if summary_pattern.fullmatch(line)
    ),
    None,
)
assert summary is not None, completed.stdout
assert len(items) == len(AUDIO) == int(summary.group(1))
assert int(summary.group(2)) == 1 and int(summary.group(3)) == 1

artifact = {
    "certificate": "WHISPER_CPP23_TRUE_BATCH_EXACT_MATCH",
    "batch_size": len(AUDIO),
    "transformers_call_count": 1,
    "transformers_input_shape": list(inputs.input_features.shape),
    "transformers_output_shape": list(generated.shape),
    "shared_checkpoint": True,
    "isolated_item_graph_state": True,
    "cpp23_execution_policy": "sequential items with shared immutable weights",
    "cpp23_vectorized_batch_linear_algebra": False,
    "all_token_sequences_exact": True,
    "all_transcripts_exact": True,
    "maximum_cache_logit_error": max(item["cache_logit_error"] for item in items),
    "transformers_generation_seconds": transformers_seconds,
    "cpp23_batch_seconds_with_cache_recomputation_checks": cpp23_seconds,
    "cpp23_peak_rss_bytes": int(summary.group(4)),
    "items": items,
    "scope": (
        "one five-item short-form English speech batch; proves ordering and "
        "independent state for these inputs, not arbitrary batch equivalence or "
        "vectorized-performance parity"
    ),
}
(OUT / "whisper_cpp23_batch.json").write_text(json.dumps(artifact, indent=2) + "\n")

rows = "\n".join(
    f"| {item['index']} | `{Path(item['audio']).name}` | "
    f"{item['attention_mask_frames']} | {len(item['token_ids'])} | "
    f"{item['cache_logit_error']:.3g} | {item['transcript']} |"
    for item in items
)
(OUT / "WHISPER_CPP23_BATCH.md").write_text(
    f"""# C++23 Whisper batch semantics

One Transformers invocation received a real tensor of shape `{list(inputs.input_features.shape)}` with five independently masked recordings. One C++23 process loaded the checkpoint and token vocabulary once, then executed five isolated item states in the same order. Every unpadded generated token and decoded transcript matched exactly.

| item | recording | valid feature frames | tokens | cached/full max error | transcript |
|---:|---|---:|---:|---:|---|
{rows}

The maximum incremental-cache versus full-prefix logit error was `{artifact['maximum_cache_logit_error']:.9g}`. Every item independently visited all 74 graph nodes. The C++ execution is deliberately described as **sequential semantic batching with shared immutable weights**, not vectorized batched matrix multiplication. Timings in the JSON are diagnostic only because the C++ run enabled expensive full-prefix cache recomputation while Transformers did not.

This finite certificate checks batch ordering, variable valid lengths, state isolation, exact tokens, and text for these five recordings. It is not a proof for every possible batch or waveform.
"""
)
print(
    json.dumps(
        {
            "certificate": artifact["certificate"],
            "batch_size": len(items),
            "all_tokens_exact": True,
            "all_text_exact": True,
            "max_cache_logit_error": artifact["maximum_cache_logit_error"],
        },
        indent=2,
    )
)
