#!/usr/bin/env python3
"""Reproducible fresh-process benchmark of the C++23 graph and Transformers."""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL = ROOT / "work/whisper_tiny_en"
AUDIO = ROOT / "work/whisper_sample.wav"
BINARY = ROOT / "work/whisper_graph_cpp23"
EXPECTED = "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."
RUNS = 5


def self_peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def transformer_worker() -> None:
    import numpy as np
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    total_start = time.perf_counter()
    with wave.open(str(AUDIO), "rb") as source:
        assert source.getframerate() == 16000 and source.getnchannels() == 1
        pcm = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768.0

    start = time.perf_counter()
    processor = WhisperProcessor.from_pretrained(MODEL, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.float32
    ).eval()
    load_seconds = time.perf_counter() - start

    start = time.perf_counter()
    inputs = processor(pcm, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    frontend_seconds = time.perf_counter() - start
    start = time.perf_counter()
    with torch.inference_mode():
        ids = model.generate(
            inputs.input_features,
            attention_mask=inputs.attention_mask,
            max_new_tokens=128,
        )
    generation_seconds = time.perf_counter() - start
    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    if text != EXPECTED:
        raise RuntimeError(f"Transformers transcript mismatch: {text!r}")
    result = {
        "implementation": "transformers_python",
        "model_load_seconds": load_seconds,
        "frontend_seconds": frontend_seconds,
        "generation_seconds": generation_seconds,
        "worker_total_seconds": time.perf_counter() - total_start,
        "peak_rss_bytes": self_peak_rss_bytes(),
        "token_count_including_specials": int(ids.numel()),
        "transcript": text,
    }
    print("BENCH_RESULT=" + json.dumps(result, separators=(",", ":")))


def compile_cpp() -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "work/venv/lib/python3.14/site-packages")}
    subprocess.run(
        [sys.executable, "audit_whisper_generation_extensions.py"],
        cwd=ROOT,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "c++", "-std=c++23", "-O3", "-DNDEBUG", "-Wall", "-Wextra", "-Wpedantic",
            "-Wno-deprecated-declarations", "whisper_graph_cpp23.cpp", "-framework", "Accelerate", "-lz",
            "-o", str(BINARY),
        ],
        cwd=ROOT,
        check=True,
    )


def cpp_command() -> list[str]:
    return [
        str(BINARY), "--transcribe", str(MODEL / "model.safetensors"),
        str(OUT / "whisper_cpp23_tensor_manifest.tsv"), str(AUDIO),
        str(OUT / "whisper_cpp23_hann_f32.bin"),
        str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
        str(OUT / "whisper_cpp23_token_manifest.tsv"),
        str(OUT / "whisper_cpp23_token_bytes.bin"),
    ]


def run_cpp_once() -> dict:
    start = time.perf_counter()
    proc = subprocess.run(cpp_command(), cwd=ROOT, text=True, capture_output=True, check=True)
    wall = time.perf_counter() - start
    line = next(line for line in proc.stdout.splitlines() if line.startswith("WHISPER_CPP23_TRANSCRIPT"))
    match = re.search(r"tokens=([^ ]+) cache_logit_error=([^ ]+) peak_rss_bytes=(\d+) graph_nodes_visited=(\d+) text=\"(.*)\"$", line)
    if not match:
        raise RuntimeError(f"cannot parse C++ result: {line}")
    text = match.group(5)
    if text != EXPECTED:
        raise RuntimeError(f"C++ transcript mismatch: {text!r}")
    return {
        "implementation": "cpp23_accelerate",
        "fresh_process_wall_seconds": wall,
        "peak_rss_bytes": int(match.group(3)),
        "graph_nodes_visited": int(match.group(4)),
        "generated_token_count": len(match.group(1).split(",")),
        "cache_logit_error_check_disabled": float(match.group(2)) == 0.0,
        "transcript": text,
    }


def run_transformers_once() -> dict:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "work/venv/lib/python3.14/site-packages")}
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, __file__, "--transformer-worker"],
        cwd=ROOT, env=env, text=True, capture_output=True, check=True,
    )
    wall = time.perf_counter() - start
    line = next(line for line in proc.stdout.splitlines() if line.startswith("BENCH_RESULT="))
    result = json.loads(line.split("=", 1)[1])
    result["fresh_process_wall_seconds"] = wall
    return result


def summarize(rows: list[dict]) -> dict:
    result = {
        "runs": len(rows),
        "fresh_process_wall_seconds": {
            "median": statistics.median(row["fresh_process_wall_seconds"] for row in rows),
            "minimum": min(row["fresh_process_wall_seconds"] for row in rows),
            "maximum": max(row["fresh_process_wall_seconds"] for row in rows),
        },
        "peak_rss_bytes": {
            "median": int(statistics.median(row["peak_rss_bytes"] for row in rows)),
            "minimum": min(row["peak_rss_bytes"] for row in rows),
            "maximum": max(row["peak_rss_bytes"] for row in rows),
        },
        "transcript": rows[0]["transcript"],
        "all_transcripts_exact": all(row["transcript"] == EXPECTED for row in rows),
    }
    for key in ("model_load_seconds", "frontend_seconds", "generation_seconds", "worker_total_seconds"):
        if key in rows[0]:
            result[key] = {
                "median": statistics.median(row[key] for row in rows),
                "minimum": min(row[key] for row in rows),
                "maximum": max(row[key] for row in rows),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer-worker", action="store_true")
    args = parser.parse_args()
    if args.transformer_worker:
        transformer_worker()
        return

    compile_cpp()
    cpp_rows = []
    transformer_rows = []
    for _ in range(RUNS):
        cpp_rows.append(run_cpp_once())
        transformer_rows.append(run_transformers_once())

    cpp = summarize(cpp_rows)
    transformers = summarize(transformer_rows)
    speedup = transformers["fresh_process_wall_seconds"]["median"] / cpp["fresh_process_wall_seconds"]["median"]
    rss_ratio = transformers["peak_rss_bytes"]["median"] / cpp["peak_rss_bytes"]["median"]
    report = {
        "benchmark": "WHISPER-CPP23-VS-TRANSFORMERS-1",
        "measurement_date": time.strftime("%Y-%m-%d"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "sample": str(AUDIO.relative_to(ROOT)),
        "checkpoint": str((MODEL / "model.safetensors").relative_to(ROOT)),
        "checkpoint_bytes": (MODEL / "model.safetensors").stat().st_size,
        "cpp_binary_bytes": BINARY.stat().st_size,
        "protocol": "five alternating fresh processes; warm OS file cache is not flushed; wall time includes process startup and model loading; peak RSS is whole-process high-water mark",
        "cpp23": cpp,
        "transformers": transformers,
        "comparison": {
            "cpp23_fresh_process_speedup": speedup,
            "transformers_to_cpp23_peak_rss_ratio": rss_ratio,
            "cpp23_peak_rss_reduction_percent": 100.0 * (1.0 - 1.0 / rss_ratio),
        },
        "raw_runs": {"cpp23": cpp_rows, "transformers": transformer_rows},
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "whisper_cpp23_vs_transformers_benchmark.json").write_text(json.dumps(report, indent=2) + "\n")

    def seconds(summary: dict) -> str:
        return f"{summary['median']:.3f} s ({summary['minimum']:.3f}–{summary['maximum']:.3f})"

    def mib(value: int) -> float:
        return value / 2**20

    markdown = f"""# C++23 graph versus Transformers: speed and memory

Both paths use the same `{(MODEL / 'model.safetensors').stat().st_size / 2**20:.1f} MiB` Whisper Tiny English binary32 checkpoint and the same `{AUDIO.name}` waveform. Every one of the {RUNS} + {RUNS} runs returned exactly:

> {EXPECTED}

| implementation | fresh-process wall time, median (range) | peak RSS, median (range) |
|---|---:|---:|
| Explicit C++23 graph, Accelerate | {seconds(cpp['fresh_process_wall_seconds'])} | {mib(cpp['peak_rss_bytes']['median']):.1f} MiB ({mib(cpp['peak_rss_bytes']['minimum']):.1f}–{mib(cpp['peak_rss_bytes']['maximum']):.1f}) |
| Hugging Face Transformers, Python/PyTorch | {seconds(transformers['fresh_process_wall_seconds'])} | {mib(transformers['peak_rss_bytes']['median']):.1f} MiB ({mib(transformers['peak_rss_bytes']['minimum']):.1f}–{mib(transformers['peak_rss_bytes']['maximum']):.1f}) |

For this sample and machine, the C++23 graph is **{speedup:.2f}× faster** end to end and the Transformers process uses **{rss_ratio:.2f}×** its peak memory; equivalently, C++23 reduces whole-process peak RSS by **{report['comparison']['cpp23_peak_rss_reduction_percent']:.1f}%**.

After Python/PyTorch/Transformers imports have completed inside the worker, the Transformers medians are model/processor loading `{transformers['model_load_seconds']['median']:.3f} s`, frontend `{transformers['frontend_seconds']['median']:.3f} s`, and autoregressive generation `{transformers['generation_seconds']['median']:.3f} s`. Their sum is much smaller than fresh-process wall time because importing the Python ML stack dominates startup. The C++ number is the complete executable from process launch through WAV parsing, lazy checkpoint loading, Mel computation, encoder, cached decoder, and byte-token decoding. A persistent already-loaded C++ mode was not measured, so the `{transformers['frontend_seconds']['median'] + transformers['generation_seconds']['median']:.3f} s` loaded Transformers pipeline must not be compared directly with the `{cpp['fresh_process_wall_seconds']['median']:.3f} s` fresh C++ process.

## Measurement boundary

- Each observation is a new process, but the operating-system file cache is deliberately not flushed. This is a fresh-process benchmark, not a cold-disk benchmark.
- Peak RSS is the process high-water mark. It includes runtime and library overhead, temporary activations, caches, and the checkpoint mapping/copy; it is not parameter bytes alone.
- The C++ production path uses only incremental decoder K/V caches. The separate regression path still compares cached logits against full causal-prefix recomputation; that expensive verification is excluded here.
- Both implementations retain the same checkpoint, so this benchmark demonstrates a smaller execution runtime, not compression of the model's information.
"""
    (OUT / "WHISPER_CPP23_VS_TRANSFORMERS_BENCHMARK.md").write_text(markdown)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
