#!/usr/bin/env python3
"""Run the complete graph on Accelerate and the portable scalar backend."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
EXPECTED_TRANSCRIPT = (
    "Mr. Quilter is the apostle of the middle classes, and we are glad to "
    "welcome his gospel."
)
STAGE = re.compile(
    r"^(\w+) max_abs=([0-9.eE+-]+) rmse=([0-9.eE+-]+) "
    r"cosine=([0-9.eE+-]+)$"
)
WALL = re.compile(r"^\s*([0-9.]+) real\s", re.MULTILINE)
RSS = re.compile(r"^\s*(\d+)\s+maximum resident set size$", re.MULTILINE)
MARKER = "WHISPER_CPP23_WAV_TO_TEXT_CACHED_PROBABILISTIC_OK "


def run(binary: str, expected_backend: str) -> dict[str, object]:
    command = [
        "/usr/bin/time",
        "-l",
        str(ROOT / "work" / binary),
        str(ROOT / "work/whisper_tiny_en/model.safetensors"),
        str(OUT / "whisper_cpp23_tensor_manifest.tsv"),
        str(ROOT / "work/whisper_sample.wav"),
        str(OUT / "whisper_cpp23_mel_f32.bin"),
        str(OUT / "whisper_cpp23_hann_f32.bin"),
        str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
        str(OUT / "whisper_cpp23_encoder_reference_f32.bin"),
        str(OUT / "whisper_cpp23_decoder_ids_i32.bin"),
        str(OUT / "whisper_cpp23_decoder_reference_f32.bin"),
        str(OUT / "whisper_cpp23_token_manifest.tsv"),
        str(OUT / "whisper_cpp23_token_bytes.bin"),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    stages = []
    final = None
    for line in completed.stdout.splitlines():
        match = STAGE.fullmatch(line)
        if match:
            stages.append(
                {
                    "name": match.group(1),
                    "max_absolute_error": float(match.group(2)),
                    "rmse": float(match.group(3)),
                    "cosine": float(match.group(4)),
                }
            )
        elif line.startswith(MARKER):
            final = dict(
                field.split("=", 1)
                for field in shlex.split(line[len(MARKER) :])
            )
    wall = WALL.search(completed.stderr)
    rss = RSS.search(completed.stderr)
    assert final is not None and wall and rss
    assert len(stages) == 15
    assert final["backend"] == expected_backend
    assert final["transcript"] == EXPECTED_TRANSCRIPT
    assert int(final["graph_nodes"]) == int(final["graph_nodes_visited"]) == 74
    assert int(final["tensors"]) == 167
    assert int(final["generated_tokens"]) == int(final["sampled_tokens"]) == 22
    assert int(final["cache_positions"]) == 24
    assert max(stage["max_absolute_error"] for stage in stages) < 3.0e-3
    assert float(final["cache_logit_error"]) < 3.0e-3
    assert float(final["mass_sum_error"]) < 2.0e-6
    assert float(final["selected_mass_error"]) < 2.0e-4
    return {
        "backend": expected_backend,
        "wall_seconds": float(wall.group(1)),
        "peak_rss_bytes": int(rss.group(1)),
        "stage_count": len(stages),
        "worst_stage_max_absolute_error": max(
            stage["max_absolute_error"] for stage in stages
        ),
        "logit_max_absolute_error": next(
            stage["max_absolute_error"] for stage in stages if stage["name"] == "logits"
        ),
        "cache_logit_error": float(final["cache_logit_error"]),
        "mass_sum_error": float(final["mass_sum_error"]),
        "selected_mass_error": float(final["selected_mass_error"]),
        "graph_nodes_visited": int(final["graph_nodes_visited"]),
        "checkpoint_tensors_validated": int(final["tensors"]),
        "generated_tokens": int(final["generated_tokens"]),
        "sampled_tokens": int(final["sampled_tokens"]),
        "transcript": final["transcript"],
        "stages": stages,
    }


accelerate = run("whisper_graph_cpp23", "accelerate-cblas-f32")
portable = run("whisper_graph_cpp23_portable", "portable-scalar-f32")
artifact = {
    "certificate": "WHISPER_CPP23_PORTABLE_WHOLE_MODEL_1",
    "greedy_expected_token_assertion_passed": True,
    "seeded_sampling_probability_cache_checks_passed": True,
    "all_transcripts_exact": True,
    "all_graph_nodes_visited": True,
    "all_checkpoint_tensors_validated": True,
    "portable_numerical_threshold_passed": True,
    "portable_to_accelerate_wall_ratio": portable["wall_seconds"]
    / accelerate["wall_seconds"],
    "portable_to_accelerate_peak_rss_ratio": portable["peak_rss_bytes"]
    / accelerate["peak_rss_bytes"],
    "backends": [accelerate, portable],
    "scope": (
        "Finite whole-model differential run for the pinned Whisper Tiny English "
        "checkpoint and one LibriSpeech recording. Both binaries execute all 74 "
        "graph nodes, validate all 167 checkpoint tensors, and internally assert "
        "the complete greedy token sequence plus seeded-sampling probability and "
        "cache invariants."
    ),
}
OUT.mkdir(exist_ok=True)
(OUT / "whisper_cpp23_portable_model.json").write_text(
    json.dumps(artifact, indent=2) + "\n"
)
(OUT / "WHISPER_CPP23_PORTABLE_MODEL.md").write_text(
    f"""# Portable C++23 whole-model validation

Both numerical backends execute the complete 74-node graph, validate all 167 checkpoint tensors, pass the embedded greedy token assertion and seeded-sampling probability/cache checks, and produce the exact transcript.

| backend | wall time | peak RSS | worst stage max error | logit max error |
|---|---:|---:|---:|---:|
| Accelerate CBLAS binary32 | {accelerate['wall_seconds']:.2f} s | {accelerate['peak_rss_bytes'] / (1024**2):.1f} MiB | {accelerate['worst_stage_max_absolute_error']:.9g} | {accelerate['logit_max_absolute_error']:.9g} |
| portable scalar binary32 | {portable['wall_seconds']:.2f} s | {portable['peak_rss_bytes'] / (1024**2):.1f} MiB | {portable['worst_stage_max_absolute_error']:.9g} | {portable['logit_max_absolute_error']:.9g} |

On this run, the scalar backend took `{artifact['portable_to_accelerate_wall_ratio']:.2f}x` the wall time and `{artifact['portable_to_accelerate_peak_rss_ratio']:.3f}x` the peak resident memory of Accelerate. These are local measurements, not portable performance guarantees.

The portable run's worst stage maximum absolute error against the PyTorch fixtures is `{portable['worst_stage_max_absolute_error']:.9g}`. This is finite evidence for one checkpoint, compiler, machine, and recording; it is not a universal floating-point equivalence theorem.
"""
)
print(
    json.dumps(
        {
            "certificate": artifact["certificate"],
            "portable_wall_seconds": portable["wall_seconds"],
            "portable_peak_rss_bytes": portable["peak_rss_bytes"],
            "portable_worst_stage_max_absolute_error": portable[
                "worst_stage_max_absolute_error"
            ],
            "portable_logit_max_absolute_error": portable[
                "logit_max_absolute_error"
            ],
            "all_graph_nodes_visited": artifact["all_graph_nodes_visited"],
            "all_transcripts_exact": artifact["all_transcripts_exact"],
        },
        indent=2,
    )
)
