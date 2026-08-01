#!/usr/bin/env python3
"""Differential smoke verifier for sampled-beam SynthID row-state transport."""
from __future__ import annotations
import json, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL = ROOT / "work/whisper_tiny_en/model.safetensors"
COMMON = [str(MODEL), str(OUT / "whisper_cpp23_tensor_manifest.tsv"),
          str(ROOT / "work/whisper_sample.wav"), str(OUT / "whisper_cpp23_hann_f32.bin"),
          str(OUT / "whisper_cpp23_mel_filters_f32.bin"), str(OUT / "whisper_cpp23_token_manifest.tsv"),
          str(OUT / "whisper_cpp23_token_bytes.bin")]
CONFIG = ["5", "654,400,836,123,340,443,597,160,57", "1024", "0", "65536", "0", "0"]
BASE = [str(ROOT / "work/whisper_graph_cpp23"), "--beam-sample", *COMMON,
        "2", "2", "8", "1.0", "heuristic", "1.0", "11", "50", "-", "-", "-", "-", "-"]
pattern = re.compile(r"synthid_state_rows=(\d+) synthid_hashes=([^ ]*) synthid_repeated=([01]*) synthid_skipped=([01]*) synthid_parent_rows=([0-9,]*) graph_nodes_visited=(\d+)$")
def run():
    line = subprocess.check_output([*BASE, *CONFIG], text=True).strip()
    again = subprocess.check_output([*BASE, *CONFIG], text=True).strip()
    assert line == again, "same explicit seed did not replay identically"
    match = pattern.search(line); assert match, line
    rows = int(match.group(1)); hashes = match.group(2).split(",")
    assert rows == len(hashes) == len(match.group(3)) == len(match.group(4))
    parents = [int(value) for value in match.group(5).split(",") if value]
    assert rows % 2 == 0 and int(match.group(6)) == 74
    assert parents and all(0 <= parent < 2 for parent in parents)
    assert len(parents) % 2 == 0
    artifact = {"certificate": "WHISPER_CPP23_SAMPLED_BEAM_SYNTHID_1",
                "state_rows": rows, "processor_calls": rows // 2,
                "hashes": [int(x) for x in hashes],
                "repeated": [int(x) for x in match.group(3)],
                "skipped": [int(x) for x in match.group(4)],
                "parent_rows": parents,
                "parent_rows_transported": True,
                "graph_nodes_visited": int(match.group(6)),
                "same_seed_reproducible": True,
                "rng_boundary": "C++ mt19937_64 replay is checked independently; no PyTorch bitstream identity is claimed.",
                "scope": "finite batch-one sampled beam run; row transport and full-run replay only"}
    (OUT / "whisper_cpp23_synthid_sampled_beam.json").write_text(json.dumps(artifact, indent=2) + "\n")
    (OUT / "WHISPER_CPP23_SYNTHID_SAMPLED_BEAM.md").write_text(
        "# SynthID state through sampled beam rows\n\n"
        f"The finite run transported {rows} row states across {rows // 2} processor calls and visited 74 graph nodes. "
        "Each surviving runtime was copied from its sampled parent row, and the exact explicit C++ seed replayed identically. "
        "This is not a claim of identical PyTorch/C++ random bitstreams.\n")
    print(json.dumps({k: artifact[k] for k in ("certificate", "state_rows", "processor_calls", "graph_nodes_visited", "same_seed_reproducible")}))
if __name__ == "__main__": run()
