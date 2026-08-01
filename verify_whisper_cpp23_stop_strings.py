#!/usr/bin/env python3
"""Verify C++23 token-byte stop-string suffix-overlap semantics."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import wave
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor, logging
from transformers.generation.stopping_criteria import StopStringCriteria, StoppingCriteriaList
from transformers.generation.utils import GenerationMixin

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL = ROOT / "work/whisper_tiny_en"
AUDIO = ROOT / "work/whisper_sample.wav"
CPP = ROOT / "work/whisper_graph_cpp23"
STOPPING_SOURCE = Path(transformers.__file__).parent / "generation/stopping_criteria.py"
STOPPING_HASH = "495628e4c877f667fbbd9ed4cc83b680de9eac2abc97ec3903f6e9219af3ecd2"
CASES = [
    ("whole_token", [" apostle"]),
    ("three_token_span", ["ilter is"]),
    ("ends_inside_final_token", ["middle cl"]),
    ("starts_inside_previous_token", ["dle classes"]),
    ("multiple_alternatives", ["absent", "glad to"]),
    ("period_completion", ["gospel."]),
    ("no_match_eos", ["this string is absent"]),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if sha256(STOPPING_SOURCE) != STOPPING_HASH:
    raise RuntimeError(f"stopping-criteria source hash mismatch: {STOPPING_SOURCE}")

logging.set_verbosity_error()
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
prefix = torch.tensor([[50257, 50362]], dtype=torch.long)


def config() -> GenerationConfig:
    value = GenerationConfig.from_model_config(model.config)
    value.max_length = 448
    value.eos_token_id = 50256
    value.pad_token_id = 50256
    value.decoder_start_token_id = 50257
    value.suppress_tokens = model.generation_config.suppress_tokens
    value.begin_suppress_tokens = model.generation_config.begin_suppress_tokens
    return value


pattern = re.compile(
    r'WHISPER_CPP23_STOP_STRING tokens=([^ ]*) '
    r'terminated_by_stop_string=(\d) terminated_by_eos=(\d) '
    r'graph_nodes_visited=(\d+) text="(.*)"'
)
rows = []
for name, stops in CASES:
    criteria = StoppingCriteriaList(
        [StopStringCriteria(processor.tokenizer, stops)]
    )
    with torch.inference_mode():
        expected = GenerationMixin.generate(
            model,
            inputs=inputs.input_features,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=prefix,
            generation_config=config(),
            stopping_criteria=criteria,
        )[0].tolist()[2:]
    reference_eos = bool(expected and expected[-1] == 50256)
    if reference_eos:
        expected = expected[:-1]
    command = [
        str(CPP),
        "--stop-string-search",
        str(MODEL / "model.safetensors"),
        str(OUT / "whisper_cpp23_tensor_manifest.tsv"),
        str(AUDIO),
        str(OUT / "whisper_cpp23_hann_f32.bin"),
        str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
        str(OUT / "whisper_cpp23_token_manifest.tsv"),
        str(OUT / "whisper_cpp23_token_bytes.bin"),
        "448",
        "|".join(stops),
    ]
    line = subprocess.check_output(command, text=True).strip()
    match = pattern.fullmatch(line)
    assert match, line
    actual = [] if not match.group(1) else [int(token) for token in match.group(1).split(",")]
    assert actual == expected, (name, actual, expected)
    assert bool(int(match.group(2))) == (not reference_eos)
    assert bool(int(match.group(3))) == reference_eos
    assert int(match.group(4)) == 74
    expected_text = processor.tokenizer.decode(expected, skip_special_tokens=True)
    assert match.group(5) == expected_text, (name, match.group(5), expected_text)
    rows.append(
        {
            "case": name,
            "stop_strings": stops,
            "tokens": actual,
            "text": expected_text,
            "terminated_by_stop_string": not reference_eos,
            "terminated_by_eos": reference_eos,
            "complete_tokens_exact": True,
            "graph_nodes_visited": int(match.group(4)),
            "constructor": "StopStringSet",
        }
    )

artifact = {
    "certificate": "WHISPER_CPP23_STOP_STRING_AUTOMATON_1",
    "transformers_version": transformers.__version__,
    "stopping_criteria_source_sha256": STOPPING_HASH,
    "case_count": len(rows),
    "all_complete_tokens_exact": all(row["complete_tokens_exact"] for row in rows),
    "all_graph_nodes_visited": all(row["graph_nodes_visited"] == 74 for row in rows),
    "cases": rows,
    "scope": (
        "Batch-one greedy Whisper generation with one or more UTF-8 stop strings, "
        "including whole-token, cross-token, token-overhang, alternative, and EOS paths."
    ),
    "framework_boundary": (
        "Transformers 4.57.3 StopStringCriteria is invoked directly because its generic "
        "generate tokenizer kwarg leaks through _sample for this Whisper wrapper version."
    ),
}
OUT.mkdir(exist_ok=True)
(OUT / "whisper_cpp23_stop_strings.json").write_text(
    json.dumps(artifact, indent=2) + "\n"
)
table = "\n".join(
    f"| `{row['case']}` | `{row['stop_strings']}` | {len(row['tokens'])} | "
    f"{'stop' if row['terminated_by_stop_string'] else 'EOS'} | yes |"
    for row in rows
)
(OUT / "WHISPER_CPP23_STOP_STRINGS.md").write_text(
    f"""# Explicit token-byte stop-string automaton

The C++23 matcher stores a non-empty `StopStringSet`. After each selected token, it joins that token's exact byte-BPE payload to the previously decoded bytes and accepts when any stop string occurrence overlaps the newly selected token. This captures whole-token matches, strings crossing token boundaries, and the reference criterion's deliberate overhang behavior where a stop string ends inside the final indivisible token.

The oracle is Transformers `{transformers.__version__}` `StopStringCriteria`, source hash `{STOPPING_HASH}`.

| case | stops | emitted tokens | termination | tokens exact |
|---|---|---:|---|---:|
{table}

All `{len(rows)}` complete token sequences match. The `no_match_eos` case proves ordinary EOS remains distinct from stop-string acceptance.

This certificate currently covers greedy batch-one generation. The matcher itself is selection-independent, but plumbing it through every beam and long-form branch remains necessary before `stop_strings` can be called a full generic override. Transformers 4.57.3's tokenizer keyword leaks into the Whisper forward call on the direct configuration route, so the oracle invokes the same pinned `StopStringCriteria` directly.
"""
)
print(
    json.dumps(
        {
            key: artifact[key]
            for key in (
                "certificate",
                "case_count",
                "all_complete_tokens_exact",
                "all_graph_nodes_visited",
            )
        },
        indent=2,
    )
)
