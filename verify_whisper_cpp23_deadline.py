#!/usr/bin/env python3
"""Verify strict post-token max_time semantics and monotonic C++ execution."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import subprocess
import sys
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import (
    GenerationConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.generation.stopping_criteria import MaxTimeCriteria
from transformers.generation.utils import GenerationMixin

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
MODE = re.compile(
    r"WHISPER_CPP23_DEADLINE_MODE mode=([^ ]+) tokens=([^ ]*) "
    r"terminated_by_deadline=([01]) deadline_checks=(\d+) "
    r"graph_nodes_visited=(\d+)"
)
REFERENCES = {
    "contrastive": {
        "repository": "transformers-community/contrastive-search",
        "revision": "89ece6d21c47e6187e86d45d98fd495feadb33cb",
        "directory": ROOT / "work/contrastive_search_reference",
        "files": {
            "custom_generate/generate.py":
                "ea33addf7128014a238f3210abba3df7f8343acfb3880383f38ce5b34881c3d3"
        },
    },
    "group_beam": {
        "repository": "transformers-community/group-beam-search",
        "revision": "1a281620f7c5fa711c6a44d61c42a4e3a9c2098b",
        "directory": ROOT / "work/group_beam_reference_4_57",
        "files": {
            "custom_generate/generate.py":
                "07cb918df0a9298b89debb926b672bf8fd688cc2e66ba97a756cd04c12d02b42",
            "custom_generate/beam_search.py":
                "b55a2e9c65c357391eb78cc37bf0ecaeb89034b4dc2789a2d9c7516f4dd803a4",
        },
    },
    "constrained_beam": {
        "repository": "transformers-community/constrained-beam-search",
        "revision": "57fb32700aa9933f2e5077030f479d4931e56267",
        "directory": ROOT / "work/constrained_beam_reference_4_57",
        "files": {
            "custom_generate/generate.py":
                "176b15cece1977680c24e5847b63185f74c96a2cc25be79d1976f7bd185415bf",
            "custom_generate/beam_search.py":
                "247baefee037b9cebda29140abcc298ed2d4fdb32b79a40550a06b004a03bda5",
            "custom_generate/beam_constraints.py":
                "72da73e8f601167c895974bbb6064f1be7d14f33247f6a8ea0acc6c162b62704",
        },
    },
}


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


def common_configuration() -> GenerationConfig:
    config = GenerationConfig.from_model_config(model.config)
    config.max_length = 8
    config.max_time = -1.0
    config.eos_token_id = 50256
    config.pad_token_id = 50256
    config.decoder_start_token_id = 50257
    config.suppress_tokens = model.generation_config.suppress_tokens
    config.begin_suppress_tokens = model.generation_config.begin_suppress_tokens
    return config


prefix = torch.tensor([[50257, 50362]], dtype=torch.long)
expected_modes = {}
for mode, updates in (
    ("beam", {"num_beams": 2}),
    ("sampled_beam", {"num_beams": 2, "do_sample": True}),
    (
        "prompt_lookup",
        {"prompt_lookup_num_tokens": 3, "max_matching_ngram_size": 2},
    ),
):
    config = common_configuration()
    for field, value in updates.items():
        setattr(config, field, value)
    torch.manual_seed(11)
    with torch.inference_mode():
        output = GenerationMixin.generate(
            model,
            inputs=inputs.input_features,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=prefix,
            generation_config=config,
        )[0].tolist()
    expected_modes[mode] = output[len(prefix[0]) :]

for reference in REFERENCES.values():
    for filename, expected_hash in reference["files"].items():
        path = reference["directory"] / filename
        if not path.exists():
            hf_hub_download(
                reference["repository"],
                filename,
                revision=reference["revision"],
                local_dir=reference["directory"],
            )
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash

community_script = r"""
import json, sys, wave
from pathlib import Path
import numpy as np
import torch
from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor
mode, reference, root = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(reference))
from custom_generate.generate import generate
model = WhisperForConditionalGeneration.from_pretrained(root / "work/whisper_tiny_en", local_files_only=True, dtype=torch.float32).eval()
processor = WhisperProcessor.from_pretrained(root / "work/whisper_tiny_en", local_files_only=True)
with wave.open(str(root / "work/whisper_sample.wav"), "rb") as source:
    pcm = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
inputs = processor(pcm, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
prefix = torch.tensor([[50257, 50362]], dtype=torch.long)
config = GenerationConfig.from_model_config(model.config)
config.max_length = 8
config.max_time = -1.0
config.eos_token_id = 50256
config.pad_token_id = 50256
config.decoder_start_token_id = 50257
config.suppress_tokens = model.generation_config.suppress_tokens
config.begin_suppress_tokens = model.generation_config.begin_suppress_tokens
config.length_penalty = 1.0
config.early_stopping = False
if mode == "contrastive":
    config.top_k = 4
    config.penalty_alpha = 0.6
    config.low_memory = True
elif mode == "group_beam":
    config.num_beams = 4
    config.num_beam_groups = 2
    config.num_return_sequences = 1
    config.diversity_penalty = 0.5
elif mode == "constrained_beam":
    config.num_beams = 4
    config.num_return_sequences = 1
    config.force_words_ids = [[1770]]
else:
    raise RuntimeError(mode)
with torch.inference_mode():
    output = generate(model, inputs=inputs.input_features, attention_mask=inputs.attention_mask, decoder_input_ids=prefix, generation_config=config)
sequence = output.sequences[0].tolist() if hasattr(output, "sequences") else output[0].tolist()
print(json.dumps(sequence))
"""
for mode in ("contrastive", "group_beam", "constrained_beam"):
    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            community_script,
            mode,
            str(REFERENCES[mode]["directory"]),
            str(ROOT),
        ],
        text=True,
    )
    sequence = json.loads(output.splitlines()[-1])
    suffix = sequence[len(prefix[0]) :]
    if mode in {"group_beam", "constrained_beam"} and suffix[-1:] == [50256]:
        suffix.pop()
    expected_modes[mode] = suffix

mode_output = subprocess.check_output(
    [
        str(CPP),
        "--deadline-all-searches",
        str(MODEL / "model.safetensors"),
        str(OUT / "whisper_cpp23_tensor_manifest.tsv"),
        str(AUDIO),
        str(OUT / "whisper_cpp23_hann_f32.bin"),
        str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
    ],
    text=True,
)
mode_rows = []
for line in mode_output.splitlines():
    match = MODE.fullmatch(line)
    assert match, line
    mode = match.group(1)
    tokens = [] if not match.group(2) else [int(x) for x in match.group(2).split(",")]
    assert tokens == expected_modes[mode], (mode, tokens, expected_modes[mode])
    assert [int(match.group(index)) for index in range(3, 6)] == [1, 1, 74]
    mode_rows.append(
        {
            "mode": mode,
            "reference_tokens": expected_modes[mode],
            "cpp23_tokens": tokens,
            "terminated_by_deadline": True,
            "deadline_checks": 1,
            "graph_nodes_visited": 74,
            "tokens_exact": True,
        }
    )
assert {row["mode"] for row in mode_rows} == set(expected_modes)

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
    "search_mode_case_count": len(mode_rows),
    "all_transition_decisions_exact": True,
    "all_generation_tokens_exact": True,
    "all_graph_nodes_visited": True,
    "all_search_mode_tokens_exact": True,
    "all_search_modes_deadline_terminated": True,
    "negative_elapsed_rejected": True,
    "clock_contract": "C++ std::chrono::steady_clock; injected elapsed time for boundary tests",
    "transition_cases": transition_rows,
    "generation_cases": generation_rows,
    "search_mode_cases": mode_rows,
    "scope": (
        "Finite strict-boundary correspondence, three real-audio greedy runs, and "
        "expired-deadline finalization for every converted search interpreter. "
        "Ordinary searches check after a selected frontier transition; prompt "
        "lookup also checks the candidate prefix before target-model selection."
    ),
}
OUT.mkdir(exist_ok=True)
(OUT / "whisper_cpp23_deadline.json").write_text(
    json.dumps(artifact, indent=2) + "\n"
)
(OUT / "WHISPER_CPP23_DEADLINE.md").write_text(
    f"""# Monotonic generation deadline

Transformers 4.57.3 stops only when `elapsed > max_time`. Ordinary search loops evaluate `MaxTimeCriteria` after a selected transition, while assisted prompt lookup also evaluates it on the candidate prefix. The C++23 ADT preserves both placements while replacing wall-clock time with `std::chrono::steady_clock` so clock adjustments cannot extend or shorten generation.

All {len(transition_rows)} injected boundary cases agree exactly with the pinned Python criterion, including equality, just-over-deadline, zero, and negative-limit cases. All {len(generation_rows)} real-audio greedy expired-deadline runs produce the same single token (`1770`), check the deadline once, retain two prefix cache positions, and visit all 74 graph nodes.

All {len(mode_rows)} separately converted search interpreters also match their pinned expired-deadline references: standard beam, sampled beam, diverse-group beam, constrained beam, and contrastive search admit token `1770`; prompt lookup admits no target token because the assisted loop checks its candidate prefix first. Every interpreter records one deadline check and visits all 74 graph nodes.

This certificate covers finite deadlines. It remains finite source-pinned evidence, not a real-time scheduling proof for every machine load.
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
                "search_mode_case_count",
                "all_transition_decisions_exact",
                "all_generation_tokens_exact",
                "all_graph_nodes_visited",
                "all_search_mode_tokens_exact",
                "negative_elapsed_rejected",
            )
        },
        indent=2,
    )
)
