#!/usr/bin/env python3
"""Verify SynthID row-state transport through constrained beam search."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import (
    GenerationConfig,
    SynthIDTextWatermarkingConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    logging,
)
from transformers.generation.logits_process import SynthIDTextWatermarkLogitsProcessor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL = ROOT / "work/whisper_tiny_en"
AUDIO = ROOT / "work/whisper_sample.wav"
CPP = ROOT / "work/whisper_graph_cpp23"
REFERENCE = ROOT / "work/constrained_beam_reference_4_57"
REFERENCE_REPOSITORY = "transformers-community/constrained-beam-search"
REFERENCE_REVISION = "57fb32700aa9933f2e5077030f479d4931e56267"
REFERENCE_FILES = (
    "custom_generate/generate.py",
    "custom_generate/beam_search.py",
    "custom_generate/beam_constraints.py",
)
PREFIX = torch.tensor([[50257, 50362]], dtype=torch.long)
DEFAULT_KEYS = [654, 400, 836, 123, 340, 443, 597, 160, 57]
logging.set_verbosity_error()

sys.path.insert(0, str(REFERENCE))
from custom_generate.generate import generate as reference_generate  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


CASES = [
    {
        "name": "single_token_phrase",
        "beams": 4,
        "returned": 4,
        "maximum": 12,
        "constraint": "p:25996",
        "force_words_ids": [[25996]],
        "synthid": (5, DEFAULT_KEYS, 1024, 0, 65536, False, False),
    },
    {
        "name": "multi_token_phrase",
        "beams": 4,
        "returned": 2,
        "maximum": 18,
        "constraint": "p:1976,37052",
        "force_words_ids": [[1976, 37052]],
        "synthid": (3, [71, -9, 123], 32, 11, 4096, False, False),
    },
    {
        "name": "startup_skip",
        "beams": 4,
        "returned": 2,
        "maximum": 12,
        "constraint": "p:25996",
        "force_words_ids": [[25996]],
        "synthid": (4, [9, 8], 32, 1, 1024, True, False),
    },
    {
        "name": "repeated_empty_context",
        "beams": 4,
        "returned": 2,
        "maximum": 12,
        "constraint": "d:25996|17180",
        "force_words_ids": [[[25996], [17180]]],
        "synthid": (1, [1], 4, 17, 257, False, False),
    },
    {
        "name": "signed_hash_no_history",
        "beams": 4,
        "returned": 2,
        "maximum": 12,
        "constraint": "p:10912",
        "force_words_ids": [[10912]],
        "synthid": (2, [-1, -(2**63)], 0, -17, 31, False, False),
    },
]


class RecordingSynthIDProcessor(SynthIDTextWatermarkLogitsProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace: list[dict] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        rows = input_ids.shape[0]
        if self.state is None:
            context = torch.zeros((rows, self.ngram_len - 1), dtype=torch.long)
            history = torch.zeros((rows, self.context_history_size), dtype=torch.long)
            call = 1
        else:
            context = torch.concat((self.state.context, input_ids[:, -1:]), dim=1)[
                :, 1:
            ]
            history = self.state.context_history.clone()
            call = self.state.num_calls + 1
        skipped_value = self.skip_first_ngram_calls and call < self.ngram_len
        if skipped_value:
            hashes = [0] * rows
            repeated = [False] * rows
        else:
            hashes_tensor = self.accumulate_hash(
                torch.ones(rows, dtype=torch.long), context
            )
            hashes = [int(value) for value in hashes_tensor]
            repeated = [
                bool(history.shape[1] and (history[row] == hashes[row]).any())
                for row in range(rows)
            ]
        output = super().__call__(input_ids, scores)
        self.trace.append(
            {
                "call": call,
                "rows": rows,
                "context_hashes": hashes,
                "repeated": repeated,
                "skipped": [skipped_value] * rows,
                "last_tokens": [int(value) for value in input_ids[:, -1]],
            }
        )
        return output


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


def configuration(case: dict) -> GenerationConfig:
    config = GenerationConfig.from_model_config(model.config)
    config.num_beams = case["beams"]
    config.num_return_sequences = case["returned"]
    config.max_length = case["maximum"]
    config.eos_token_id = 50256
    config.pad_token_id = 50256
    config.decoder_start_token_id = 50257
    config.return_dict_in_generate = True
    config.output_scores = True
    config.suppress_tokens = model.generation_config.suppress_tokens
    config.begin_suppress_tokens = model.generation_config.begin_suppress_tokens
    config.length_penalty = 1.0
    config.early_stopping = False
    config.force_words_ids = case["force_words_ids"]
    return config


pattern = re.compile(
    r"WHISPER_CPP23_SYNTHID_CONSTRAINED_BEAM beams=(\d+) returned=(\d+) "
    r"sequences=([^ ]+) scores=([^ ]+) calls=(\d+) "
    r"context_hashes=([-0-9,]*) repeated=([01]*) skipped=([01]*) "
    r"expanded_candidates=(\d+) cache_branches=(\d+) graph_nodes_visited=(\d+)"
)
common = [
    str(MODEL / "model.safetensors"),
    str(OUT / "whisper_cpp23_tensor_manifest.tsv"),
    str(AUDIO),
    str(OUT / "whisper_cpp23_hann_f32.bin"),
    str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
    str(OUT / "whisper_cpp23_token_manifest.tsv"),
    str(OUT / "whisper_cpp23_token_bytes.bin"),
]

rows = []
for case in CASES:
    ngram, keys, history, seed, table_size, skip, debug = case["synthid"]
    builtin_config = configuration(case)
    builtin_config.watermarking_config = SynthIDTextWatermarkingConfig(
        ngram_len=ngram,
        keys=keys,
        context_history_size=history,
        sampling_table_seed=seed,
        sampling_table_size=table_size,
        skip_first_ngram_calls=skip,
        debug_mode=debug,
    )
    with torch.inference_mode():
        builtin = reference_generate(
            model,
            inputs=inputs.input_features,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=PREFIX,
            generation_config=builtin_config,
        )

    recorder = RecordingSynthIDProcessor(
        ngram_len=ngram,
        keys=keys,
        context_history_size=history,
        sampling_table_seed=seed,
        sampling_table_size=table_size,
        device=torch.device("cpu"),
        skip_first_ngram_calls=skip,
        debug_mode=debug,
    )
    recorded_config = configuration(case)
    with torch.inference_mode():
        recorded = reference_generate(
            model,
            inputs=inputs.input_features,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=PREFIX,
            generation_config=recorded_config,
            logits_processor=[recorder],
        )
    assert recorded.sequences.tolist() == builtin.sequences.tolist(), case["name"]
    assert np.array_equal(
        recorded.sequences_scores.numpy(), builtin.sequences_scores.numpy()
    ), case["name"]

    encoded = [
        str(ngram),
        ",".join(map(str, keys)),
        str(history),
        str(seed),
        str(table_size),
        str(int(skip)),
        str(int(debug)),
    ]
    line = subprocess.check_output(
        [
            str(CPP),
            "--constrained-beam-search-synthid",
            *common,
            str(case["beams"]),
            str(case["returned"]),
            str(case["maximum"]),
            "1.0",
            "heuristic",
            case["constraint"],
            *encoded,
        ],
        text=True,
    ).strip()
    match = pattern.fullmatch(line)
    assert match, line
    actual_sequences = [
        [int(token) for token in sequence.split(",")]
        for sequence in match.group(3).split(";")
    ]
    actual_scores = [float(value) for value in match.group(4).split(",")]
    expected_sequences = recorded.sequences.tolist()
    expected_scores = recorded.sequences_scores.tolist()
    expected_hashes = [
        value for call in recorder.trace for value in call["context_hashes"]
    ]
    expected_repeated = [
        value for call in recorder.trace for value in call["repeated"]
    ]
    expected_skipped = [
        value for call in recorder.trace for value in call["skipped"]
    ]
    actual_hashes = [int(value) for value in match.group(6).split(",")]
    actual_repeated = [value == "1" for value in match.group(7)]
    actual_skipped = [value == "1" for value in match.group(8)]
    score_errors = [
        abs(actual - expected)
        for actual, expected in zip(actual_scores, expected_scores)
    ]
    assert actual_sequences == expected_sequences, case["name"]
    assert len(actual_scores) == len(expected_scores) == case["returned"]
    # A forced low-probability constraint edge magnifies the already measured
    # float32 SynthID probability error when converted back to log space.
    assert max(score_errors) < 1e-2, (case["name"], score_errors)
    assert int(match.group(5)) == len(recorder.trace)
    assert actual_hashes == expected_hashes, case["name"]
    assert actual_repeated == expected_repeated, case["name"]
    assert actual_skipped == expected_skipped, case["name"]
    assert int(match.group(11)) == 74
    rows.append(
        {
            "case": case["name"],
            "num_beams": case["beams"],
            "num_return_sequences": case["returned"],
            "maximum_positions": case["maximum"],
            "constraint": case["constraint"],
            "processor_calls": len(recorder.trace),
            "flattened_row_states": len(expected_hashes),
            "ranked_sequences": actual_sequences,
            "maximum_sequence_score_error": max(score_errors),
            "all_ranked_sequences_exact": True,
            "all_context_hashes_exact": True,
            "all_repeated_decisions_exact": True,
            "all_skipped_decisions_exact": True,
            "graph_nodes_visited": int(match.group(11)),
            "constructor": (
                "Selection.ConstrainedBeamSearch + "
                "WatermarkPolicy.SynthIDTextWatermark"
            ),
        }
    )

artifact = {
    "certificate": "WHISPER_CPP23_SYNTHID_CONSTRAINED_BEAM_ROW_STATE_1",
    "reference_repository": REFERENCE_REPOSITORY,
    "reference_revision": REFERENCE_REVISION,
    "reference_source_sha256": {
        name: sha256(REFERENCE / name) for name in REFERENCE_FILES
    },
    "case_count": len(rows),
    "all_ranked_sequences_exact": all(
        row["all_ranked_sequences_exact"] for row in rows
    ),
    "all_context_hashes_exact": all(row["all_context_hashes_exact"] for row in rows),
    "all_repeated_decisions_exact": all(
        row["all_repeated_decisions_exact"] for row in rows
    ),
    "all_skipped_decisions_exact": all(
        row["all_skipped_decisions_exact"] for row in rows
    ),
    "worst_sequence_score_error": max(
        row["maximum_sequence_score_error"] for row in rows
    ),
    "all_graph_nodes_visited": all(row["graph_nodes_visited"] == 74 for row in rows),
    "cases": rows,
    "scope": (
        "Batch-one deterministic constrained beam search with persistent SynthID "
        "state indexed by processor row slot, independently of constraint-bank and "
        "parent-hypothesis state transport."
    ),
}
(OUT / "whisper_cpp23_synthid_constrained_beam.json").write_text(
    json.dumps(artifact, indent=2) + "\n"
)
table = "\n".join(
    f"| `{row['case']}` | {row['num_beams']} | `{row['constraint']}` | "
    f"{row['processor_calls']} | {row['flattened_row_states']} | "
    f"{row['maximum_sequence_score_error']:.6g} | yes | yes |"
    for row in rows
)
(OUT / "WHISPER_CPP23_SYNTHID_CONSTRAINED_BEAM.md").write_text(
    f"""# SynthID state through constrained beam rows

The pinned constrained-search loop calls its logits processors on all beam rows before constraint-bank selection. The C++23 graph therefore keeps SynthID state by logical processor row, while token/K/V ancestry and constraint-machine progress follow selected parent edges. These are intentionally distinct state transports.

| case | beams | constraint | calls | row states | max score error | ranked sequences exact | state exact |
|---|---:|---|---:|---:|---:|---:|---:|
{table}

All `{len(rows)}` configurations reproduce the complete ranked sequence tensor. Every signed context hash, repeated-context decision, and startup-skip decision matches the source processor exactly. Worst normalized score error is `{artifact['worst_sequence_score_error']:.9g}`.

The oracle is revision `{REFERENCE_REVISION}` of `{REFERENCE_REPOSITORY}`. Its three relevant source files are hash-pinned in the JSON artifact.
"""
)
print(
    json.dumps(
        {
            key: artifact[key]
            for key in (
                "certificate",
                "case_count",
                "all_ranked_sequences_exact",
                "all_context_hashes_exact",
                "all_repeated_decisions_exact",
                "all_skipped_decisions_exact",
                "worst_sequence_score_error",
                "all_graph_nodes_visited",
            )
        },
        indent=2,
    )
)
