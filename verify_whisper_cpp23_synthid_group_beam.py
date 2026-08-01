#!/usr/bin/env python3
"""Verify SynthID's shared within-group row state in diverse beam search."""

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
REFERENCE = ROOT / "work/group_beam_reference_4_57"
REFERENCE_REPOSITORY = "transformers-community/group-beam-search"
REFERENCE_REVISION = "1a281620f7c5fa711c6a44d61c42a4e3a9c2098b"
REFERENCE_FILES = ("custom_generate/generate.py", "custom_generate/beam_search.py")
PREFIX = torch.tensor([[50257, 50362]], dtype=torch.long)
DEFAULT_KEYS = [654, 400, 836, 123, 340, 443, 597, 160, 57]
logging.set_verbosity_error()

sys.path.insert(0, str(REFERENCE))
from custom_generate.generate import generate as reference_generate  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


CASES = [
    ("two_groups", 4, 2, 4, 8, 1.0, 0.5, 5, DEFAULT_KEYS, 1024, 0, 65536, False),
    ("three_groups", 6, 3, 6, 8, 1.0, 1.0, 3, [71, -9, 123], 32, 11, 4096, False),
    ("startup_skip", 4, 2, 4, 8, 0.7, 2.0, 4, [9, 8], 32, 1, 1024, True),
    ("repeated_empty", 4, 2, 4, 8, 1.0, 0.5, 1, [1], 4, 17, 257, False),
    ("three_rows_per_group", 6, 2, 4, 8, 1.0, 0.5, 2, [-1, -(2**63)], 0, -17, 31, False),
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


def configuration(beams, groups, returned, maximum, penalty, diversity):
    config = GenerationConfig.from_model_config(model.config)
    config.num_beams = beams
    config.num_beam_groups = groups
    config.num_return_sequences = returned
    config.max_length = maximum
    config.eos_token_id = 50256
    config.pad_token_id = 50256
    config.decoder_start_token_id = 50257
    config.return_dict_in_generate = True
    config.output_scores = True
    config.suppress_tokens = model.generation_config.suppress_tokens
    config.begin_suppress_tokens = model.generation_config.begin_suppress_tokens
    config.length_penalty = penalty
    config.diversity_penalty = diversity
    config.early_stopping = False
    return config


pattern = re.compile(
    r"WHISPER_CPP23_SYNTHID_GROUP_BEAM beams=(\d+) groups=(\d+) returned=(\d+) "
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
    (
        name,
        beams,
        groups,
        returned,
        maximum,
        penalty,
        diversity,
        ngram,
        keys,
        history,
        seed,
        table_size,
        skip,
    ) = case
    builtin_config = configuration(beams, groups, returned, maximum, penalty, diversity)
    builtin_config.watermarking_config = SynthIDTextWatermarkingConfig(
        ngram_len=ngram,
        keys=keys,
        context_history_size=history,
        sampling_table_seed=seed,
        sampling_table_size=table_size,
        skip_first_ngram_calls=skip,
        debug_mode=False,
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
        debug_mode=False,
    )
    recorded_config = configuration(beams, groups, returned, maximum, penalty, diversity)
    with torch.inference_mode():
        recorded = reference_generate(
            model,
            inputs=inputs.input_features,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=PREFIX,
            generation_config=recorded_config,
            logits_processor=[recorder],
        )
    assert recorded.sequences.tolist() == builtin.sequences.tolist(), name
    assert np.array_equal(
        recorded.sequences_scores.numpy(), builtin.sequences_scores.numpy()
    ), name

    line = subprocess.check_output(
        [
            str(CPP),
            "--group-beam-search-synthid",
            *common,
            str(beams),
            str(groups),
            str(returned),
            str(maximum),
            str(penalty),
            str(diversity),
            "heuristic",
            str(ngram),
            ",".join(map(str, keys)),
            str(history),
            str(seed),
            str(table_size),
            str(int(skip)),
            "0",
        ],
        text=True,
    ).strip()
    match = pattern.fullmatch(line)
    assert match, line
    actual_sequences = [
        [int(token) for token in sequence.split(",")]
        for sequence in match.group(4).split(";")
    ]
    actual_scores = [float(value) for value in match.group(5).split(",")]
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
    actual_hashes = [int(value) for value in match.group(7).split(",")]
    actual_repeated = [value == "1" for value in match.group(8)]
    actual_skipped = [value == "1" for value in match.group(9)]
    score_errors = [
        abs(actual - expected)
        for actual, expected in zip(actual_scores, expected_scores)
    ]
    assert actual_sequences == expected_sequences, (name, actual_sequences, expected_sequences)
    assert len(actual_scores) == len(expected_scores) == returned
    assert max(score_errors) < 1e-2, (
        name,
        actual_scores,
        expected_scores,
        score_errors,
    )
    assert int(match.group(6)) == len(recorder.trace)
    assert actual_hashes == expected_hashes, name
    assert actual_repeated == expected_repeated, name
    assert actual_skipped == expected_skipped, name
    assert int(match.group(12)) == 74
    rows.append(
        {
            "case": name,
            "num_beams": beams,
            "num_beam_groups": groups,
            "group_size": beams // groups,
            "processor_calls": len(recorder.trace),
            "flattened_row_states": len(expected_hashes),
            "ranked_sequences": actual_sequences,
            "maximum_sequence_score_error": max(score_errors),
            "all_ranked_sequences_exact": True,
            "all_context_hashes_exact": True,
            "all_repeated_decisions_exact": True,
            "all_skipped_decisions_exact": True,
            "graph_nodes_visited": int(match.group(12)),
            "constructor": (
                "Selection.DiverseGroupBeamSearch + "
                "WatermarkPolicy.SynthIDTextWatermark"
            ),
        }
    )

artifact = {
    "certificate": "WHISPER_CPP23_SYNTHID_GROUP_BEAM_SHARED_ROW_STATE_1",
    "reference_repository": REFERENCE_REPOSITORY,
    "reference_revision": REFERENCE_REVISION,
    "reference_source_sha256": {
        name: sha256(REFERENCE / name) for name in REFERENCE_FILES
    },
    "case_count": len(rows),
    "all_ranked_sequences_exact": all(row["all_ranked_sequences_exact"] for row in rows),
    "all_context_hashes_exact": all(row["all_context_hashes_exact"] for row in rows),
    "all_repeated_decisions_exact": all(row["all_repeated_decisions_exact"] for row in rows),
    "all_skipped_decisions_exact": all(row["all_skipped_decisions_exact"] for row in rows),
    "worst_sequence_score_error": max(row["maximum_sequence_score_error"] for row in rows),
    "all_graph_nodes_visited": all(row["graph_nodes_visited"] == 74 for row in rows),
    "cases": rows,
    "scope": (
        "Batch-one diverse-group beam search with one persistent SynthID state "
        "tensor shared by sequential group calls and indexed by within-group row."
    ),
}
(OUT / "whisper_cpp23_synthid_group_beam.json").write_text(
    json.dumps(artifact, indent=2) + "\n"
)
table = "\n".join(
    f"| `{row['case']}` | {row['num_beams']} | {row['num_beam_groups']} | "
    f"{row['group_size']} | {row['processor_calls']} | "
    f"{row['maximum_sequence_score_error']:.6g} | yes | yes |"
    for row in rows
)
(OUT / "WHISPER_CPP23_SYNTHID_GROUP_BEAM.md").write_text(
    f"""# SynthID state through diverse-group beam calls

The pinned implementation calls the same stateful SynthID processor once per group at each generated position. Its state therefore has `beams / groups` rows and is shared across groups: group 1 updates the state left by group 0, rather than owning an independent watermark state. The C++23 graph makes this unusual sequencing explicit and applies Hamming diversity after SynthID, matching processor-list order.

| case | beams | groups | state rows | processor calls | max score error | ranked sequences exact | state exact |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

All `{len(rows)}` ranked sequence tensors match. Every signed context hash, repetition decision, and startup-skip decision matches the pinned source. Worst normalized score error is `{artifact['worst_sequence_score_error']:.9g}`.
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
