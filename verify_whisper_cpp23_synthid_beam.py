#!/usr/bin/env python3
"""Verify explicit SynthID state transport through standard beam-row slots."""

from __future__ import annotations

import json
import re
import subprocess
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
from transformers.generation.utils import GenerationMixin

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL = ROOT / "work/whisper_tiny_en"
AUDIO = ROOT / "work/whisper_sample.wav"
CPP = ROOT / "work/whisper_graph_cpp23"
PREFIX = torch.tensor([[50257, 50362]], dtype=torch.long)
DEFAULT_KEYS = [654, 400, 836, 123, 340, 443, 597, 160, 57]
logging.set_verbosity_error()

CASES = [
    ("two_rows_short", 2, 2, 8, 5, DEFAULT_KEYS, 1024, 0, 65536, False, False),
    ("four_rows_short", 4, 4, 8, 5, DEFAULT_KEYS, 1024, 0, 65536, False, False),
    ("startup_skip_rows", 2, 2, 8, 4, [9, 8], 32, 1, 1024, True, False),
    ("repeated_empty_context_rows", 2, 2, 8, 1, [1], 4, 17, 257, False, False),
    ("signed_zero_history", 3, 3, 8, 2, [-1, -(2**63)], 0, -17, 31, False, False),
    ("eos_finalization", 2, 1, 30, 5, DEFAULT_KEYS, 1024, 0, 65536, False, False),
]


class RecordingSynthIDBeamProcessor(SynthIDTextWatermarkLogitsProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace: list[dict] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        batch = input_ids.shape[0]
        if self.state is None:
            context = torch.zeros((batch, self.ngram_len - 1), dtype=torch.long)
            history = torch.zeros((batch, self.context_history_size), dtype=torch.long)
            call = 1
        else:
            context = torch.concat((self.state.context, input_ids[:, -1:]), dim=1)[
                :, 1:
            ]
            history = self.state.context_history.clone()
            call = self.state.num_calls + 1
        skipped_value = self.skip_first_ngram_calls and call < self.ngram_len
        skipped = [skipped_value] * batch
        if skipped_value:
            hashes = [0] * batch
            repeated = [False] * batch
        else:
            hashes_tensor = self.accumulate_hash(
                torch.ones(batch, dtype=torch.long), context
            )
            hashes = [int(value) for value in hashes_tensor]
            repeated = [
                bool(history.shape[1] > 0 and (history[row] == hashes[row]).any())
                for row in range(batch)
            ]
        result = super().__call__(input_ids, scores)
        self.trace.append(
            {
                "call": call,
                "rows": batch,
                "context_hashes": hashes,
                "repeated": repeated,
                "skipped": skipped,
                "last_tokens": [int(value) for value in input_ids[:, -1]],
            }
        )
        return result


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


def generation_configuration(beams: int, returned: int, maximum: int) -> GenerationConfig:
    configuration = GenerationConfig.from_model_config(model.config)
    configuration.num_beams = beams
    configuration.num_return_sequences = returned
    configuration.max_length = maximum
    configuration.eos_token_id = 50256
    configuration.pad_token_id = 50256
    configuration.decoder_start_token_id = 50257
    configuration.return_dict_in_generate = True
    configuration.output_scores = True
    configuration.suppress_tokens = model.generation_config.suppress_tokens
    configuration.begin_suppress_tokens = model.generation_config.begin_suppress_tokens
    configuration.length_penalty = 1.0
    configuration.early_stopping = False
    return configuration


pattern = re.compile(
    r"WHISPER_CPP23_SYNTHID_BEAM beams=(\d+) returned=(\d+) sequences=([^ ]+) "
    r"scores=([^ ]+) calls=(\d+) context_hashes=([-0-9,]*) "
    r"repeated=([01]*) skipped=([01]*) expanded_candidates=(\d+) "
    r"cache_branches=(\d+) graph_nodes_visited=(\d+)"
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
for (
    name,
    beams,
    returned,
    maximum,
    ngram,
    keys,
    history_size,
    table_seed,
    table_size,
    skip_initial,
    debug_mode,
) in CASES:
    synthid_configuration = SynthIDTextWatermarkingConfig(
        ngram_len=ngram,
        keys=keys,
        context_history_size=history_size,
        sampling_table_seed=table_seed,
        sampling_table_size=table_size,
        skip_first_ngram_calls=skip_initial,
        debug_mode=debug_mode,
    )
    builtin_configuration = generation_configuration(beams, returned, maximum)
    builtin_configuration.watermarking_config = synthid_configuration
    with torch.inference_mode():
        builtin = GenerationMixin.generate(
            model,
            inputs=inputs.input_features,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=PREFIX,
            generation_config=builtin_configuration,
        )

    recorder = RecordingSynthIDBeamProcessor(
        ngram_len=ngram,
        keys=keys,
        sampling_table_size=table_size,
        sampling_table_seed=table_seed,
        context_history_size=history_size,
        device=torch.device("cpu"),
        skip_first_ngram_calls=skip_initial,
        debug_mode=debug_mode,
    )
    recorded_configuration = generation_configuration(beams, returned, maximum)
    with torch.inference_mode():
        recorded = GenerationMixin.generate(
            model,
            inputs=inputs.input_features,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=PREFIX,
            generation_config=recorded_configuration,
            logits_processor=[recorder],
        )
    assert recorded.sequences.tolist() == builtin.sequences.tolist(), name
    assert np.allclose(
        recorded.sequences_scores.numpy(),
        builtin.sequences_scores.numpy(),
        rtol=0,
        atol=0,
    ), name

    encoded = [
        str(ngram),
        ",".join(map(str, keys)),
        str(history_size),
        str(table_seed),
        str(table_size),
        str(int(skip_initial)),
        str(int(debug_mode)),
    ]
    line = subprocess.check_output(
        [
            str(CPP),
            "--beam-search-synthid",
            *common,
            str(beams),
            str(returned),
            str(maximum),
            "1.0",
            "heuristic",
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
    expected_sequences = builtin.sequences.tolist()
    expected_scores = builtin.sequences_scores.tolist()
    expected_hashes = [
        value for call in recorder.trace for value in call["context_hashes"]
    ]
    expected_repeated = "".join(
        "1" if value else "0"
        for call in recorder.trace
        for value in call["repeated"]
    )
    expected_skipped = "".join(
        "1" if value else "0"
        for call in recorder.trace
        for value in call["skipped"]
    )
    actual_hashes = (
        []
        if not match.group(6)
        else [int(value) for value in match.group(6).split(",")]
    )
    score_errors = [
        abs(actual - expected)
        for actual, expected in zip(actual_scores, expected_scores)
    ]
    assert int(match.group(1)) == beams and int(match.group(2)) == returned
    assert actual_sequences == expected_sequences, (name, actual_sequences, expected_sequences)
    assert len(actual_scores) == len(expected_scores) == returned
    assert max(score_errors) < 3.0e-4, (name, actual_scores, expected_scores)
    assert int(match.group(5)) == len(recorder.trace)
    assert actual_hashes == expected_hashes, (name, actual_hashes, expected_hashes)
    assert match.group(7) == expected_repeated, name
    assert match.group(8) == expected_skipped, name
    assert int(match.group(9)) > 0 and int(match.group(10)) > 0
    assert int(match.group(11)) == 74
    rows.append(
        {
            "case": name,
            "num_beams": beams,
            "num_return_sequences": returned,
            "maximum_positions": maximum,
            "ngram_len": ngram,
            "key_depth": len(keys),
            "context_history_size": history_size,
            "sampling_table_seed": table_seed,
            "sampling_table_size": table_size,
            "skip_first_ngram_calls": skip_initial,
            "debug_mode": debug_mode,
            "processor_calls": len(recorder.trace),
            "flattened_state_rows": len(expected_hashes),
            "repeated_rows": expected_repeated.count("1"),
            "skipped_rows": expected_skipped.count("1"),
            "ranked_sequences": actual_sequences,
            "cpp23_sequence_scores": actual_scores,
            "transformers_sequence_scores": expected_scores,
            "maximum_sequence_score_error": max(score_errors),
            "all_ranked_sequences_exact": True,
            "all_context_hashes_exact": True,
            "all_repeated_decisions_exact": True,
            "all_skipped_decisions_exact": True,
            "graph_nodes_visited": 74,
            "constructor": "Selection.StandardBeamSearch + WatermarkPolicy.SynthIDTextWatermark",
        }
    )

assert next(row for row in rows if row["case"] == "repeated_empty_context_rows")["repeated_rows"] > 0
assert next(row for row in rows if row["case"] == "startup_skip_rows")["skipped_rows"] > 0

artifact = {
    "certificate": "WHISPER_CPP23_SYNTHID_BEAM_ROW_STATE_1",
    "case_count": len(rows),
    "all_ranked_sequences_exact": all(row["all_ranked_sequences_exact"] for row in rows),
    "all_context_hashes_exact": all(row["all_context_hashes_exact"] for row in rows),
    "all_repeated_decisions_exact": all(row["all_repeated_decisions_exact"] for row in rows),
    "all_skipped_decisions_exact": all(row["all_skipped_decisions_exact"] for row in rows),
    "worst_sequence_score_error": max(row["maximum_sequence_score_error"] for row in rows),
    "all_graph_nodes_visited": all(row["graph_nodes_visited"] == 74 for row in rows),
    "cases": rows,
    "scope": (
        "Batch-one deterministic standard beam search with SynthID processor state "
        "attached to persistent beam-row slots exactly as in Transformers, including "
        "row reordering, startup skips, repeated contexts, signed hashes, EOS and maximum-length finalization."
    ),
}
OUT.mkdir(exist_ok=True)
(OUT / "whisper_cpp23_synthid_beam.json").write_text(json.dumps(artifact, indent=2) + "\n")
table = "\n".join(
    f"| `{row['case']}` | {row['num_beams']} | {row['processor_calls']} | "
    f"{row['flattened_state_rows']} | {row['repeated_rows']} | {row['skipped_rows']} | "
    f"{row['maximum_sequence_score_error']:.3g} | yes |"
    for row in rows
)
(OUT / "WHISPER_CPP23_SYNTHID_BEAM.md").write_text(
    f"""# SynthID state through standard beam rows

Transformers' SynthID processor state follows beam-row slots rather than parent hypothesis ancestry. The C++23 graph represents that behavior explicitly: the first single live C++ frontier initializes all `{max(row['num_beams'] for row in rows)}` logical processor rows identically, then every subsequent row appends the token currently occupying that slot without copying state from the selected parent beam.

| case | beams | calls | flattened row states | repeated | skipped | max score error | exact ranked sequence/state |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

All `{len(rows)}` ranked sequence tensors, signed context hashes, repeated-context decisions, and startup-skip decisions match. Worst normalized sequence-score error is `{artifact['worst_sequence_score_error']:.3g}`. The one-token n-gram case forces the empty context to repeat on every later call without introducing score ties; the signed case uses negative keys, a negative sampling seed, zero history, and a non-power-of-two table.

This certificate covers standard deterministic beam search. Sampled, constrained, and diverse-group beam scheduling remain separate stateful cross-algorithm checks.
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
