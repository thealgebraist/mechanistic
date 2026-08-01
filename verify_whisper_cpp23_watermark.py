#!/usr/bin/env python3
"""Verify the explicit C++23 watermark graph against pinned Transformers."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import (
    SynthIDTextWatermarkingConfig,
    WatermarkingConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    logging,
)
from transformers.generation.logits_process import (
    SynthIDTextWatermarkLogitsProcessor,
    WatermarkLogitsProcessor,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL = ROOT / "work/whisper_tiny_en"
AUDIO = ROOT / "work/whisper_sample.wav"
CPP = ROOT / "work/whisper_graph_cpp23"
LOGITS_PROCESS_HASH = "6481506629ff6f021157e893fc64c4e7b5b06a72072771809d679387cb54a4b1"
CONFIGURATION_HASH = "d83f2281f939402be1633a29f3c760e29f5d2f284258d8ed99693b873744074b"
TORCH_COMMIT = "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0"
MAXIMUM_POSITIONS = 30
PREFIX = torch.tensor([[50257, 50362]], dtype=torch.long)

CASES = [
    ("left_default", "lefthash", 0.25, 2.0, 15485863, 1),
    ("left_context_two", "lefthash", 0.33, 1.25, 104729, 2),
    ("left_negative_key_bias", "lefthash", 0.10, -0.75, -17, 1),
    ("self_default", "selfhash", 0.25, 2.0, 15485863, 1),
    ("self_context_two", "selfhash", 0.40, 1.5, 104729, 2),
    ("self_delayed_context", "selfhash", 0.15, 3.0, -17, 3),
]
SYNTHID_CASES = [
    (
        "synthid_default",
        5,
        [654, 400, 836, 123, 340, 443, 597, 160, 57],
        1024,
        0,
        65536,
        False,
        False,
    ),
    ("synthid_small_table", 3, [1, 2, 3], 8, 17, 257, False, False),
    ("synthid_skip_initial", 4, [9, 8], 32, 1, 1024, True, False),
    ("synthid_repeated_debug", 2, [1], 4, 17, 257, False, True),
    ("synthid_signed_seed_keys", 2, [-1, -(2**63)], 0, -17, 31, False, False),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


generation_root = Path(transformers.__file__).parent / "generation"
if sha256(generation_root / "logits_process.py") != LOGITS_PROCESS_HASH:
    raise RuntimeError("watermark logits-processor source hash mismatch")
if sha256(generation_root / "configuration_utils.py") != CONFIGURATION_HASH:
    raise RuntimeError("watermark configuration source hash mismatch")
if torch.version.git_version != TORCH_COMMIT:
    raise RuntimeError("PyTorch randperm implementation commit mismatch")

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

with torch.inference_mode():
    baseline = model.generate(
        inputs.input_features,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_scores=True,
    )
baseline_scores = baseline.scores[0]


class RecordingSynthIDProcessor(SynthIDTextWatermarkLogitsProcessor):
    """Pinned processor with an observational state trace only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace: list[dict] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.state is None:
            context = torch.zeros(
                (input_ids.shape[0], self.ngram_len - 1), dtype=torch.long
            )
            history = torch.zeros(
                (input_ids.shape[0], self.context_history_size), dtype=torch.long
            )
            call = 1
        else:
            context = torch.concat((self.state.context, input_ids[:, -1:]), dim=1)[
                :, 1:
            ]
            history = self.state.context_history.clone()
            call = self.state.num_calls + 1
        skipped = self.skip_first_ngram_calls and call < self.ngram_len
        if skipped:
            context_hash = 0
            repeated = False
        else:
            context_hash = int(
                self.accumulate_hash(
                    torch.ones(input_ids.shape[0], dtype=torch.long), context
                )[0]
            )
            repeated = bool(
                history.shape[1] > 0 and (history[0] == context_hash).any()
            )
        result = super().__call__(input_ids, scores)
        self.trace.append(
            {
                "call": call,
                "context_hash": context_hash,
                "repeated": repeated,
                "skipped": skipped,
            }
        )
        return result

common = [
    str(MODEL / "model.safetensors"),
    str(OUT / "whisper_cpp23_tensor_manifest.tsv"),
    str(AUDIO),
    str(OUT / "whisper_cpp23_hann_f32.bin"),
    str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
    str(OUT / "whisper_cpp23_token_manifest.tsv"),
    str(OUT / "whisper_cpp23_token_bytes.bin"),
]
mass_pattern = re.compile(
    r"WHISPER_CPP23_WATERMARK_MASS scheme=(lefthash|selfhash) green=(\d+) "
    r"selected=(\d+) sum=([^ ]+) graph_nodes_visited=(\d+)"
)
sequence_pattern = re.compile(
    r"WHISPER_CPP23_WATERMARK_TRANSCRIPT scheme=(lefthash|selfhash) "
    r"tokens=([0-9,]*) terminated_by_eos=([01]) graph_nodes_visited=(\d+)"
)

rows = []
with tempfile.TemporaryDirectory(prefix="whisper-watermark-") as temporary:
    temporary = Path(temporary)
    for index, (name, scheme, ratio, bias, key, context) in enumerate(CASES):
        configuration = WatermarkingConfig(
            greenlist_ratio=ratio,
            bias=bias,
            hashing_key=key,
            seeding_scheme=scheme,
            context_width=context,
        )
        reference = WatermarkLogitsProcessor(
            vocab_size=51864,
            device="cpu",
            greenlist_ratio=ratio,
            bias=bias,
            hashing_key=key,
            seeding_scheme=scheme,
            context_width=context,
        )
        if PREFIX.shape[-1] < context:
            green_ids = torch.empty(0, dtype=torch.long)
        elif scheme == "lefthash":
            green_ids = reference._get_greenlist_ids(PREFIX[0])
        else:
            green_ids = reference._score_rejection_sampling(
                PREFIX[0], baseline_scores[0]
            )
        expected_mask = np.zeros(51864, dtype=np.uint8)
        expected_mask[green_ids.numpy()] = 1
        expected_mass = reference(PREFIX, baseline_scores).softmax(-1)[0].numpy()

        mass_path = temporary / f"case-{index}-mass.bin"
        mask_path = temporary / f"case-{index}-mask.bin"
        encoded = [scheme, str(ratio), str(bias), str(key), str(context)]
        mass_line = subprocess.check_output(
            [str(CPP), "--watermark-mass", *common, *encoded, str(mass_path), str(mask_path)],
            text=True,
        ).strip()
        mass_match = mass_pattern.fullmatch(mass_line)
        assert mass_match, mass_line
        actual_mass = np.fromfile(mass_path, dtype="<f4")
        actual_mask = np.fromfile(mask_path, dtype=np.uint8)
        assert actual_mass.shape == (51864,) and actual_mask.shape == (51864,)
        assert np.array_equal(actual_mask, expected_mask), name
        maximum_probability_error = float(np.max(np.abs(actual_mass - expected_mass)))
        assert maximum_probability_error < 1.0e-4, (name, maximum_probability_error)
        assert int(mass_match.group(2)) == int(expected_mask.sum())
        assert int(mass_match.group(3)) == int(expected_mass.argmax())
        assert abs(float(mass_match.group(4)) - 1.0) < 1.0e-5
        assert int(mass_match.group(5)) == 73

        with torch.inference_mode():
            expected_tokens = model.generate(
                inputs.input_features,
                attention_mask=inputs.attention_mask,
                max_length=MAXIMUM_POSITIONS,
                watermarking_config=configuration,
            )[0].tolist()
        sequence_line = subprocess.check_output(
            [
                str(CPP),
                "--transcribe-watermark",
                *common,
                *encoded,
                str(MAXIMUM_POSITIONS),
            ],
            text=True,
        ).strip()
        sequence_match = sequence_pattern.fullmatch(sequence_line)
        assert sequence_match, sequence_line
        actual_tokens = (
            []
            if not sequence_match.group(2)
            else [int(token) for token in sequence_match.group(2).split(",")]
        )
        assert actual_tokens == expected_tokens, (name, actual_tokens, expected_tokens)
        assert sequence_match.group(3) == "1" and int(sequence_match.group(4)) == 74
        rows.append(
            {
                "case": name,
                "constructor": (
                    "WatermarkPolicy.LeftHashWatermark"
                    if scheme == "lefthash"
                    else "WatermarkPolicy.SelfHashWatermark"
                ),
                "configuration": {
                    "greenlist_ratio": ratio,
                    "bias": bias,
                    "hashing_key": key,
                    "seeding_scheme": scheme,
                    "context_width": context,
                },
                "first_step_green_tokens": int(expected_mask.sum()),
                "first_step_green_mask_exact": True,
                "first_step_argmax": int(expected_mass.argmax()),
                "maximum_full_probability_error": maximum_probability_error,
                "generated_token_ids": actual_tokens,
                "generated_text": processor.decode(
                    actual_tokens, skip_special_tokens=True
                ).strip(),
                "complete_sequence_exact": True,
                "graph_nodes_visited": 74,
            }
        )


REJECTIONS = [
    ("unknown_scheme", "unknown", 0.25, 2.0, 15485863, 1),
    ("zero_ratio", "lefthash", 0.0, 2.0, 15485863, 1),
    ("unit_ratio", "lefthash", 1.0, 2.0, 15485863, 1),
    ("zero_context", "selfhash", 0.25, 2.0, 15485863, 0),
]
rejections = []
for name, scheme, ratio, bias, key, context in REJECTIONS:
    transformers_rejected = False
    try:
        configuration = WatermarkingConfig(
            greenlist_ratio=ratio,
            bias=bias,
            hashing_key=key,
            seeding_scheme=scheme,
            context_width=context,
        )
        configuration.validate()
        configuration.construct_processor(51864, "cpu")
    except (ValueError, RuntimeError):
        transformers_rejected = True
    process = subprocess.run(
        [
            str(CPP),
            "--transcribe-watermark",
            *common,
            scheme,
            str(ratio),
            str(bias),
            str(key),
            str(context),
            str(MAXIMUM_POSITIONS),
        ],
        text=True,
        capture_output=True,
    )
    cpp_rejected = process.returncode != 0
    assert transformers_rejected and cpp_rejected, name
    rejections.append(
        {
            "case": name,
            "transformers_rejected": True,
            "cpp23_rejected": True,
        }
    )

synthid_mass_pattern = re.compile(
    r"WHISPER_CPP23_SYNTHID_MASS call=(\d+) context_hash=(-?\d+) "
    r"repeated=([01]) skipped=([01]) g_ones=(\d+) selected=(\d+) "
    r"sum=([^ ]+) graph_nodes_visited=(\d+)"
)
synthid_sequence_pattern = re.compile(
    r"WHISPER_CPP23_SYNTHID_TRANSCRIPT tokens=([0-9,]*) calls=(\d+) "
    r"context_hashes=([-0-9,]*) repeated=([01]*) skipped=([01]*) "
    r"terminated_by_eos=([01]) graph_nodes_visited=(\d+)"
)
synthid_rows = []
with tempfile.TemporaryDirectory(prefix="whisper-synthid-") as temporary:
    temporary = Path(temporary)
    for index, (
        name,
        ngram,
        keys,
        history_size,
        table_seed,
        table_size,
        skip_initial,
        debug_mode,
    ) in enumerate(SYNTHID_CASES):
        reference = SynthIDTextWatermarkLogitsProcessor(
            ngram_len=ngram,
            keys=keys,
            sampling_table_size=table_size,
            sampling_table_seed=table_seed,
            context_history_size=history_size,
            device=torch.device("cpu"),
            skip_first_ngram_calls=skip_initial,
            debug_mode=debug_mode,
        )
        first_skipped = skip_initial and 1 < ngram
        if first_skipped:
            expected_g = np.empty(0, dtype=np.uint8)
            expected_context_hash = 0
        else:
            context = torch.zeros((1, ngram - 1), dtype=torch.long)
            indices = torch.arange(51864, dtype=torch.long)[None, :]
            ngram_keys, context_hash = reference._compute_keys(context, indices)
            expected_g = (
                reference.sample_g_values(ngram_keys)
                .numpy()[0]
                .astype(np.uint8)
                .reshape(-1)
            )
            expected_context_hash = int(context_hash[0])
        expected_mass = reference(PREFIX, baseline_scores).softmax(-1)[0].numpy()

        mass_path = temporary / f"case-{index}-mass.bin"
        g_path = temporary / f"case-{index}-g.bin"
        encoded = [
            str(ngram),
            "-" if not keys else ",".join(map(str, keys)),
            str(history_size),
            str(table_seed),
            str(table_size),
            str(int(skip_initial)),
            str(int(debug_mode)),
        ]
        mass_line = subprocess.check_output(
            [str(CPP), "--synthid-mass", *common, *encoded, str(mass_path), str(g_path)],
            text=True,
        ).strip()
        mass_match = synthid_mass_pattern.fullmatch(mass_line)
        assert mass_match, mass_line
        actual_mass = np.fromfile(mass_path, dtype="<f4")
        actual_g = np.fromfile(g_path, dtype=np.uint8)
        assert np.array_equal(actual_g, expected_g), name
        maximum_probability_error = float(np.max(np.abs(actual_mass - expected_mass)))
        assert maximum_probability_error < 1.0e-4, (name, maximum_probability_error)
        assert int(mass_match.group(1)) == 1
        assert int(mass_match.group(2)) == expected_context_hash
        assert bool(int(mass_match.group(3))) is False
        assert bool(int(mass_match.group(4))) == first_skipped
        assert int(mass_match.group(5)) == int(expected_g.sum())
        assert int(mass_match.group(6)) == int(expected_mass.argmax())
        assert abs(float(mass_match.group(7)) - 1.0) < 1.0e-5
        assert int(mass_match.group(8)) == 73

        configuration = SynthIDTextWatermarkingConfig(
            ngram_len=ngram,
            keys=keys,
            context_history_size=history_size,
            sampling_table_seed=table_seed,
            sampling_table_size=table_size,
            skip_first_ngram_calls=skip_initial,
            debug_mode=debug_mode,
        )
        with torch.inference_mode():
            expected_tokens = model.generate(
                inputs.input_features,
                attention_mask=inputs.attention_mask,
                max_length=MAXIMUM_POSITIONS,
                watermarking_config=configuration,
            )[0].tolist()
        recorder = RecordingSynthIDProcessor(
            ngram_len=ngram,
            keys=keys,
            sampling_table_size=table_size,
            sampling_table_seed=table_seed,
            context_history_size=history_size,
            device=torch.device("cpu"),
            skip_first_ngram_calls=skip_initial,
            debug_mode=debug_mode,
        )
        with torch.inference_mode():
            recorded_tokens = model.generate(
                inputs.input_features,
                attention_mask=inputs.attention_mask,
                max_length=MAXIMUM_POSITIONS,
                logits_processor=[recorder],
            )[0].tolist()
        assert recorded_tokens == expected_tokens, name

        sequence_line = subprocess.check_output(
            [
                str(CPP),
                "--transcribe-synthid",
                *common,
                *encoded,
                str(MAXIMUM_POSITIONS),
            ],
            text=True,
        ).strip()
        sequence_match = synthid_sequence_pattern.fullmatch(sequence_line)
        assert sequence_match, sequence_line
        actual_tokens = (
            []
            if not sequence_match.group(1)
            else [int(token) for token in sequence_match.group(1).split(",")]
        )
        actual_hashes = (
            []
            if not sequence_match.group(3)
            else [int(value) for value in sequence_match.group(3).split(",")]
        )
        expected_hashes = [item["context_hash"] for item in recorder.trace]
        expected_repeated = "".join("1" if item["repeated"] else "0" for item in recorder.trace)
        expected_skipped = "".join("1" if item["skipped"] else "0" for item in recorder.trace)
        assert actual_tokens == expected_tokens, (name, actual_tokens, expected_tokens)
        assert int(sequence_match.group(2)) == len(recorder.trace)
        assert actual_hashes == expected_hashes, (name, actual_hashes, expected_hashes)
        assert sequence_match.group(4) == expected_repeated, name
        assert sequence_match.group(5) == expected_skipped, name
        expected_eos_termination = len(expected_tokens) < MAXIMUM_POSITIONS - PREFIX.shape[-1]
        assert bool(int(sequence_match.group(6))) == expected_eos_termination
        assert int(sequence_match.group(7)) == 74
        synthid_rows.append(
            {
                "case": name,
                "constructor": "WatermarkPolicy.SynthIDTextWatermark",
                "configuration": {
                    "ngram_len": ngram,
                    "keys": keys,
                    "context_history_size": history_size,
                    "sampling_table_seed": table_seed,
                    "sampling_table_size": table_size,
                    "skip_first_ngram_calls": skip_initial,
                    "debug_mode": debug_mode,
                },
                "first_step_g_value_count": int(expected_g.size),
                "first_step_g_values_exact": True,
                "first_context_hash": expected_context_hash,
                "maximum_full_probability_error": maximum_probability_error,
                "state_calls": len(recorder.trace),
                "context_hash_trajectory_exact": True,
                "repeated_context_trajectory_exact": True,
                "skipped_call_trajectory_exact": True,
                "repeated_context_calls": expected_repeated.count("1"),
                "skipped_initial_calls": expected_skipped.count("1"),
                "generated_token_ids": actual_tokens,
                "generated_text": processor.decode(actual_tokens, skip_special_tokens=True).strip(),
                "complete_sequence_exact": True,
                "terminated_by_eos": expected_eos_termination,
                "graph_nodes_visited": 74,
            }
        )

SYNTHID_REJECTIONS = [
    ("synthid_zero_ngram", 0, [1], 4, 0, 257, False, False),
    ("synthid_zero_table", 2, [1], 4, 0, 0, False, False),
    ("synthid_oversized_table", 2, [1], 4, 0, 2**24 + 1, False, False),
    ("synthid_negative_history", 2, [1], -1, 0, 257, False, False),
    ("synthid_empty_depth", 3, [], 16, 7, 17, False, False),
]
for name, ngram, keys, history_size, table_seed, table_size, skip_initial, debug_mode in SYNTHID_REJECTIONS:
    transformers_rejected = False
    try:
        configuration = SynthIDTextWatermarkingConfig(
            ngram_len=ngram,
            keys=keys,
            context_history_size=history_size,
            sampling_table_seed=table_seed,
            sampling_table_size=table_size,
            skip_first_ngram_calls=skip_initial,
            debug_mode=debug_mode,
        )
        configuration.validate()
        candidate = configuration.construct_processor(51864, "cpu")
        candidate(PREFIX, baseline_scores)
    except (ValueError, RuntimeError, ZeroDivisionError):
        transformers_rejected = True
    encoded = [
        str(ngram),
        "-" if not keys else ",".join(map(str, keys)),
        str(history_size),
        str(table_seed),
        str(table_size),
        str(int(skip_initial)),
        str(int(debug_mode)),
    ]
    process = subprocess.run(
        [str(CPP), "--transcribe-synthid", *common, *encoded, str(MAXIMUM_POSITIONS)],
        text=True,
        capture_output=True,
    )
    cpp_rejected = process.returncode != 0
    assert transformers_rejected and cpp_rejected, name
    rejections.append(
        {"case": name, "transformers_rejected": True, "cpp23_rejected": True}
    )

artifact = {
    "certificate": "WHISPER_CPP23_WATERMARK_GRAPH_2",
    "transformers_version": transformers.__version__,
    "transformers_logits_process_sha256": LOGITS_PROCESS_HASH,
    "transformers_configuration_sha256": CONFIGURATION_HASH,
    "torch_version": torch.__version__,
    "torch_git_commit": TORCH_COMMIT,
    "randperm_lowering": "MT19937 low-32-bit seed plus forward Fisher-Yates swaps",
    "case_count": len(rows) + len(synthid_rows),
    "classic_case_count": len(rows),
    "synthid_case_count": len(synthid_rows),
    "rejection_case_count": len(rejections),
    "all_first_step_green_masks_exact": all(
        row["first_step_green_mask_exact"] for row in rows
    ),
    "all_complete_sequences_exact": all(
        row["complete_sequence_exact"] for row in rows + synthid_rows
    ),
    "all_synthid_g_values_exact": all(row["first_step_g_values_exact"] for row in synthid_rows),
    "all_synthid_state_trajectories_exact": all(
        row["context_hash_trajectory_exact"]
        and row["repeated_context_trajectory_exact"]
        and row["skipped_call_trajectory_exact"]
        for row in synthid_rows
    ),
    "all_synthid_complete_sequences_exact": all(
        row["complete_sequence_exact"] for row in synthid_rows
    ),
    "all_invalid_configurations_rejected": all(
        row["transformers_rejected"] and row["cpp23_rejected"] for row in rejections
    ),
    "worst_maximum_full_probability_error": max(
        row["maximum_full_probability_error"] for row in rows + synthid_rows
    ),
    "classic_cases": rows,
    "synthid_cases": synthid_rows,
    "rejections": rejections,
    "scope": (
        "Real-audio greedy watermark generation plus complete first-step 51,864-token "
        "probability vectors and exact green-set membership for lefthash and selfhash, "
        "plus exact SynthID g-values, signed context hashes, bounded repetition state, "
        "startup-skip state, and complete real-audio trajectories. "
        "The probability tolerance includes the separately audited Accelerate-versus-PyTorch "
        "floating-point model error; green masks and complete token sequences are exact."
    ),
}
OUT.mkdir(exist_ok=True)
(OUT / "whisper_cpp23_watermark.json").write_text(json.dumps(artifact, indent=2) + "\n")
table = "\n".join(
    f"| `{row['case']}` | `{row['constructor'].split('.')[-1]}` | "
    f"{row['first_step_green_tokens']} | {row['maximum_full_probability_error']:.3g} | "
    f"{len(row['generated_token_ids'])} | yes |"
    for row in rows
)
synthid_table = "\n".join(
    f"| `{row['case']}` | {row['configuration']['ngram_len']} | "
    f"{len(row['configuration']['keys'])} | {row['first_step_g_value_count']} | "
    f"{row['state_calls']} | {row['repeated_context_calls']} | "
    f"{row['skipped_initial_calls']} | {row['maximum_full_probability_error']:.3g} | yes |"
    for row in synthid_rows
)
assert next(
    row for row in synthid_rows if row["case"] == "synthid_repeated_debug"
)["repeated_context_calls"] > 0
assert next(
    row for row in synthid_rows if row["case"] == "synthid_skip_initial"
)["skipped_initial_calls"] > 0
(OUT / "WHISPER_CPP23_WATERMARK.md").write_text(
    f"""# Explicit Whisper watermark graph

The C++23 graph now lowers Transformers `{transformers.__version__}` watermarking into three named ADT constructors. `LeftHashWatermark` maps the final context token through a keyed MT19937 permutation and adds a bias to a fixed-ratio green vocabulary. `SelfHashWatermark` constructs the pinned 1,000,003-entry key table, examines the top 40 candidates, and adds the bias only when a candidate belongs to its own candidate-conditioned green set. `SynthIDTextWatermark` carries a rolling n-minus-one token context, bounded context-hash history, call count, keyed sampling table, and depth-wise probability tournament.

The PyTorch `{torch.__version__}` CPU permutation at commit `{TORCH_COMMIT}` is represented without PyTorch as a low-32-bit MT19937 seed followed by forward Fisher-Yates swaps. This is linear in vocabulary size and does not materialize a history-by-token transition table.

| case | typed scheme | first-step green tokens | maximum probability error | output tokens | exact sequence |
|---|---|---:|---:|---:|---:|
{table}

All `{len(rows)}` complete real-audio token sequences and all first-step green masks match exactly. The worst full-distribution error is `{artifact['worst_maximum_full_probability_error']:.3g}` across all 51,864 probabilities; this includes the already-audited model-backend floating-point difference. All `{len(rejections)}` invalid scheme, ratio, and context cases are rejected by both implementations.

## SynthID state graph

| case | n-gram | key depths | first g-values | state calls | repeated | skipped | maximum probability error | exact sequence/state |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{synthid_table}

All `{len(synthid_rows)}` first-step g-value tensors, complete signed context-hash trajectories, repeated-context decisions, startup-skip decisions, and generated token sequences match exactly. The repeated-debug case deliberately reuses the same one-token context; the skip case verifies that no context hash enters history before the configured n-gram startup boundary. Zero-length context history, signed keys, signed sampling seeds, and non-power-of-two sampling tables are included; empty key depth is verified as a matched runtime rejection.

This closes both configuration families for batch-one greedy execution. Stateful SynthID transport through reordered beam rows remains separate cross-algorithm work and is not claimed here.
"""
)
print(
    json.dumps(
        {
            key: artifact[key]
            for key in (
                "certificate",
                "case_count",
                "rejection_case_count",
                "all_first_step_green_masks_exact",
                "all_synthid_g_values_exact",
                "all_synthid_state_trajectories_exact",
                "all_complete_sequences_exact",
                "all_invalid_configurations_rejected",
                "worst_maximum_full_probability_error",
            )
        },
        indent=2,
    )
)
