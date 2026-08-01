#!/usr/bin/env python3
"""Verify sampled-beam probability transport and stochastic invariants."""
from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import (
    GenerationConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    logging,
)
from transformers.generation.utils import GenerationMixin

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL = ROOT / "work/whisper_tiny_en"
AUDIO = ROOT / "work/whisper_sample.wav"
CPP = ROOT / "work/whisper_graph_cpp23"
logging.set_verbosity_error()

MASS_CASES = [
    (
        "two_beam_default_top_k",
        2,
        {"temperature": 1.0, "top_k": 50},
        ["1", "50", "-", "-", "-", "-", "-"],
    ),
    (
        "four_beam_cool_top_k",
        4,
        {"temperature": 0.7, "top_k": 20},
        [".7", "20", "-", "-", "-", "-", "-"],
    ),
    (
        "four_beam_composed",
        4,
        {"temperature": 1.2, "top_k": 100, "top_p": 0.98},
        ["1.2", "100", ".98", "-", "-", "-", "-"],
    ),
    (
        "three_beam_full_vocabulary",
        3,
        {"temperature": 1.0, "top_k": 0},
        ["1", "-", "-", "-", "-", "-", "-"],
    ),
]

RUN_CASES = [
    ("two_beam_short", 2, 2, 8, 1.0, "heuristic", 1.0, 11, ["50", "-", "-", "-", "-", "-"]),
    ("four_beam_top_k", 4, 4, 12, 1.0, "heuristic", 0.8, 29, ["50", "-", "-", "-", "-", "-"]),
    ("three_beam_warm", 3, 2, 24, 0.7, "heuristic", 1.2, 47, ["100", "-", "-", "-", "-", "-"]),
    ("two_beam_eos", 2, 2, 448, 1.0, "heuristic", 0.7, 7, ["50", "-", "-", "-", "-", "-"]),
]

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
decoder_prefix = torch.tensor([[50257, 50362]], dtype=torch.long)

mass_rows = []
with tempfile.TemporaryDirectory(prefix="whisper-beam-sample-") as temporary:
    temporary = Path(temporary)
    for name, beams, overrides, cpp_filters in MASS_CASES:
        config = GenerationConfig.from_model_config(model.config)
        config.num_beams = beams
        config.num_return_sequences = 1
        config.max_length = 3
        config.do_sample = True
        config.eos_token_id = 50256
        config.pad_token_id = 50256
        config.decoder_start_token_id = 50257
        config.return_dict_in_generate = True
        config.output_scores = True
        config.suppress_tokens = model.generation_config.suppress_tokens
        config.begin_suppress_tokens = model.generation_config.begin_suppress_tokens
        for key, value in overrides.items():
            setattr(config, key, value)
        torch.manual_seed(123)
        with torch.inference_mode():
            expected_output = GenerationMixin.generate(
                model,
                inputs=inputs.input_features,
                attention_mask=inputs.attention_mask,
                decoder_input_ids=decoder_prefix,
                generation_config=config,
            )
        initial_scores = torch.full((beams,), -1.0e9)
        initial_scores[0] = 0.0
        expected = torch.softmax(
            (expected_output.scores[0] + initial_scores[:, None]).flatten(),
            dim=0,
        ).cpu().numpy()

        mass_path = temporary / f"{name}.bin"
        command = [
            str(CPP),
            "--beam-sample-mass",
            str(MODEL / "model.safetensors"),
            str(OUT / "whisper_cpp23_tensor_manifest.tsv"),
            str(AUDIO),
            str(OUT / "whisper_cpp23_hann_f32.bin"),
            str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
            str(OUT / "whisper_cpp23_token_manifest.tsv"),
            str(OUT / "whisper_cpp23_token_bytes.bin"),
            str(beams),
            *cpp_filters,
            str(mass_path),
        ]
        line = subprocess.check_output(command, text=True).strip()
        match = re.fullmatch(
            r"WHISPER_CPP23_BEAM_SAMPLE_MASS beams=(\d+) support=(\d+) "
            r"sum=([0-9.eE+-]+) graph_nodes_visited=(\d+)",
            line,
        )
        assert match, line
        actual = np.fromfile(mass_path, dtype="<f4")
        assert actual.shape == expected.shape == (beams * model.config.vocab_size,)
        actual_support = actual > 0.0
        expected_support = expected > 0.0
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        maximum = float(difference.max())
        l1 = float(difference.sum())
        assert np.array_equal(actual_support, expected_support), name
        assert int(match.group(1)) == beams
        assert int(match.group(2)) == int(actual_support.sum())
        assert abs(float(match.group(3)) - 1.0) < 1e-6
        assert int(match.group(4)) == 73
        assert maximum < 1e-4 and l1 < 1e-4, (name, maximum, l1)
        mass_rows.append(
            {
                "case": name,
                "beams": beams,
                "generation_overrides": overrides,
                "state_count": int(actual.size),
                "support_count": int(actual_support.sum()),
                "support_exact": True,
                "argmax_flat_index": int(actual.argmax()),
                "max_absolute_probability_error": maximum,
                "l1_probability_error": l1,
                "cpp23_probability_sum": float(actual.astype(np.float64).sum()),
                "graph_nodes_visited": 73,
            }
        )

run_pattern = re.compile(
    r"WHISPER_CPP23_BEAM_SAMPLE beams=(\d+) returned=(\d+) "
    r"sequences=([^ ]+) scores=([^ ]+) sampled_candidates=(\d+) "
    r"expanded_candidates=(\d+) cache_branches=(\d+) unique_draw_sets=(\d+) "
    r"mass_error=([0-9.eE+-]+) graph_nodes_visited=(\d+)"
)


def run_cpp(case) -> tuple[str, dict]:
    name, beams, returned, maximum, penalty, stopping, temperature, seed, filters = case
    command = [
        str(CPP),
        "--beam-sample",
        str(MODEL / "model.safetensors"),
        str(OUT / "whisper_cpp23_tensor_manifest.tsv"),
        str(AUDIO),
        str(OUT / "whisper_cpp23_hann_f32.bin"),
        str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
        str(OUT / "whisper_cpp23_token_manifest.tsv"),
        str(OUT / "whisper_cpp23_token_bytes.bin"),
        str(beams),
        str(returned),
        str(maximum),
        str(penalty),
        stopping,
        str(temperature),
        str(seed),
        *filters,
    ]
    line = subprocess.check_output(command, text=True).strip()
    match = run_pattern.fullmatch(line)
    assert match, line
    sequences = [
        [int(token) for token in sequence.split(",")]
        for sequence in match.group(3).split(";")
    ]
    scores = [float(score) for score in match.group(4).split(",")]
    assert int(match.group(1)) == beams
    assert int(match.group(2)) == returned == len(sequences) == len(scores)
    assert all(0 <= token < model.config.vocab_size for row in sequences for token in row)
    assert all(math.isfinite(score) for score in scores)
    assert int(match.group(5)) > 0 and int(match.group(5)) % (2 * beams) == 0
    assert int(match.group(6)) > 0 and int(match.group(7)) > 0
    assert int(match.group(8)) == 1
    assert float(match.group(9)) < 1e-12
    assert int(match.group(10)) == 74
    return line, {
        "case": name,
        "beams": beams,
        "num_return_sequences": returned,
        "maximum_positions": maximum,
        "length_penalty": penalty,
        "stopping": stopping,
        "temperature": temperature,
        "seed": seed,
        "sampling_filters": filters,
        "sequences": sequences,
        "sequence_scores": scores,
        "sampled_candidates": int(match.group(5)),
        "expanded_candidates": int(match.group(6)),
        "cache_branches": int(match.group(7)),
        "every_draw_set_unique": True,
        "maximum_conditional_mass_error": float(match.group(9)),
        "graph_nodes_visited": int(match.group(10)),
        "constructor": "Selection.SampledBeamSearch",
    }


run_rows = []
for case in RUN_CASES:
    line, row = run_cpp(case)
    repeated, _ = run_cpp(case)
    assert line == repeated, case[0]
    row["same_seed_reproducible"] = True
    run_rows.append(row)

artifact = {
    "certificate": "WHISPER_CPP23_SAMPLED_BEAM_SEARCH_1",
    "probability_case_count": len(mass_rows),
    "run_case_count": len(run_rows),
    "all_flattened_supports_exact": all(row["support_exact"] for row in mass_rows),
    "worst_max_absolute_probability_error": max(
        row["max_absolute_probability_error"] for row in mass_rows
    ),
    "worst_l1_probability_error": max(row["l1_probability_error"] for row in mass_rows),
    "all_draw_sets_unique": all(row["every_draw_set_unique"] for row in run_rows),
    "all_same_seed_runs_reproducible": all(row["same_seed_reproducible"] for row in run_rows),
    "all_graph_nodes_visited": all(row["graph_nodes_visited"] == 74 for row in run_rows),
    "probability_cases": mass_rows,
    "run_cases": run_rows,
    "rng_boundary": (
        "C++ uses std::mt19937_64 while PyTorch CPU multinomial uses its own "
        "generator and sampling kernel; probability laws are compared exactly, "
        "not pseudorandom bitstreams or sampled token identity."
    ),
    "scope": (
        "Batch-one beam sampling with explicit flattened beam-token mass, "
        "ordered weighted draws without replacement, score-ranked live "
        "continuations, first-B completion eligibility, and branchable K/V state."
    ),
}
(OUT / "whisper_cpp23_beam_sampling.json").write_text(
    json.dumps(artifact, indent=2) + "\n"
)
mass_table = "\n".join(
    f"| `{row['case']}` | {row['beams']} | {row['state_count']} | "
    f"{row['support_count']} | {row['max_absolute_probability_error']:.6g} | "
    f"{row['l1_probability_error']:.6g} | yes |"
    for row in mass_rows
)
run_table = "\n".join(
    f"| `{row['case']}` | {row['beams']} | {row['num_return_sequences']} | "
    f"{row['maximum_positions']} | {row['sampled_candidates']} | "
    f"{row['maximum_conditional_mass_error']:.3g} | yes | yes |"
    for row in run_rows
)
(OUT / "WHISPER_CPP23_BEAM_SAMPLING.md").write_text(
    f"""# Explicit C++23 sampled-beam probability graph

At each transition, the graph forms one categorical law over the complete live-beam × 51,864-token product. It draws `2B` distinct continuations without replacement, preserving draw order for completion eligibility, then score-ranks the nonfinished sampled states to retain `B` branchable token/K/V states.

| probability case | beams | product states | nonzero support | max probability error | L1 error | support exact |
|---|---:|---:|---:|---:|---:|---:|
{mass_table}

| stochastic run | beams | returned | max positions | sampled candidates | max conditional mass error | unique draw sets | reproducible seed |
|---|---:|---:|---:|---:|---:|---:|---:|
{run_table}

All four complete first-transition product distributions match Transformers support exactly. Worst maximum probability error is `{artifact['worst_max_absolute_probability_error']:.9g}` and worst L1 error is `{artifact['worst_l1_probability_error']:.9g}`. All full runs visit 74 graph nodes, conserve each conditional mass within `{max(row['maximum_conditional_mass_error'] for row in run_rows):.3g}`, and never repeat a candidate within a without-replacement draw set.

The RNG boundary is explicit: C++ uses `std::mt19937_64`; PyTorch uses its own CPU generator and multinomial kernel. Semantic equivalence is therefore asserted for the probability law and transition algorithm, not identical random bitstreams.
"""
)
print(
    json.dumps(
        {
            key: artifact[key]
            for key in (
                "certificate",
                "probability_case_count",
                "run_case_count",
                "all_flattened_supports_exact",
                "worst_max_absolute_probability_error",
                "worst_l1_probability_error",
                "all_draw_sets_unique",
                "all_same_seed_runs_reproducible",
                "all_graph_nodes_visited",
            )
        },
        indent=2,
    )
)
