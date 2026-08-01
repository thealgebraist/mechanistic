#!/usr/bin/env python3
"""Verify explicit prompt-lookup speculative decoding against Transformers."""
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
from transformers.generation.candidate_generator import PromptLookupCandidateGenerator
from transformers.generation.utils import GenerationMixin

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL = ROOT / "work/whisper_tiny_en"
AUDIO = ROOT / "work/whisper_sample.wav"
CPP = ROOT / "work/whisper_graph_cpp23"
CANDIDATE_HASH = "9809a08720d61cb0e7bb998685e6fb98c4b0e76a3f73cc0cf07603d69a00950b"
UTILS_HASH = "a20024b1e82ed5361a524d238d2197be5407abc91297dd9888c57e8284d63fef"

ACCEPTED_PREFIX = [
    50257, 50362, 1770, 13, 2264, 346, 353, 318, 262, 46329, 286, 262, 46329
]
MISMATCH_PREFIX = [50257, 50362, 1770, 13, 2264, 346, 353, 50257, 50362]
REPEATED_PHRASE = [
    50257, 50362, 1770, 13, 2264, 346, 353, 318, 262, 46329,
    286, 262, 3504, 286, 262,
]
CASES = [
    ("accepted_two_then_correct", ACCEPTED_PREFIX, 5, 1, 25),
    ("single_token_proposals", ACCEPTED_PREFIX, 1, 1, 24),
    ("longest_ngram_scan", ACCEPTED_PREFIX, 4, 4, 28),
    ("first_candidate_mismatch", MISMATCH_PREFIX, 5, 2, 25),
    ("repeated_phrase", REPEATED_PHRASE, 4, 3, 27),
    ("eos_termination", ACCEPTED_PREFIX, 5, 2, 448),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


generation_root = Path(transformers.__file__).parent / "generation"
if sha256(generation_root / "candidate_generator.py") != CANDIDATE_HASH:
    raise RuntimeError("candidate-generator source hash mismatch")
if sha256(generation_root / "utils.py") != UTILS_HASH:
    raise RuntimeError("assisted-decoding source hash mismatch")

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


def configuration(proposal: int, ngram: int, maximum: int) -> GenerationConfig:
    config = GenerationConfig.from_model_config(model.config)
    config.max_length = maximum
    config.prompt_lookup_num_tokens = proposal
    config.max_matching_ngram_size = ngram
    config.eos_token_id = 50256
    config.pad_token_id = 50256
    config.decoder_start_token_id = 50257
    config.suppress_tokens = model.generation_config.suppress_tokens
    config.begin_suppress_tokens = model.generation_config.begin_suppress_tokens
    return config


pattern = re.compile(
    r'WHISPER_CPP23_PROMPT_LOOKUP initial_positions=(\d+) tokens=([^ ]*) '
    r'proposal_rounds=(\d+) proposed_tokens=(\d+) accepted_candidates=(\d+) '
    r'correction_tokens=(\d+) target_evaluations=(\d+) first_proposal=([^ ]*) '
    r'first_accepted=(\d+) terminated_by_eos=(\d) graph_nodes_visited=(\d+) '
    r'text="(.*)"'
)
rows = []
for name, initial, proposal_count, ngram, maximum in CASES:
    config = configuration(proposal_count, ngram, maximum)
    initial_tensor = torch.tensor([initial], dtype=torch.long)
    generator = PromptLookupCandidateGenerator(
        eos_token_id=torch.tensor([50256]),
        num_output_tokens=proposal_count,
        max_matching_ngram_size=ngram,
        max_length=maximum,
    )
    reference_candidate_ids, _ = generator.get_candidates(initial_tensor)
    first_proposal = reference_candidate_ids[0, len(initial) :].tolist()
    assert first_proposal, name
    with torch.inference_mode():
        complete = GenerationMixin.generate(
            model,
            inputs=inputs.input_features,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=initial_tensor,
            generation_config=config,
        )[0].tolist()
    expected = complete[len(initial) :]
    expected_eos = bool(expected and expected[-1] == 50256)
    if expected_eos:
        expected = expected[:-1]
    expected_first_accepted = 0
    for candidate, target in zip(first_proposal, expected):
        if candidate != target:
            break
        expected_first_accepted += 1

    command = [
        str(CPP),
        "--prompt-lookup",
        str(MODEL / "model.safetensors"),
        str(OUT / "whisper_cpp23_tensor_manifest.tsv"),
        str(AUDIO),
        str(OUT / "whisper_cpp23_hann_f32.bin"),
        str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
        str(OUT / "whisper_cpp23_token_manifest.tsv"),
        str(OUT / "whisper_cpp23_token_bytes.bin"),
        ",".join(map(str, initial)),
        str(proposal_count),
        str(ngram),
        str(maximum),
    ]
    line = subprocess.check_output(command, text=True).strip()
    match = pattern.fullmatch(line)
    assert match, line
    actual = [] if not match.group(2) else [int(token) for token in match.group(2).split(",")]
    actual_first = [] if not match.group(8) else [int(token) for token in match.group(8).split(",")]
    assert int(match.group(1)) == len(initial)
    assert actual == expected, (name, actual, expected)
    assert actual_first == first_proposal, (name, actual_first, first_proposal)
    assert int(match.group(9)) == expected_first_accepted
    assert bool(int(match.group(10))) == expected_eos
    assert int(match.group(11)) == 74
    assert int(match.group(3)) > 0 and int(match.group(4)) >= len(first_proposal)
    assert int(match.group(5)) >= expected_first_accepted
    assert int(match.group(6)) > 0 and int(match.group(7)) >= len(actual)
    expected_text = processor.tokenizer.decode(expected, skip_special_tokens=True)
    assert match.group(12) == expected_text, (name, match.group(12), expected_text)
    rows.append(
        {
            "case": name,
            "initial_positions": len(initial),
            "prompt_lookup_num_tokens": proposal_count,
            "max_matching_ngram_size": ngram,
            "maximum_positions": maximum,
            "first_proposal": first_proposal,
            "first_accepted": int(match.group(9)),
            "output_tokens": actual,
            "complete_tokens_exact": True,
            "proposal_rounds": int(match.group(3)),
            "proposed_tokens": int(match.group(4)),
            "accepted_candidate_tokens": int(match.group(5)),
            "correction_tokens": int(match.group(6)),
            "target_evaluations": int(match.group(7)),
            "terminated_by_eos": expected_eos,
            "graph_nodes_visited": int(match.group(11)),
            "constructor": "Selection.PromptLookupSearch",
        }
    )

artifact = {
    "certificate": "WHISPER_CPP23_PROMPT_LOOKUP_SPECULATION_1",
    "transformers_version": transformers.__version__,
    "candidate_generator_source_sha256": CANDIDATE_HASH,
    "assisted_decoding_source_sha256": UTILS_HASH,
    "case_count": len(rows),
    "all_first_proposals_exact": True,
    "all_first_acceptance_counts_exact": True,
    "all_complete_tokens_exact": all(row["complete_tokens_exact"] for row in rows),
    "all_graph_nodes_visited": all(row["graph_nodes_visited"] == 74 for row in rows),
    "total_proposed_tokens": sum(row["proposed_tokens"] for row in rows),
    "total_accepted_candidate_tokens": sum(
        row["accepted_candidate_tokens"] for row in rows
    ),
    "cases": rows,
    "scope": (
        "Batch-one greedy prompt-lookup assisted decoding with longest-first/leftmost "
        "ngram proposals, target verification, accepted-prefix transport, correction "
        "tokens, dynamic K/V state, maximum-length and EOS termination."
    ),
}
OUT.mkdir(exist_ok=True)
(OUT / "whisper_cpp23_prompt_lookup.json").write_text(
    json.dumps(artifact, indent=2) + "\n"
)
table = "\n".join(
    f"| `{row['case']}` | {row['prompt_lookup_num_tokens']} | "
    f"{row['max_matching_ngram_size']} | `{row['first_proposal']}` | "
    f"{row['first_accepted']} | {row['accepted_candidate_tokens']} | yes |"
    for row in rows
)
(OUT / "WHISPER_CPP23_PROMPT_LOOKUP.md").write_text(
    f"""# Explicit prompt-lookup speculative graph

The typed `PromptLookupSearch` scans the current decoder token stack from the largest configured suffix n-gram down to one token and chooses the first earlier left-to-right occurrence with a continuation. That copied continuation becomes a proposal edge. Whisper evaluates each proposed token; the matching prefix is committed to the K/V state, the first mismatch is replaced by the target token, and a fully accepted proposal receives one extra target-model token.

The oracle is Transformers `{transformers.__version__}`. `candidate_generator.py` is hash `{CANDIDATE_HASH}` and the assisted acceptance/cache implementation in `utils.py` is hash `{UTILS_HASH}`.

| case | proposal width | max n-gram | first proposal | first accepted | all accepted candidates | complete tokens exact |
|---|---:|---:|---|---:|---:|---:|
{table}

All `{len(rows)}` first proposals, first accepted-prefix lengths, and complete output sequences match. Across the cases, `{artifact['total_accepted_candidate_tokens']}` of `{artifact['total_proposed_tokens']}` proposed token occurrences were accepted. The remaining transitions were explicit target-model corrections rather than silently discarded speculative state.

This finite certificate covers batch-one greedy prompt lookup. Sampled speculative acceptance and an external assistant model are separate algorithms.
"""
)
print(
    json.dumps(
        {
            key: artifact[key]
            for key in (
                "certificate",
                "case_count",
                "all_first_proposals_exact",
                "all_first_acceptance_counts_exact",
                "all_complete_tokens_exact",
                "total_proposed_tokens",
                "total_accepted_candidate_tokens",
                "all_graph_nodes_visited",
            )
        },
        indent=2,
    )
)
