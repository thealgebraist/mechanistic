#!/usr/bin/env python3
"""Verify explicit external-assistant state against Transformers 4.57.3."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import shutil
import subprocess
import tempfile
import wave
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor
from transformers.generation.candidate_generator import AssistedCandidateGenerator
from transformers.generation.utils import GenerationMixin

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL = ROOT / "work/whisper_tiny_en"
AUDIO = ROOT / "work/whisper_sample.wav"
CPP = ROOT / "work/whisper_graph_cpp23"
MANIFEST = OUT / "whisper_cpp23_tensor_manifest.tsv"
ASSISTED_SOURCE_HASH = "95f034e33e7f441f1d2fbb0c27fdc50853ef991a27aeacd376e674850d5b65d0"
CANDIDATE_SOURCE_HASH = "7a1aecd451126c1a97eb354a7c31b28e036dc6c5b9351c74fe985d34059c1283"


def source_hash(value) -> str:
    return hashlib.sha256(inspect.getsource(value).encode()).hexdigest()


assert source_hash(GenerationMixin._assisted_decoding) == ASSISTED_SOURCE_HASH
assert source_hash(AssistedCandidateGenerator) == CANDIDATE_SOURCE_HASH

processor = WhisperProcessor.from_pretrained(MODEL, local_files_only=True)
target = WhisperForConditionalGeneration.from_pretrained(
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
prefix = torch.tensor([[50257, 50362]], dtype=torch.long)

CASES = (
    ("full_constant", None, "constant", 3.0, None, 12),
    ("zero_heuristic", 1, "heuristic", 3.0, None, 8),
    ("partial_one_transient", 2, "heuristic_transient", 3.0, None, 10),
    ("partial_two_confidence", 3, "constant", 3.0, 0.9, 10),
    ("eos_full_acceptance", None, "constant", 5.0, None, 448),
)

PATTERN = re.compile(
    r'WHISPER_CPP23_EXTERNAL_ASSISTANT initial_positions=(\d+) tokens=([^ ]*) '
    r'rounds=(\d+) proposed=(\d+) accepted=(\d+) corrections=(\d+) '
    r'target_verifications=(\d+) target_steps=(\d+) assistant_steps=(\d+) '
    r'rollback_events=(\d+) rollback_positions=(\d+) target_cache=(\d+) '
    r'assistant_cache=(\d+) eos=(\d) maximum=(\d) proposal_trace=([^ ]*) '
    r'accepted_trace=([^ ]*) correction_trace=([^ ]*) round_trace=([^ ]*) '
    r'graph_nodes_visited=(\d+) text="(.*)"'
)


def configuration(maximum: int) -> GenerationConfig:
    value = GenerationConfig.from_model_config(target.config)
    value.max_length = maximum
    value.eos_token_id = 50256
    value.pad_token_id = 50256
    value.decoder_start_token_id = 50257
    value.suppress_tokens = target.generation_config.suppress_tokens
    value.begin_suppress_tokens = target.generation_config.begin_suppress_tokens
    value.return_dict_in_generate = True
    return value


def parse_ints(value: str) -> list[int]:
    return [] if not value else [int(item) for item in value.split(",")]


def parse_proposals(value: str) -> list[list[int]]:
    return [parse_ints(item) for item in value.split("/")] if value else []


def patched_checkpoint(directory: Path, zero_position: int | None) -> tuple[Path, Path]:
    checkpoint = directory / "model.safetensors"
    manifest = directory / "manifest.tsv"
    shutil.copyfile(MODEL / "model.safetensors", checkpoint)
    rows = [line.split("\t") for line in MANIFEST.read_text().splitlines()]
    if zero_position is not None:
        row = next(item for item in rows[1:] if item[0] == "model.decoder.embed_positions.weight")
        begin, end = int(row[2]), int(row[3])
        with checkpoint.open("r+b") as output:
            output.seek(begin + zero_position * 384 * 4)
            output.write(bytes(384 * 4))
            output.seek(begin)
            raw = output.read(end - begin)
        row[6] = f"{zlib.crc32(raw) & 0xFFFFFFFF:08x}"
    manifest.write_text("\n".join("\t".join(row) for row in rows) + "\n")
    return checkpoint, manifest


def reference_case(
    zero_position: int | None,
    schedule: str,
    budget: float,
    confidence: float | None,
    maximum: int,
) -> tuple[dict, WhisperForConditionalGeneration]:
    assistant = WhisperForConditionalGeneration.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.float32
    ).eval()
    if zero_position is not None:
        with torch.no_grad():
            assistant.model.decoder.embed_positions.weight[zero_position].zero_()
    assistant.generation_config.num_assistant_tokens = budget
    assistant.generation_config.num_assistant_tokens_schedule = schedule
    assistant.generation_config.assistant_confidence_threshold = confidence
    records: list[dict] = []
    holder: list[AssistedCandidateGenerator] = []
    original_get = AssistedCandidateGenerator.get_candidates
    original_update = AssistedCandidateGenerator.update_candidate_strategy

    def recording_get(self, input_ids):
        holder[:] = [self]
        before = self.assistant_kwargs.get("past_key_values")
        result = original_get(self, input_ids)
        after = self.assistant_kwargs["past_key_values"]
        records.append(
            {
                "input_length": int(input_ids.shape[1]),
                "budget_before": float(self.num_assistant_tokens),
                "proposal": result[0][0, input_ids.shape[1] :].tolist(),
                "cache_before": 0 if before is None else int(before.get_seq_length()),
                "cache_after_proposal": int(after.get_seq_length()),
            }
        )
        return result

    def recording_update(self, input_ids, scores, num_matches):
        count = int(num_matches)
        records[-1]["accepted"] = count
        records[-1]["correction"] = int(
            input_ids[0, records[-1]["input_length"] + count]
        )
        result = original_update(self, input_ids, scores, num_matches)
        records[-1]["budget_after"] = float(self.num_assistant_tokens)
        return result

    with (
        patch.object(AssistedCandidateGenerator, "get_candidates", recording_get),
        patch.object(
            AssistedCandidateGenerator,
            "update_candidate_strategy",
            recording_update,
        ),
        torch.inference_mode(),
    ):
        output = GenerationMixin.generate(
            target,
            inputs=inputs.input_features,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=prefix,
            generation_config=configuration(maximum),
            assistant_model=assistant,
        )
    complete = output.sequences[0].tolist()
    suffix = complete[prefix.shape[1] :]
    terminated_by_eos = bool(suffix and suffix[-1] == 50256)
    if terminated_by_eos:
        suffix = suffix[:-1]
    return (
        {
            "tokens": suffix,
            "rounds": records,
            "target_cache": int(output.past_key_values.get_seq_length()),
            "assistant_cache": int(
                holder[0].assistant_kwargs["past_key_values"].get_seq_length()
            ),
            "terminated_by_eos": terminated_by_eos,
            "terminated_by_maximum": len(complete) == maximum and not terminated_by_eos,
        },
        assistant,
    )


rows = []
with tempfile.TemporaryDirectory(prefix="whisper-external-assistant-") as temporary:
    temporary = Path(temporary)
    for name, zero_position, schedule, budget, confidence, maximum in CASES:
        reference, assistant = reference_case(
            zero_position, schedule, budget, confidence, maximum
        )
        case_directory = temporary / name
        case_directory.mkdir()
        checkpoint, manifest = patched_checkpoint(case_directory, zero_position)
        command = [
            str(CPP),
            "--external-assistant",
            str(MODEL / "model.safetensors"),
            str(MANIFEST),
            str(checkpoint),
            str(manifest),
            str(AUDIO),
            str(OUT / "whisper_cpp23_hann_f32.bin"),
            str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
            str(OUT / "whisper_cpp23_token_manifest.tsv"),
            str(OUT / "whisper_cpp23_token_bytes.bin"),
            "50257,50362",
            str(budget),
            schedule,
            "none" if confidence is None else str(confidence),
            str(maximum),
        ]
        line = subprocess.check_output(command, text=True).strip()
        match = PATTERN.fullmatch(line)
        assert match, line
        actual_tokens = parse_ints(match.group(2))
        actual_proposals = parse_proposals(match.group(16))
        actual_accepted = parse_ints(match.group(17))
        actual_corrections = parse_ints(match.group(18))
        expected_proposals = [round_["proposal"] for round_ in reference["rounds"]]
        expected_accepted = [round_["accepted"] for round_ in reference["rounds"]]
        expected_corrections = [round_["correction"] for round_ in reference["rounds"]]
        assert actual_tokens == reference["tokens"], (name, actual_tokens, reference["tokens"])
        assert actual_proposals == expected_proposals, (name, actual_proposals, expected_proposals)
        assert actual_accepted == expected_accepted, (name, actual_accepted, expected_accepted)
        assert actual_corrections == expected_corrections, (name, actual_corrections, expected_corrections)
        assert int(match.group(12)) == reference["target_cache"], name
        assert int(match.group(13)) == reference["assistant_cache"], name
        assert bool(int(match.group(14))) == reference["terminated_by_eos"], name
        assert bool(int(match.group(15))) == reference["terminated_by_maximum"], name
        assert int(match.group(20)) == 74
        first_accepted = expected_accepted[0]
        rows.append(
            {
                "case": name,
                "assistant_fixture": "pinned checkpoint"
                if zero_position is None
                else f"position_embedding_row_{zero_position}_zeroed",
                "schedule": schedule,
                "initial_budget": budget,
                "confidence_threshold": confidence,
                "maximum_positions": maximum,
                "first_proposal": expected_proposals[0],
                "first_accepted": first_accepted,
                "first_acceptance_class": (
                    "ZERO"
                    if first_accepted == 0
                    else "FULL"
                    if first_accepted == len(expected_proposals[0])
                    else "PARTIAL"
                ),
                "tokens": actual_tokens,
                "round_count": int(match.group(3)),
                "proposed_tokens": int(match.group(4)),
                "accepted_candidate_tokens": int(match.group(5)),
                "correction_tokens": int(match.group(6)),
                "target_verification_rounds": int(match.group(7)),
                "target_decoder_steps": int(match.group(8)),
                "assistant_decoder_steps": int(match.group(9)),
                "rollback_events": int(match.group(10)),
                "rollback_positions": int(match.group(11)),
                "target_cache_position": int(match.group(12)),
                "assistant_cache_position": int(match.group(13)),
                "terminated_by_eos": bool(int(match.group(14))),
                "terminated_by_maximum": bool(int(match.group(15))),
                "graph_nodes_visited": int(match.group(20)),
            }
        )

artifact = {
    "certificate": "WHISPER_CPP23_EXTERNAL_ASSISTANT_COMMON_VOCABULARY_1",
    "transformers_version": "4.57.3",
    "assisted_decoding_source_sha256": ASSISTED_SOURCE_HASH,
    "candidate_generator_source_sha256": CANDIDATE_SOURCE_HASH,
    "case_count": len(rows),
    "all_complete_tokens_exact": True,
    "all_proposal_stacks_exact": True,
    "all_acceptance_counts_exact": True,
    "all_correction_tokens_exact": True,
    "all_final_cache_positions_exact": True,
    "all_graph_nodes_visited": all(row["graph_nodes_visited"] == 74 for row in rows),
    "first_acceptance_classes": sorted({row["first_acceptance_class"] for row in rows}),
    "rows": rows,
    "scope": (
        "Batch-one deterministic external assisted decoding with a common Whisper "
        "Tiny English vocabulary, separate target/assistant encoder and dynamic cache "
        "ownership, constant/heuristic/transient budgets, confidence stopping, accepted "
        "prefixes, correction tokens, rollback, maximum-length and EOS termination. "
        "Three deliberately perturbed position-embedding fixtures test zero and partial "
        "acceptance; they are state-machine fixtures, not useful speech assistants. "
        "Different-tokenizer UAG, sampled speculative rejection sampling, early-exit "
        "self-assistance, and a genuinely smaller trained assistant remain open."
    ),
}
OUT.mkdir(exist_ok=True)
(OUT / "whisper_cpp23_external_assistant.json").write_text(
    json.dumps(artifact, indent=2) + "\n"
)
table = "\n".join(
    f"| `{row['case']}` | `{row['schedule']}` | {row['initial_budget']:g} | "
    f"`{row['first_proposal']}` | {row['first_accepted']} | "
    f"{row['first_acceptance_class']} | {row['rollback_positions']} | "
    f"{row['target_cache_position']}/{row['assistant_cache_position']} | yes |"
    for row in rows
)
(OUT / "WHISPER_CPP23_EXTERNAL_ASSISTANT.md").write_text(
    f"""# External assistant product-state graph

The C++23 interpreter now runs target and assistant Whisper models as a typed product state: each owns an encoder memory and dynamic decoder cache; proposal tokens cross a common-vocabulary edge; target verification emits accepted-prefix or correction transitions; unused assistant positions are cropped before reconciliation. The source oracles are Transformers 4.57.3 `_assisted_decoding` (`{ASSISTED_SOURCE_HASH}`) and `AssistedCandidateGenerator` (`{CANDIDATE_SOURCE_HASH}`).

| case | schedule | initial budget | first proposal | accepted | class | rolled-back positions | final target/assistant cache | exact |
|---|---|---:|---|---:|---|---:|---:|---:|
{table}

All {len(rows)} complete sequences, every proposal stack, accepted-prefix length, correction token, and final target/assistant cache position match the pinned Python implementation. The cases include full, zero, and partial first-round acceptance plus maximum-length and EOS termination. The zero/partial fixtures alter one position-embedding row solely to force finite rollback branches; they are not claimed to be trained or useful assistants.

This is meaningful progress, not closure of assistant generation. Different-tokenizer UAG, sampled speculative rejection sampling, early-exit self-assistance, adaptive ROC confidence, a genuinely smaller trained checkpoint, and target multi-token forward fusion remain open.
"""
)
print(json.dumps({key: artifact[key] for key in (
    "certificate", "case_count", "all_complete_tokens_exact",
    "all_proposal_stacks_exact", "all_acceptance_counts_exact",
    "all_correction_tokens_exact", "all_final_cache_positions_exact",
    "all_graph_nodes_visited", "first_acceptance_classes",
)}, indent=2))
