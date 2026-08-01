#!/usr/bin/env python3
"""Verify Whisper-specific rejection and ignored-generation semantics."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import warnings
import wave
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor, logging

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL = ROOT / "work/whisper_tiny_en"
AUDIO = ROOT / "work/whisper_sample.wav"
CPP = ROOT / "work/whisper_graph_cpp23"
REFERENCE = ROOT / "work/dola_reference"
DOLA_REVISION = "af6cdc351e7e0bd28a86ce32aac461494a09a9c1"
DOLA_HASH = "ea3651c5b87a1a67443d8ed349a4f57fdbdc75bcabf43ae5d15354cca46b5d4e"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


source = REFERENCE / "custom_generate/generate.py"
if not source.exists():
    hf_hub_download(
        "transformers-community/dola",
        "custom_generate/generate.py",
        revision=DOLA_REVISION,
        local_dir=REFERENCE,
    )
if sha256(source) != DOLA_HASH:
    raise RuntimeError(f"DoLa reference hash mismatch: {source}")
sys.path.insert(0, str(REFERENCE))
from custom_generate.generate import generate as dola_generate  # noqa: E402

logging.set_verbosity_error()
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


def cpp(field: str, value: str) -> tuple[str, str]:
    line = subprocess.check_output(
        [str(CPP), "--generation-applicability", field, value], text=True
    ).strip()
    prefix = f"WHISPER_CPP23_GENERATION_APPLICABILITY field={field} status="
    assert line.startswith(prefix), line
    status, reason = line[len(prefix) :].split(" reason=", 1)
    return status, reason


rows = []
for label, request, encoded in (
    ("dola_low", "low", "low"),
    ("dola_high", "high", "high"),
    ("dola_explicit", [0, 2], "0,2"),
):
    config = GenerationConfig.from_model_config(model.config)
    config.dola_layers = request
    config.max_length = 4
    config.eos_token_id = 50256
    config.pad_token_id = 50256
    config.decoder_start_token_id = 50257
    try:
        with torch.inference_mode():
            dola_generate(
                model,
                inputs=inputs.input_features,
                attention_mask=inputs.attention_mask,
                generation_config=config,
            )
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("DoLa unexpectedly accepted Whisper")
    assert "only available for decoder-only models" in message
    status, reason = cpp("dola_layers", encoded)
    assert status == "MODEL_REJECTED" and reason == "encoder_decoder_model"
    rows.append(
        {
            "case": label,
            "field": "dola_layers",
            "value": request,
            "transformers_behavior": "ValueError: decoder-only models required",
            "cpp23_status": status,
            "cpp23_reason": reason,
            "behavior_exact": True,
            "constructor": "RejectDolaForEncoderDecoder",
        }
    )

for scale in (0.5, 1.5):
    try:
        with torch.inference_mode():
            model.generate(
                inputs.input_features,
                attention_mask=inputs.attention_mask,
                max_new_tokens=2,
                guidance_scale=scale,
            )
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("guidance unexpectedly accepted Whisper")
    assert "mel input features" in message and "found 1" in message
    status, reason = cpp("guidance_scale", str(scale))
    assert status == "MODEL_REJECTED"
    assert reason == "unconditional_branch_requires_mel_features"
    rows.append(
        {
            "case": f"guidance_{scale}",
            "field": "guidance_scale",
            "value": scale,
            "transformers_behavior": "ValueError: unconditional token IDs are invalid Mel features",
            "cpp23_status": status,
            "cpp23_reason": reason,
            "behavior_exact": True,
            "constructor": "RejectUnbatchedGuidanceForMelEncoder",
        }
    )

with torch.inference_mode():
    baseline = model.generate(
        inputs.input_features,
        attention_mask=inputs.attention_mask,
        max_new_tokens=6,
    ).tolist()

for field, values, warning_fragment in (
    (
        "encoder_repetition_penalty",
        (0.8, 1.2),
        "encoder_repetition_penalty` requires some form of `input_ids",
    ),
    (
        "encoder_no_repeat_ngram_size",
        (1, 3),
        "encoder_no_repeat_ngram_size` requires some form of `input_ids",
    ),
):
    for value in values:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            with torch.inference_mode():
                output = model.generate(
                    inputs.input_features,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=6,
                    **{field: value},
                ).tolist()
        messages = [str(item.message) for item in captured]
        assert any(warning_fragment in message for message in messages), messages
        assert output == baseline
        status, reason = cpp(field, str(value))
        assert status == "MODEL_IGNORED"
        assert reason == "continuous_encoder_has_no_input_ids"
        rows.append(
            {
                "case": f"{field}_{value}",
                "field": field,
                "value": value,
                "transformers_behavior": "warning then ignored; output equals baseline",
                "cpp23_status": status,
                "cpp23_reason": reason,
                "behavior_exact": True,
                "constructor": "IgnoreEncoderTokenPenaltyWithoutEncoderTokenIds",
            }
        )

artifact = {
    "certificate": "WHISPER_CPP23_MODEL_APPLICABILITY_1",
    "reference_repository": "transformers-community/dola",
    "reference_revision": DOLA_REVISION,
    "reference_generate_sha256": DOLA_HASH,
    "case_count": len(rows),
    "rejected_case_count": sum(row["cpp23_status"] == "MODEL_REJECTED" for row in rows),
    "ignored_case_count": sum(row["cpp23_status"] == "MODEL_IGNORED" for row in rows),
    "all_behaviors_exact": all(row["behavior_exact"] for row in rows),
    "cases": rows,
    "scope": (
        "Whisper-specific behavior for DoLa, unbatched classifier-free guidance, "
        "and encoder-token repetition processors under Transformers 4.57.3."
    ),
}
OUT.mkdir(exist_ok=True)
(OUT / "whisper_cpp23_model_applicability.json").write_text(
    json.dumps(artifact, indent=2) + "\n"
)
table = "\n".join(
    f"| `{row['case']}` | `{row['field']}` | `{row['value']}` | "
    f"{row['transformers_behavior']} | `{row['cpp23_status']}` |"
    for row in rows
)
(OUT / "WHISPER_CPP23_MODEL_APPLICABILITY.md").write_text(
    f"""# Whisper-specific generation applicability

Not every generic text-generation option denotes an executable path for Whisper. A complete typed conversion must preserve those boundaries instead of silently applying a different algorithm.

| case | field | value | pinned Transformers behavior | C++23 classification |
|---|---|---:|---|---|
{table}

DoLa is source-pinned at revision `{DOLA_REVISION}` with SHA-256 `{DOLA_HASH}`. Its implementation rejects encoder-decoder models before selecting premature layers. Unbatched classifier-free guidance sends token IDs through the unconditional model branch; Whisper interprets that positional input as Mel features and rejects its length. Encoder repetition processors require encoder token IDs, which Whisper's continuous audio encoder does not provide, so Transformers warns and ignores them.

All `{len(rows)}` tested rejection/no-op behaviors are represented by explicit C++23 ADTs and agree with the pinned Python behavior. This closes these model-applicability fields; it does not count rejection as a neural decoding implementation.
"""
)
print(
    json.dumps(
        {
            key: artifact[key]
            for key in (
                "certificate",
                "case_count",
                "rejected_case_count",
                "ignored_case_count",
                "all_behaviors_exact",
            )
        },
        indent=2,
    )
)
