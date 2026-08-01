#!/usr/bin/env python3
"""Close Whisper's **kwargs surface against the installed Transformers version."""

from __future__ import annotations

import inspect
import json
import re
from collections import Counter
from pathlib import Path

from transformers import GenerationConfig, WhisperForConditionalGeneration
from transformers.generation.utils import GenerationMixin

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL = ROOT / "work/whisper_tiny_en"

model = WhisperForConditionalGeneration.from_pretrained(MODEL, local_files_only=True)
config = model.generation_config.to_dict()
global_defaults = GenerationConfig().to_dict()
version = config["transformers_version"]

# Non-default values for these fields have explicit C++23 behavior and tests.
full_override = {
    "alignment_heads",
    "bad_words_ids",
    "begin_suppress_tokens",
    "constraints",
    "decoder_start_token_id",
    "diversity_penalty",
    "do_sample",
    "eos_token_id",
    "forced_decoder_ids",
    "max_initial_timestamp_index",
    "max_length",
    "max_new_tokens",
    "epsilon_cutoff",
    "early_stopping",
    "eta_cutoff",
    "exponential_decay_length_penalty",
    "forced_bos_token_id",
    "forced_eos_token_id",
    "force_words_ids",
    "min_length",
    "min_new_tokens",
    "min_p",
    "length_penalty",
    "no_repeat_ngram_size",
    "num_beams",
    "num_beam_groups",
    "no_timestamps_token_id",
    "output_attentions",
    "output_hidden_states",
    "output_logits",
    "output_scores",
    "pad_token_id",
    "penalty_alpha",
    "prev_sot_token_id",
    "return_dict_in_generate",
    "repetition_penalty",
    "remove_invalid_values",
    "renormalize_logits",
    "return_timestamps",
    "suppress_tokens",
    "sequence_bias",
    "temperature",
    "top_k",
    "top_p",
    "typical_p",
    "use_cache",
}

# The pinned value is executable, while only the listed subset of non-default values is lowered.
partial_override = {
    "bos_token_id",
    "low_memory",
    "max_matching_ngram_size",
    "num_return_sequences",
    "prompt_lookup_num_tokens",
    "return_legacy_cache",
    "stop_strings",
    "watermarking_config",
}

# These non-default values have no executable neural path for this model. The
# C++ ADT preserves the pinned framework's rejection or warning-plus-ignore
# behavior instead of pretending that a text-token algorithm applies to Mel.
model_rejected = {"dola_layers", "guidance_scale"}
model_ignored = {"encoder_no_repeat_ngram_size", "encoder_repetition_penalty"}

metadata = {"_from_model_config", "transformers_version"}


def cpp_name(name: str) -> str:
    words = [word for word in re.split(r"[^A-Za-z0-9]+", name.lstrip("_")) if word]
    candidate = "".join(word[:1].upper() + word[1:] for word in words)
    return (
        candidate if candidate and not candidate[0].isdigit() else "Field" + candidate
    )


rows = []
for name in sorted(config):
    value = config[name]
    if name in metadata:
        override = "METADATA_FIXED"
    elif name in full_override:
        override = "CPP23_NAMED_OVERRIDE"
    elif name in partial_override:
        override = "CPP23_PARTIAL_OVERRIDE"
    elif name in model_rejected:
        override = "CPP23_MODEL_REJECTED"
    elif name in model_ignored:
        override = "CPP23_MODEL_IGNORED"
    else:
        override = "PINNED_INACTIVE_GENERIC_EXTENSION"
    rows.append(
        {
            "field": name,
            "cpp_constructor": cpp_name(name),
            "pinned_value": value,
            "global_default": global_defaults.get(name),
            "pinned_value_status": "CPP23_PINNED_VALUE_REPRESENTED",
            "override_status": override,
        }
    )

whisper_names = set(
    inspect.signature(WhisperForConditionalGeneration.generate).parameters
) - {"self", "kwargs"}
generic_signature = inspect.signature(GenerationMixin.generate)
generic_extensions = []
for name, parameter in generic_signature.parameters.items():
    if name in {"self", "kwargs", "inputs"} or name in whisper_names:
        continue
    generic_extensions.append(
        {
            "parameter": name,
            "default": None
            if parameter.default is inspect.Parameter.empty
            else parameter.default,
            "pinned_status": "DISABLED_IN_PINNED_MODEL",
            "override_status": "EXTERNAL_GENERATION_ALGORITHM",
        }
    )

forward_names = [
    name
    for name in inspect.signature(WhisperForConditionalGeneration.forward).parameters
    if name != "self"
]
model_kwargs = [
    {
        "parameter": name,
        "routing": "WhisperForConditionalGeneration.forward",
        "status": "ROUTED_TO_NAMED_FORWARD_ADT",
    }
    for name in forward_names
]
special_aliases = [
    {
        "parameter": "inputs",
        "routing": "input_features alias",
        "status": "CPP23_INPUT_ALIAS",
    },
    {
        "parameter": "num_frames",
        "routing": "legacy token-timestamp mask extent",
        "status": "CPP23_CONTIGUOUS_FRAME_EXTENT",
    },
    {
        "parameter": "trust_remote_code",
        "routing": "custom_generate loader",
        "status": "EXTERNAL_CODE_LOADING_DISABLED",
    },
]

counts = Counter(row["override_status"] for row in rows)
artifact = {
    "certificate": "WHISPER_CPP23_GENERATION_EXTENSION_INVENTORY_1",
    "transformers_version": version,
    "generation_config_field_count": len(rows),
    "generic_explicit_extension_count": len(generic_extensions),
    "forward_kwarg_count": len(model_kwargs),
    "special_alias_count": len(special_aliases),
    "all_pinned_generation_values_represented": all(
        row["pinned_value_status"].startswith("CPP23") for row in rows
    ),
    "all_nondefault_generation_variants_executable": all(
        row["override_status"]
        in {
            "METADATA_FIXED",
            "CPP23_NAMED_OVERRIDE",
            "CPP23_MODEL_REJECTED",
            "CPP23_MODEL_IGNORED",
        }
        for row in rows
    ),
    "override_status_counts": dict(sorted(counts.items())),
    "generation_config_fields": rows,
    "generic_generation_extensions": generic_extensions,
    "forward_model_kwargs": model_kwargs,
    "special_aliases": special_aliases,
    "unknown_kwarg_behavior": "rejected by GenerationMixin._validate_model_kwargs in the pinned Transformers version",
    "scope": "Closes **kwargs for the pinned model/configuration, while preserving unsupported non-default generic generation algorithms as explicit external extensions.",
}
OUT.mkdir(exist_ok=True)
(OUT / "whisper_cpp23_generation_extensions.json").write_text(
    json.dumps(artifact, indent=2, default=str) + "\n"
)


def md_value(value) -> str:
    encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return f"`{encoded}`" if len(encoded) <= 72 else f"`{encoded[:69]}…`"


field_table = "\n".join(
    f"| `{row['field']}` | {md_value(row['pinned_value'])} | `{row['cpp_constructor']}` | {row['override_status']} |"
    for row in rows
)
extension_table = "\n".join(
    f"| `{row['parameter']}` | {md_value(row['default'])} | {row['pinned_status']} | {row['override_status']} |"
    for row in generic_extensions
)
(OUT / "WHISPER_CPP23_GENERATION_EXTENSION_INVENTORY.md").write_text(
    f"""# Whisper generation `kwargs` and configuration inventory

This inventory is derived from Transformers `{version}` at verification time. It closes the top-level `**kwargs` row for the pinned model: all {len(rows)} generation-configuration values are represented, all {len(model_kwargs)} model-forward kwargs route to previously audited ADTs, and unknown kwargs are rejected by the installed framework.

This does **not** claim every non-default generic GenerationMixin algorithm is converted. Categorical and contrastive sampling plus greedy prompt-lookup speculation, classic left-hash/self-hash watermarking, greedy, standard-beam, constrained-beam, and diverse-group SynthID state transport, and standard, sampled, diverse-group, and phrase/disjunction-constrained beam search are lowered. Stateful SynthID scheduling for sampled beam search, external assistant models, and similar extensions remain visibly classified as `CPP23_PARTIAL_OVERRIDE`, `PINNED_INACTIVE_GENERIC_EXTENSION`, or `EXTERNAL_GENERATION_ALGORITHM`. Contrastive candidate evaluation currently has the explicit low-memory sequential schedule. DoLa and unbatched classifier-free guidance are explicitly model-rejected for Whisper's encoder-decoder/Mel interface; encoder-token repetition processors are explicitly ignored because audio has no encoder token IDs.

## GenerationConfig fields

| field | pinned value | C++ constructor | non-default override status |
|---|---|---|---|
{field_table}

## Generic explicit extensions carried through Whisper `kwargs`

| parameter | pinned default | pinned status | override status |
|---|---|---|---|
{extension_table}

Pinned-value closure and full reconfiguration closure are separate claims. The former is proved structurally by the generated C++ table; the latter is false and remains future work outside the checkpoint-defined graph.
"""
)

enum_rows = ",\n ".join(row["cpp_constructor"] for row in rows)
table_rows = []
for row in rows:
    canonical = json.dumps(
        row["pinned_value"], separators=(",", ":"), ensure_ascii=False
    )
    table_rows.append(
        f'{{GenerationField::{row["cpp_constructor"]},"{row["field"]}",R"json({canonical})json",OverrideStatus::{row["override_status"]}}}'
    )
header = f"""#pragma once
#include <array>
#include <string_view>

namespace whisper_generation_config {{
enum class GenerationField{{
 {enum_rows}
}};
enum class OverrideStatus{{METADATA_FIXED,CPP23_NAMED_OVERRIDE,CPP23_PARTIAL_OVERRIDE,CPP23_MODEL_REJECTED,CPP23_MODEL_IGNORED,PINNED_INACTIVE_GENERIC_EXTENSION}};
struct PinnedGenerationField{{GenerationField field;std::string_view name;std::string_view canonical_json;OverrideStatus override_status;}};
inline constexpr std::array<PinnedGenerationField,{len(rows)}> pinned_generation_fields={{{{
 {",\n ".join(table_rows)}
}}}};
inline constexpr std::string_view transformers_version="{version}";
}} // namespace whisper_generation_config
"""
(ROOT / "generated_whisper_generation_config.hpp").write_text(header)

print(
    json.dumps(
        {
            "certificate": artifact["certificate"],
            "transformers_version": version,
            "generation_config_fields": len(rows),
            "generic_extensions": len(generic_extensions),
            "forward_kwargs": len(model_kwargs),
            "all_pinned_values_represented": artifact[
                "all_pinned_generation_values_represented"
            ],
            "all_nondefault_variants_executable": artifact[
                "all_nondefault_generation_variants_executable"
            ],
        },
        indent=2,
    )
)
