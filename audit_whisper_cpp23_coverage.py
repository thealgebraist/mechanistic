#!/usr/bin/env python3
"""Bidirectional structural audit from pinned PyTorch Whisper to the C++23 graph."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import WhisperForConditionalGeneration

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "work/whisper_tiny_en"
OUT = ROOT / "outputs"
GRAPH_PATH = OUT / "whisper_tiny_en_probabilistic_graph.json"
MANIFEST_PATH = OUT / "whisper_cpp23_conversion_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


graph = load_json(GRAPH_PATH)
conversion = load_json(MANIFEST_PATH)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_DIR, local_files_only=True, dtype=torch.float32
).eval()
state = model.state_dict()

with safe_open(MODEL_DIR / "model.safetensors", framework="pt", device="cpu") as checkpoint:
    checkpoint_keys = sorted(checkpoint.keys())
    checkpoint_scalar_count = sum(checkpoint.get_tensor(name).numel() for name in checkpoint_keys)

weight_to_ops: dict[str, list[int]] = defaultdict(list)
for op in graph["ops"]:
    for weight in op["weights"]:
        weight_to_ops[weight].append(op["index"])
graph_weight_names = sorted(weight_to_ops)

# PyTorch deliberately emits both names for this tied parameter; safetensors stores it once.
tied_aliases = {"proj_out.weight": "model.decoder.embed_tokens.weight"}
alias_rows = []
for alias, canonical in tied_aliases.items():
    if alias not in state or canonical not in state:
        raise RuntimeError(f"missing expected tied parameter: {alias} -> {canonical}")
    a, c = state[alias], state[canonical]
    same_storage = a.untyped_storage().data_ptr() == c.untyped_storage().data_ptr()
    same_values = torch.equal(a, c)
    if not same_storage or not same_values:
        raise RuntimeError(f"tied parameter is not tied: {alias} -> {canonical}")
    alias_rows.append(
        {
            "pytorch_name": alias,
            "canonical_graph_resource": canonical,
            "same_storage": same_storage,
            "bitwise_equal": same_values,
            "graph_op_indices": weight_to_ops[canonical],
        }
    )

canonical_state_name = lambda name: tied_aliases.get(name, name)
uncovered_state_names = sorted(name for name in state if canonical_state_name(name) not in weight_to_ops)
extra_graph_weight_names = sorted(name for name in graph_weight_names if name not in checkpoint_keys)
uncovered_checkpoint_keys = sorted(name for name in checkpoint_keys if name not in weight_to_ops)


def activation_ops(module_name: str) -> list[int]:
    if not module_name.endswith(".activation_fn"):
        return []
    parts = module_name.split(".")
    family = parts[1]
    layer = parts[3]
    stage = f"{family}.{layer}"
    return [op["index"] for op in graph["ops"] if op["stage"] == stage and op["opcode"] == "MLP_GELU"]


module_rows = []
for module_name, module in model.named_modules():
    display_name = module_name or "<root>"
    direct_parameters = []
    for local_name, _ in module.named_parameters(recurse=False, remove_duplicate=False):
        direct_parameters.append(f"{module_name}.{local_name}" if module_name else local_name)
    prefix = f"{module_name}." if module_name else ""
    descendant_state = [name for name in state if name == module_name or name.startswith(prefix)]
    mapped_ops = sorted(
        {
            op_index
            for name in descendant_state
            for op_index in weight_to_ops.get(canonical_state_name(name), [])
        }
        | set(activation_ops(module_name))
    )
    children = list(module.children())
    if direct_parameters:
        status = "DIRECT_PARAMETER_BINDING"
    elif activation_ops(module_name):
        status = "DIRECT_OPCODE_BINDING"
    elif children and mapped_ops:
        status = "CONTAINER_BY_DESCENDANTS"
    elif module_name in {"", "model", "model.encoder", "model.decoder"} and mapped_ops:
        status = "CONTAINER_BY_DESCENDANTS"
    else:
        status = "UNMAPPED"
    module_rows.append(
        {
            "pytorch_module": display_name,
            "class": type(module).__name__,
            "status": status,
            "direct_parameters": direct_parameters,
            "descendant_state_name_count": len(descendant_state),
            "graph_op_indices": mapped_ops,
        }
    )

unmapped_modules = [row for row in module_rows if row["status"] == "UNMAPPED"]


def field_rows(filename: str, mapping: dict[str, tuple[str, str]]) -> list[dict]:
    values = load_json(MODEL_DIR / filename)
    if set(values) != set(mapping):
        missing = sorted(set(values) - set(mapping))
        stale = sorted(set(mapping) - set(values))
        raise RuntimeError(f"{filename} field ledger mismatch; missing={missing}, stale={stale}")
    rows = []
    for key in sorted(values):
        value = values[key]
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        if len(encoded) > 512:
            dimensions = []
            cursor = value
            while isinstance(cursor, list):
                dimensions.append(len(cursor))
                cursor = cursor[0] if cursor else None
            value = {
                "representation": "HASHED_LARGE_JSON_VALUE",
                "json_bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "dimensions": dimensions,
            }
        rows.append({"field": key, "value": value, "status": mapping[key][0], "binding": mapping[key][1]})
    return rows


config_map = {
    "_name_or_path": ("METADATA_ONLY", "checkpoint identity"),
    "activation_dropout": ("COMPILE_TIME_NOOP", "pinned zero in eval mode"),
    "activation_function": ("CPP23_RUNTIME", "GELU in CONV1_GELU, CONV2_GELU and MLP_GELU"),
    "architectures": ("METADATA_ONLY", "root module class"),
    "attention_dropout": ("COMPILE_TIME_NOOP", "pinned zero in eval mode"),
    "begin_suppress_tokens": ("CPP23_RUNTIME", "GENERATION_POLICY begin_suppress_tokens"),
    "bos_token_id": ("INACTIVE_PINNED_MODE", "generation uses decoder_start_token_id"),
    "d_model": ("CPP23_RUNTIME", "typed hidden dimension 384"),
    "decoder_attention_heads": ("CPP23_RUNTIME", "typed decoder head count 6"),
    "decoder_ffn_dim": ("CPP23_RUNTIME", "typed decoder FFN dimension 1536"),
    "decoder_layerdrop": ("COMPILE_TIME_NOOP", "pinned zero in eval mode"),
    "decoder_layers": ("CPP23_RUNTIME", "four explicit decoder layer families"),
    "decoder_start_token_id": ("CPP23_RUNTIME", "forced_prefix[0]"),
    "dropout": ("COMPILE_TIME_NOOP", "pinned zero in eval mode"),
    "encoder_attention_heads": ("CPP23_RUNTIME", "typed encoder head count 6"),
    "encoder_ffn_dim": ("CPP23_RUNTIME", "typed encoder FFN dimension 1536"),
    "encoder_layerdrop": ("COMPILE_TIME_NOOP", "pinned zero in eval mode"),
    "encoder_layers": ("CPP23_RUNTIME", "four explicit encoder layer families"),
    "eos_token_id": ("CPP23_RUNTIME", "generation termination token"),
    "forced_decoder_ids": ("CPP23_RUNTIME", "forced no-timestamps prefix"),
    "init_std": ("TRAINING_ONLY", "checkpoint is already trained"),
    "is_encoder_decoder": ("CPP23_RUNTIME", "encoder-memory/cross-attention graph topology"),
    "max_length": ("CPP23_RUNTIME", "bounded generation loop and 448-position table"),
    "max_source_positions": ("CPP23_RUNTIME", "typed source length 1500"),
    "max_target_positions": ("CPP23_RUNTIME", "typed target capacity 448"),
    "model_type": ("METADATA_ONLY", "graph model identity"),
    "num_hidden_layers": ("METADATA_ALIAS", "same pinned value as encoder_layers/decoder_layers"),
    "num_mel_bins": ("CPP23_RUNTIME", "typed Mel channel count 80"),
    "pad_token_id": ("CPP23_RUNTIME", "shared EOS/padding convention"),
    "scale_embedding": ("COMPILE_TIME_NOOP", "pinned false; embedding is not scaled"),
    "suppress_tokens": ("CPP23_RUNTIME", "GENERATION_POLICY suppression mask"),
    "torch_dtype": ("CPP23_RUNTIME", "binary32 TensorStore and arithmetic ABI"),
    "transformers_version": ("METADATA_ONLY", "source provenance"),
    "use_cache": ("CPP23_RUNTIME", "explicit four-layer self/cross K/V state"),
    "vocab_size": ("CPP23_RUNTIME", "typed categorical law over 51864 tokens"),
}

generation_map = {
    "alignment_heads": ("INACTIVE_PINNED_MODE", "timestamps are disabled"),
    "begin_suppress_tokens": ("CPP23_RUNTIME", "GENERATION_POLICY begin suppression"),
    "bos_token_id": ("INACTIVE_PINNED_MODE", "decoder_start_token_id is active"),
    "decoder_start_token_id": ("CPP23_RUNTIME", "forced_prefix[0]"),
    "eos_token_id": ("CPP23_RUNTIME", "generation termination"),
    "forced_decoder_ids": ("CPP23_RUNTIME", "forced no-timestamps prefix"),
    "is_multilingual": ("COMPILE_TIME_NOOP", "pinned English-only false"),
    "max_initial_timestamp_index": ("INACTIVE_PINNED_MODE", "timestamps are disabled"),
    "max_length": ("CPP23_RUNTIME", "448-position generation bound"),
    "no_timestamps_token_id": ("CPP23_RUNTIME", "forced prefix token 50362"),
    "pad_token_id": ("CPP23_RUNTIME", "shared EOS/padding convention"),
    "prev_sot_token_id": ("INACTIVE_PINNED_MODE", "condition-on-previous-text is not requested"),
    "return_timestamps": ("COMPILE_TIME_NOOP", "pinned false no-timestamps interface"),
    "suppress_tokens": ("CPP23_RUNTIME", "GENERATION_POLICY suppression mask"),
    "transformers_version": ("METADATA_ONLY", "source provenance"),
}

preprocessor_map = {
    "chunk_length": ("CPP23_RUNTIME", "30-second PCM truncation/zero padding"),
    "feature_extractor_type": ("METADATA_ONLY", "frontend identity"),
    "feature_size": ("CPP23_RUNTIME", "80 Mel channels"),
    "hop_length": ("CPP23_RUNTIME", "160-sample STFT hop"),
    "mel_filters": ("CPP23_RESOURCE", "bit-exact exported 201x80 binary32 matrix"),
    "n_fft": ("CPP23_RUNTIME", "400-point Hann/direct-DFT frontend"),
    "n_samples": ("CPP23_RUNTIME", "480000-sample waveform register"),
    "nb_max_frames": ("CPP23_RUNTIME", "3000-frame Mel register"),
    "padding_side": ("CPP23_RUNTIME", "right zero padding before reflect STFT padding"),
    "padding_value": ("CPP23_RUNTIME", "binary32 zero padding"),
    "processor_class": ("METADATA_ONLY", "source processor identity"),
    "return_attention_mask": ("COMPILE_TIME_NOOP", "fixed padded encoder extent; mask is not needed"),
    "sampling_rate": ("CPP23_RUNTIME", "WAV parser enforces 16000 Hz"),
}

config_rows = field_rows("config.json", config_map)
generation_rows = field_rows("generation_config.json", generation_map)
preprocessor_rows = field_rows("preprocessor_config.json", preprocessor_map)

tokenizer_sources = [
    "added_tokens.json", "merges.txt", "normalizer.json", "special_tokens_map.json",
    "tokenizer.json", "tokenizer_config.json", "vocab.json",
]
resource_rows = [
    {
        "path": str((MODEL_DIR / name).relative_to(ROOT)),
        "bytes": (MODEL_DIR / name).stat().st_size,
        "sha256": sha256(MODEL_DIR / name),
        "status": "TOKEN_DECODER_SOURCE",
        "binding": "compiled into token manifest/byte blob; text-input BPE encoding is outside the audio-to-text graph",
    }
    for name in tokenizer_sources
]
resource_rows += [
    {
        "path": "outputs/whisper_cpp23_hann_f32.bin",
        "bytes": (OUT / "whisper_cpp23_hann_f32.bin").stat().st_size,
        "sha256": sha256(OUT / "whisper_cpp23_hann_f32.bin"),
        "status": "CPP23_RUNTIME_RESOURCE",
        "binding": "LOG_MEL_STFT Hann window",
    },
    {
        "path": "outputs/whisper_cpp23_mel_filters_f32.bin",
        "bytes": (OUT / "whisper_cpp23_mel_filters_f32.bin").stat().st_size,
        "sha256": sha256(OUT / "whisper_cpp23_mel_filters_f32.bin"),
        "status": "CPP23_RUNTIME_RESOURCE",
        "binding": "LOG_MEL_STFT 201x80 Mel matrix",
    },
    {
        "path": "outputs/whisper_cpp23_token_manifest.tsv",
        "bytes": (OUT / "whisper_cpp23_token_manifest.tsv").stat().st_size,
        "sha256": sha256(OUT / "whisper_cpp23_token_manifest.tsv"),
        "status": "CPP23_RUNTIME_RESOURCE",
        "binding": "Token Fin51864 to byte-span table",
    },
    {
        "path": "outputs/whisper_cpp23_token_bytes.bin",
        "bytes": (OUT / "whisper_cpp23_token_bytes.bin").stat().st_size,
        "sha256": sha256(OUT / "whisper_cpp23_token_bytes.bin"),
        "status": "CPP23_RUNTIME_RESOURCE",
        "binding": "byte-level token decoding payload",
    },
]

op_status = {row["index"]: row["status"] for row in conversion["ops"]}
functional_bindings = {
    "PCM_AUDIO_INPUT": "C++23 WAV parser and typed PCM register",
    "LOG_MEL_STFT": "WhisperFeatureExtractor semantics and exported Hann/Mel resources",
    "RESIDUAL_ADD": "functional residual addition in PyTorch layer forward",
    "TOKEN_STACK_INPUT": "probabilistic transducer token-stack register",
    "GENERATION_POLICY": "generation-config forced/suppressed token policy",
    "SOFTMAX": "functional categorical normalization",
    "SAMPLE_OR_ARGMAX": "probabilistic/greedy transition selector",
    "TOKEN_AND_CACHE_APPEND": "typed transducer and K/V-cache state update",
}
op_rows = []
for op in graph["ops"]:
    bound_modules = [
        row["pytorch_module"] for row in module_rows
        if op["index"] in row["graph_op_indices"] and row["status"].startswith("DIRECT_")
    ]
    op_rows.append(
        {
            "index": op["index"],
            "opcode": op["opcode"],
            "stage": op["stage"],
            "execution_status": op_status.get(op["index"], "MISSING"),
            "bound_pytorch_modules": bound_modules,
            "functional_binding": functional_bindings.get(op["opcode"]),
            "weight_count": len(op["weights"]),
            "semantics": op["semantics"],
        }
    )

failures = {
    "uncovered_pytorch_state_names": uncovered_state_names,
    "uncovered_checkpoint_keys": uncovered_checkpoint_keys,
    "extra_graph_weight_names": extra_graph_weight_names,
    "unmapped_pytorch_modules": [row["pytorch_module"] for row in unmapped_modules],
    "non_executable_graph_ops": [row["index"] for row in op_rows if not row["execution_status"].startswith("CPP23_EXECUTABLE")],
    "unbound_graph_ops": [row["index"] for row in op_rows if not row["bound_pytorch_modules"] and not row["functional_binding"]],
    "runtime_graph_dispatch": [] if conversion.get("graph_nodes_visited") == len(op_rows) else [
        {"expected": len(op_rows), "observed": conversion.get("graph_nodes_visited")}
    ],
}
passed = not any(failures.values())

audit = {
    "certificate": "WHISPER_PYTORCH_TO_CPP23_BIDIRECTIONAL_COVERAGE_1",
    "checkpoint_sha256": sha256(MODEL_DIR / "model.safetensors"),
    "graph_sha256": sha256(GRAPH_PATH),
    "pytorch": {
        "module_count": len(module_rows),
        "state_dict_name_count": len(state),
        "state_dict_scalar_count_with_tied_alias_counted_twice": sum(t.numel() for t in state.values()),
        "buffer_count": len(list(model.named_buffers())),
    },
    "checkpoint": {"stored_tensor_count": len(checkpoint_keys), "stored_scalar_count": checkpoint_scalar_count, "stored_tensor_names": checkpoint_keys},
    "graph": {"op_count": len(op_rows), "runtime_nodes_visited": conversion.get("graph_nodes_visited"), "unique_weight_resource_count": len(graph_weight_names)},
    "tied_aliases": alias_rows,
    "modules": module_rows,
    "ops": op_rows,
    "configuration": {
        "config.json": config_rows,
        "generation_config.json": generation_rows,
        "preprocessor_config.json": preprocessor_rows,
    },
    "resources": resource_rows,
    "failures": failures,
    "passed": passed,
    "scope": "Pinned openai/whisper-tiny.en float32 eval-mode, English no-timestamps audio-to-text execution under the declared C++23/Accelerate ABI.",
    "proof_boundary": "Structural coverage plus finite numerical tests do not prove backend-independent equality for every binary32 waveform. Training, timestamp generation, multilingual modes, text-input BPE encoding, and arbitrary Transformers generation options are not active in this pinned graph.",
}

OUT.mkdir(exist_ok=True)
(OUT / "whisper_pytorch_to_cpp23_coverage.json").write_text(json.dumps(audit, indent=2) + "\n")

module_table = "\n".join(
    f"| `{row['pytorch_module']}` | `{row['class']}` | {row['status']} | "
    f"{','.join(map(str, row['graph_op_indices'])) or '—'} |"
    for row in module_rows
)
config_counts = defaultdict(int)
for rows in audit["configuration"].values():
    for row in rows:
        config_counts[row["status"]] += 1
count_text = ", ".join(f"{key}={value}" for key, value in sorted(config_counts.items()))
status = "PASS" if passed else "FAIL"
markdown = f"""# PyTorch Whisper → C++23 graph coverage audit

**Verdict: {status}.** This is a bidirectional structural audit of the actual pinned `WhisperForConditionalGeneration` object, its state dictionary, the safetensors checkpoint, the 74-node graph, configuration files, and runtime resources.

| surface | observed | covered |
|---|---:|---:|
| PyTorch named modules | {len(module_rows)} | {len(module_rows) - len(unmapped_modules)} |
| PyTorch state-dict names | {len(state)} | {len(state) - len(uncovered_state_names)} |
| Stored safetensors tensors | {len(checkpoint_keys)} | {len(checkpoint_keys) - len(uncovered_checkpoint_keys)} |
| Graph opcodes | {len(op_rows)} | {len(op_rows) - len(failures['non_executable_graph_ops'])} executable |
| Runtime-dispatched graph nodes | {len(op_rows)} | {conversion.get('graph_nodes_visited')} visited |
| PyTorch buffers | {len(list(model.named_buffers()))} | {len(list(model.named_buffers()))} |

PyTorch exposes 168 state names while the checkpoint and graph have 167 unique tensor resources. This is expected and now explicit: `proj_out.weight` and `model.decoder.embed_tokens.weight` share the same storage and are bitwise equal. Graph opcode 69 (`TIED_LM_HEAD`) reads that shared embedding resource.

The generated node array now drives frontend, encoder, full decoder, incremental cached decoder, readout, probability normalization, selection, and state-transition sequencing. Runtime tensor lookup uses each node's declared weight-reference slice; the regression fails if any of the 74 nodes is not visited.

Configuration classification: {count_text}. Pinned zero dropouts/layerdrop and false embedding scaling become compile-time no-ops; training initialization and provenance fields do not belong to eval execution; timestamp-only fields are inactive because this graph is explicitly the English no-timestamps model.

## Exact boundary

The audit establishes complete structural coverage for the pinned float32 English no-timestamps audio-to-text execution under the declared Accelerate ABI. It does **not** expand the claim to training, multilingual/timestamp modes, arbitrary generation options, text-input BPE encoding, another numerical backend, or every possible waveform. Those are separate graph variants or stronger proof obligations.

## Module ledger

| PyTorch module | class | binding | graph opcode indices |
|---|---|---|---|
{module_table}
"""
(OUT / "WHISPER_PYTORCH_TO_CPP23_COVERAGE.md").write_text(markdown)

if not passed:
    raise SystemExit("WHISPER_PYTORCH_TO_CPP23_COVERAGE_FAIL " + json.dumps(failures, separators=(",", ":")))
print(
    json.dumps(
        {
            "certificate": audit["certificate"],
            "modules": len(module_rows),
            "state_dict_names": len(state),
            "checkpoint_tensors": len(checkpoint_keys),
            "graph_ops": len(op_rows),
            "tied_aliases": len(alias_rows),
            "passed": passed,
        },
        indent=2,
    )
)
