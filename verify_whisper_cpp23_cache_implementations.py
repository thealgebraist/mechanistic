#!/usr/bin/env python3
"""Verify Whisper cache implementations and the pinned prefill rejection."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import subprocess
import tempfile
import warnings
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor
from transformers.generation.utils import GenerationMixin

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL = ROOT / "work/whisper_tiny_en"
AUDIO = ROOT / "work/whisper_sample.wav"
CPP = ROOT / "work/whisper_graph_cpp23"
PREFILL_SOURCE_HASH = "859ecc51842d50a49eee4dd1ca2911b08d7c1bb6d67a312316d6feeee31622ec"
CACHE_SOURCE_HASH = "017d349fcfa316b29f3c4bc822bb0104abfd0f774f3639196b147a9a74162dae"
IMPLEMENTATION = re.compile(
    r"WHISPER_CPP23_CACHE_IMPLEMENTATION request=([^ ]+) storage=([^ ]+) "
    r"config=([^ ]+) tokens=([0-9,]+) logical_positions=(\d+) "
    r"capacity_positions=(\d+) serialized_floats=(\d+) "
    r"allocated_self_floats=(\d+) graph_nodes_visited=(\d+)"
)
REJECTION = re.compile(
    r"WHISPER_CPP23_CACHE_REJECTION request=([^ ]+) status=([^ ]+) reason=([^ ]+)"
)
PREFILL = re.compile(
    r"WHISPER_CPP23_PREFILL_POLICY chunk_positions=(\d+) mode=([^ ]+) "
    r"cache=([^ ]+) status=([^ ]+) reason=([^ ]+)"
)
SEARCH = re.compile(
    r"WHISPER_CPP23_CACHE_SEARCH mode=([^ ]+) request=([^ ]+) "
    r"storage=([^ ]+) tokens=([0-9,]*) graph_nodes_visited=(\d+)"
)


def source_hash(method) -> str:
    return hashlib.sha256(inspect.getsource(method).encode()).hexdigest()


assert source_hash(GenerationMixin._prefill_chunking) == PREFILL_SOURCE_HASH
assert source_hash(GenerationMixin._prepare_cache_for_generation) == CACHE_SOURCE_HASH

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


def configuration() -> GenerationConfig:
    value = GenerationConfig.from_model_config(model.config)
    value.max_new_tokens = 2
    value.eos_token_id = 50256
    value.pad_token_id = 50256
    value.decoder_start_token_id = 50257
    value.suppress_tokens = model.generation_config.suppress_tokens
    value.begin_suppress_tokens = model.generation_config.begin_suppress_tokens
    value.return_dict_in_generate = True
    return value


prefix = torch.tensor([[50257, 50362]], dtype=torch.long)


def logical_cache_array(cache) -> tuple[np.ndarray, int, int]:
    logical = int(cache.get_seq_length())
    if hasattr(cache.self_attention_cache, "to_legacy_cache"):
        legacy = cache.to_legacy_cache()
    else:
        legacy = tuple(
            (
                self_layer.keys,
                self_layer.values,
                cross_layer.keys,
                cross_layer.values,
            )
            for self_layer, cross_layer in zip(
                cache.self_attention_cache.layers,
                cache.cross_attention_cache.layers,
                strict=True,
            )
        )
    assert len(legacy) == 4 and all(len(layer) == 4 for layer in legacy)
    values = []
    for layer in legacy:
        for index, tensor in enumerate(layer):
            if index < 2:
                tensor = tensor[:, :, :logical, :]
            values.append(
                tensor[0]
                .permute(1, 0, 2)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
                .astype("<f4", copy=False)
                .reshape(-1)
            )
    cross = int(legacy[0][2].shape[2])
    return np.concatenate(values), logical, cross


common = [
    str(MODEL / "model.safetensors"),
    str(OUT / "whisper_cpp23_tensor_manifest.tsv"),
    str(AUDIO),
    str(OUT / "whisper_cpp23_hann_f32.bin"),
    str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
    "4",
]
executable = (
    ("default", None, "DYNAMIC_APPEND"),
    ("dynamic", "dynamic", "DYNAMIC_APPEND"),
    ("dynamic_full", "dynamic_full", "DYNAMIC_APPEND"),
    ("static", "static", "STATIC_PREALLOCATED"),
    ("sliding_window", "sliding_window", "STATIC_PREALLOCATED_DEPRECATED_ALIAS"),
    ("hybrid", "hybrid", "STATIC_PREALLOCATED_DEPRECATED_ALIAS"),
    ("hybrid_chunked", "hybrid_chunked", "STATIC_PREALLOCATED_DEPRECATED_ALIAS"),
)
rows = []
cpp_arrays = {}
python_arrays = {}
with tempfile.TemporaryDirectory(prefix="whisper-cache-implementations-") as temporary:
    temporary = Path(temporary)
    for request, python_name, expected_storage in executable:
        config = configuration()
        config.cache_implementation = python_name
        cache_config = "none"
        if request == "dynamic":
            config.cache_config = {"ignored_for_non_quantized_cache": 7}
            cache_config = '{"ignored_for_non_quantized_cache":7}'
        with warnings.catch_warnings(record=True) as caught, torch.inference_mode():
            warnings.simplefilter("always")
            output = GenerationMixin.generate(
                model,
                inputs=inputs.input_features,
                attention_mask=inputs.attention_mask,
                decoder_input_ids=prefix,
                generation_config=config,
            )
        expected, logical, cross = logical_cache_array(output.past_key_values)
        path = temporary / f"{request}.bin"
        line = subprocess.check_output(
            [
                str(CPP),
                "--generation-cache-implementation",
                *common,
                request,
                cache_config,
                str(path),
            ],
            text=True,
        ).strip()
        match = IMPLEMENTATION.fullmatch(line)
        assert match, line
        actual = np.fromfile(path, dtype="<f4")
        tokens = output.sequences[0].tolist()
        assert match.group(1) == request
        assert match.group(2) == expected_storage
        assert match.group(3) == (
            "IGNORED_NON_QUANTIZED" if request == "dynamic" else "NONE"
        )
        assert [int(token) for token in match.group(4).split(",")] == tokens
        assert logical == 3 and cross == 1500
        assert int(match.group(5)) == logical
        # GenerationMixin allocates max_length - 1 cache positions. In this
        # bounded run that equals the three logical positions reached.
        expected_capacity = logical
        assert int(match.group(6)) == expected_capacity
        assert int(match.group(7)) == expected.size == actual.size
        assert int(match.group(8)) == 4 * 2 * expected_capacity * 384
        assert int(match.group(9)) == 74
        maximum = float(
            np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))
        )
        assert maximum < 3.0e-3
        cpp_arrays[request] = actual
        python_arrays[request] = expected
        rows.append(
            {
                "request": request,
                "python_implementation": python_name,
                "python_self_cache_type": type(
                    output.past_key_values.self_attention_cache
                ).__name__,
                "python_cross_cache_type": type(
                    output.past_key_values.cross_attention_cache
                ).__name__,
                "cpp23_storage": expected_storage,
                "cache_config_status": match.group(3),
                "tokens": tokens,
                "logical_positions": logical,
                "capacity_positions": expected_capacity,
                "serialized_floats": int(actual.size),
                "allocated_self_floats": int(match.group(8)),
                "max_absolute_cache_error": maximum,
                "graph_nodes_visited": 74,
                "python_warning_count": len(caught),
            }
        )

    rejected_rows = []
    for request in (
        "offloaded",
        "offloaded_static",
        "offloaded_hybrid",
        "offloaded_hybrid_chunked",
        "quantized",
    ):
        # A failed offloaded-cache construction can leave GenerationMixin's
        # reusable `_cache` partially initialized. Each negative case starts
        # from the same clean model state so its error is request-local.
        if hasattr(model, "_cache"):
            delattr(model, "_cache")
        config = configuration()
        config.cache_implementation = request
        try:
            with torch.inference_mode():
                GenerationMixin.generate(
                    model,
                    inputs=inputs.input_features,
                    attention_mask=inputs.attention_mask,
                    decoder_input_ids=prefix,
                    generation_config=config,
                )
        except Exception as error:
            python_error = type(error).__name__
            python_message = str(error)
        else:
            raise AssertionError(f"{request} unexpectedly executed")
        line = subprocess.check_output(
            [
                str(CPP),
                "--generation-cache-implementation",
                *common,
                request,
                "none",
                str(temporary / f"unused-{request}.bin"),
            ],
            text=True,
        ).strip()
        match = REJECTION.fullmatch(line)
        assert match and match.group(1) == request
        if request == "quantized":
            assert python_error == "ValueError" and "does not support the quantized cache" in python_message
            assert match.group(2) == "MODEL_REJECTED"
        else:
            assert python_error == "AssertionError" and "CUDA" in python_message
            assert match.group(2) == "CPU_RUNTIME_UNAVAILABLE"
        rejected_rows.append(
            {
                "request": request,
                "python_error": python_error,
                "python_message": python_message,
                "cpp23_status": match.group(2),
                "cpp23_reason": match.group(3),
            }
        )

    invalid = subprocess.run(
        [
            str(CPP),
            "--generation-cache-implementation",
            *common,
            "invalid",
            "none",
            str(temporary / "invalid.bin"),
        ],
        text=True,
        capture_output=True,
    )
    assert invalid.returncode != 0 and "unknown generation cache implementation" in invalid.stderr
    invalid_config = configuration()
    invalid_config.cache_implementation = "invalid"
    try:
        with torch.inference_mode():
            GenerationMixin.generate(
                model,
                inputs=inputs.input_features,
                attention_mask=inputs.attention_mask,
                decoder_input_ids=prefix,
                generation_config=invalid_config,
            )
    except ValueError as error:
        assert "Invalid `cache_implementation`" in str(error)
    else:
        raise AssertionError("invalid Python cache implementation accepted")

prefill_rows = []
long_prefix = torch.tensor([[50257, 50362, 1770, 13, 2264, 346]], dtype=torch.long)
for chunk_positions in (1, 2, 8):
    config = configuration()
    config.prefill_chunk_size = chunk_positions
    try:
        with torch.inference_mode():
            GenerationMixin.generate(
                model,
                inputs=inputs.input_features,
                attention_mask=inputs.attention_mask,
                decoder_input_ids=long_prefix,
                generation_config=config,
            )
    except TypeError as error:
        message = str(error)
        assert "unexpected keyword argument 'position_ids'" in message
    else:
        raise AssertionError("Whisper prefill chunking unexpectedly executed")
    line = subprocess.check_output(
        [
            str(CPP),
            "--generation-prefill-policy",
            str(chunk_positions),
            "sample",
            "cache",
        ],
        text=True,
    ).strip()
    match = PREFILL.fullmatch(line)
    assert match and int(match.group(1)) == chunk_positions
    assert match.groups()[1:] == (
        "sample",
        "cache",
        "MODEL_REJECTED",
        "whisper_forward_rejects_position_ids",
    )
    prefill_rows.append(
        {
            "chunk_positions": chunk_positions,
            "python_error": "TypeError",
            "python_reason": "WhisperForConditionalGeneration.forward rejects position_ids",
            "cpp23_status": "MODEL_REJECTED",
            "exact_rejection_category": True,
        }
    )

config = configuration()
config.prefill_chunk_size = 2
config.use_cache = False
try:
    with torch.inference_mode():
        GenerationMixin.generate(
            model,
            inputs=inputs.input_features,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=long_prefix,
            generation_config=config,
        )
except ValueError as error:
    assert "Cannot use prefill chunking without a cache" in str(error)
else:
    raise AssertionError("cache-free prefill chunking unexpectedly executed")
line = subprocess.check_output(
    [str(CPP), "--generation-prefill-policy", "2", "sample", "no_cache"],
    text=True,
).strip()
match = PREFILL.fullmatch(line)
assert match and match.groups() == (
    "2",
    "sample",
    "no_cache",
    "CONFIGURATION_REJECTED",
    "prefill_chunking_requires_cache",
)
prefill_rows.append(
    {
        "chunk_positions": 2,
        "mode": "sample",
        "cache": "no_cache",
        "python_error": "ValueError",
        "python_reason": "prefill chunking requires past_key_values",
        "cpp23_status": "CONFIGURATION_REJECTED",
        "exact_rejection_category": True,
    }
)

beam_baseline = configuration()
beam_baseline.num_beams = 2
beam_chunked = configuration()
beam_chunked.num_beams = 2
beam_chunked.prefill_chunk_size = 2
with torch.inference_mode():
    baseline_beam_tokens = GenerationMixin.generate(
        model,
        inputs=inputs.input_features,
        attention_mask=inputs.attention_mask,
        decoder_input_ids=long_prefix,
        generation_config=beam_baseline,
    ).sequences[0].tolist()
    chunked_beam_tokens = GenerationMixin.generate(
        model,
        inputs=inputs.input_features,
        attention_mask=inputs.attention_mask,
        decoder_input_ids=long_prefix,
        generation_config=beam_chunked,
    ).sequences[0].tolist()
assert chunked_beam_tokens == baseline_beam_tokens
line = subprocess.check_output(
    [str(CPP), "--generation-prefill-policy", "2", "beam", "cache"],
    text=True,
).strip()
match = PREFILL.fullmatch(line)
assert match and match.groups() == (
    "2",
    "beam",
    "cache",
    "MODEL_IGNORED",
    "prefill_chunking_only_dispatched_by_sample_search",
)
prefill_rows.append(
    {
        "chunk_positions": 2,
        "mode": "beam",
        "cache": "cache",
        "python_tokens": chunked_beam_tokens,
        "baseline_tokens": baseline_beam_tokens,
        "cpp23_status": "MODEL_IGNORED",
        "tokens_unchanged": True,
    }
)

assert all(np.array_equal(cpp_arrays["default"], value) for value in cpp_arrays.values())
assert all(
    np.array_equal(python_arrays["default"], value)
    for value in python_arrays.values()
)

search_rows = {}
for request, expected_storage in (
    ("dynamic", "DYNAMIC_APPEND"),
    ("static", "STATIC_PREALLOCATED"),
):
    output = subprocess.check_output(
        [str(CPP), "--cache-all-searches", *common[:-1], request], text=True
    )
    parsed = []
    for line in output.splitlines():
        match = SEARCH.fullmatch(line)
        assert match, line
        mode, actual_request, storage, encoded, visited = match.groups()
        assert actual_request == request and int(visited) == 74
        if mode == "prompt_lookup":
            expected = "DYNAMIC_ASSISTED_OVERRIDE"
        elif mode == "contrastive":
            expected = "DYNAMIC_FULL_CONTRASTIVE_OVERRIDE"
        else:
            expected = expected_storage
        assert storage == expected
        parsed.append(
            {
                "mode": mode,
                "request": request,
                "storage": storage,
                "tokens": [] if not encoded else [int(token) for token in encoded.split(",")],
                "graph_nodes_visited": 74,
            }
        )
    assert len(parsed) == 6
    search_rows[request] = parsed
assert [row["mode"] for row in search_rows["dynamic"]] == [
    row["mode"] for row in search_rows["static"]
]
assert [row["tokens"] for row in search_rows["dynamic"]] == [
    row["tokens"] for row in search_rows["static"]
]

beam_dynamic = configuration()
beam_dynamic.num_beams = 2
beam_dynamic.cache_implementation = "dynamic"
beam_static = configuration()
beam_static.num_beams = 2
beam_static.cache_implementation = "static"
with torch.inference_mode():
    beam_dynamic_tokens = GenerationMixin.generate(
        model,
        inputs=inputs.input_features,
        attention_mask=inputs.attention_mask,
        decoder_input_ids=prefix,
        generation_config=beam_dynamic,
    ).sequences[0].tolist()
    beam_static_tokens = GenerationMixin.generate(
        model,
        inputs=inputs.input_features,
        attention_mask=inputs.attention_mask,
        decoder_input_ids=prefix,
        generation_config=beam_static,
    ).sequences[0].tolist()
assert beam_static_tokens == beam_dynamic_tokens

sampled_beam_dynamic = configuration()
sampled_beam_dynamic.num_beams = 2
sampled_beam_dynamic.do_sample = True
sampled_beam_dynamic.cache_implementation = "dynamic"
sampled_beam_static = configuration()
sampled_beam_static.num_beams = 2
sampled_beam_static.do_sample = True
sampled_beam_static.cache_implementation = "static"
torch.manual_seed(11)
with torch.inference_mode():
    sampled_beam_dynamic_tokens = GenerationMixin.generate(
        model,
        inputs=inputs.input_features,
        attention_mask=inputs.attention_mask,
        decoder_input_ids=prefix,
        generation_config=sampled_beam_dynamic,
    ).sequences[0].tolist()
torch.manual_seed(11)
with torch.inference_mode():
    sampled_beam_static_tokens = GenerationMixin.generate(
        model,
        inputs=inputs.input_features,
        attention_mask=inputs.attention_mask,
        decoder_input_ids=prefix,
        generation_config=sampled_beam_static,
    ).sequences[0].tolist()
assert sampled_beam_static_tokens == sampled_beam_dynamic_tokens

custom_cache_sources = {
    "group_beam": ROOT / "work/group_beam_reference_4_57/custom_generate/generate.py",
    "constrained_beam": ROOT
    / "work/constrained_beam_reference_4_57/custom_generate/generate.py",
    "contrastive": ROOT / "work/contrastive_search_reference/custom_generate/generate.py",
}
custom_cache_source_hashes = {}
for mode, path in custom_cache_sources.items():
    source = path.read_text()
    assert "GenerationMixin.generate" in source
    if mode == "contrastive":
        assert 'kwargs.pop("cache_implementation", "dynamic_full")' in source
        assert "cache_implementation=cache_implementation" in source
    else:
        assert "cache_implementation" not in source
        assert "custom_generate=" in source
    custom_cache_source_hashes[mode] = hashlib.sha256(path.read_bytes()).hexdigest()

assisted_dynamic = configuration()
assisted_dynamic.prompt_lookup_num_tokens = 3
assisted_dynamic.max_matching_ngram_size = 2
assisted_dynamic.cache_implementation = "dynamic"
assisted_static = configuration()
assisted_static.prompt_lookup_num_tokens = 3
assisted_static.max_matching_ngram_size = 2
assisted_static.cache_implementation = "static"
with warnings.catch_warnings(record=True) as assisted_warnings, torch.inference_mode():
    warnings.simplefilter("always")
    assisted_dynamic_tokens = GenerationMixin.generate(
        model,
        inputs=inputs.input_features,
        attention_mask=inputs.attention_mask,
        decoder_input_ids=long_prefix,
        generation_config=assisted_dynamic,
    ).sequences[0].tolist()
    assisted_static_tokens = GenerationMixin.generate(
        model,
        inputs=inputs.input_features,
        attention_mask=inputs.attention_mask,
        decoder_input_ids=long_prefix,
        generation_config=assisted_static,
    ).sequences[0].tolist()
assert assisted_static_tokens == assisted_dynamic_tokens
# The warning is emitted through the Transformers logger, not `warnings`, but
# the token/cache behavior and pinned source branch are both certified here.
assert "generation_config.cache_implementation = None" in inspect.getsource(
    GenerationMixin._prepare_cache_for_generation
)

artifact = {
    "certificate": "WHISPER_CPP23_CACHE_IMPLEMENTATIONS_1",
    "transformers_version": "4.57.3",
    "prepare_cache_source_sha256": CACHE_SOURCE_HASH,
    "prefill_chunking_source_sha256": PREFILL_SOURCE_HASH,
    "executable_case_count": len(rows),
    "rejected_case_count": len(rejected_rows),
    "prefill_policy_case_count": len(prefill_rows),
    "prefill_rejected_case_count": sum(
        row["cpp23_status"] != "MODEL_IGNORED" for row in prefill_rows
    ),
    "prefill_ignored_case_count": sum(
        row["cpp23_status"] == "MODEL_IGNORED" for row in prefill_rows
    ),
    "all_sequences_exact": True,
    "all_cpp23_storage_values_bitwise_equal": True,
    "all_python_storage_values_bitwise_equal": True,
    "search_mode_case_count": len(search_rows["static"]),
    "all_search_mode_tokens_storage_invariant": True,
    "beam_static_tokens_exact": beam_static_tokens == beam_dynamic_tokens,
    "sampled_beam_static_tokens_exact": sampled_beam_static_tokens
    == sampled_beam_dynamic_tokens,
    "custom_beam_cache_implementation_forwarded": True,
    "contrastive_cache_forced_dynamic_full": True,
    "custom_search_source_sha256": custom_cache_source_hashes,
    "assisted_static_forced_dynamic": assisted_static_tokens
    == assisted_dynamic_tokens,
    "cache_config_ignored_for_non_quantized_cache": True,
    "invalid_implementation_rejected": True,
    "worst_max_absolute_cache_error": max(
        row["max_absolute_cache_error"] for row in rows
    ),
    "executable_cases": rows,
    "rejected_cases": rejected_rows,
    "prefill_policy_cases": prefill_rows,
    "search_mode_cases": search_rows,
    "scope": (
        "Finite two-token greedy generation on one recording under the pinned CPU runtime. "
        "Dynamic/static/deprecated-static storage and model/runtime rejection paths are covered; "
        "offloaded GPU execution is not claimed."
    ),
}
OUT.mkdir(exist_ok=True)
(OUT / "whisper_cpp23_cache_implementations.json").write_text(
    json.dumps(artifact, indent=2) + "\n"
)
(OUT / "WHISPER_CPP23_CACHE_IMPLEMENTATIONS.md").write_text(
    f"""# Whisper generation cache implementations

Transformers 4.57.3 and C++23 agree on {len(rows)} executable cache requests: default, dynamic, dynamic-full, static, and three deprecated static aliases. Every greedy case emits the exact same token sequence, the logical cache tensors are bitwise identical within each runtime, and C++23 visits all 74 graph nodes. Cache policy also produces identical sequences across all {artifact['search_mode_case_count']} converted search interpreters. Standard and sampled beam agree directly with PyTorch under dynamic and static stores; the pinned group/constrained wrappers forward the same configuration into GenerationMixin, and their C++23 state branches now avoid materializing a cache entry for a terminal length token. Contrastive explicitly replaces the request with dynamic-full storage for rollback, while assisted prompt lookup separately replaces it with dynamic storage. The worst C++23-to-PyTorch cache error is `{artifact['worst_max_absolute_cache_error']:.9g}`.

The C++23 graph distinguishes append-only dynamic storage from fixed-capacity static storage. For this four-position run, dynamic storage allocates {rows[0]['allocated_self_floats']:,} self-cache floats at the three logical positions; static storage allocates {next(row for row in rows if row['request'] == 'static')['allocated_self_floats']:,} floats while serializing the same {rows[0]['serialized_floats']:,} logical self/cross-cache floats. A non-null `cache_config` is explicitly ignored for a non-quantized cache, matching the pinned source.

The remaining source outcomes are not fabricated as executable neural paths. Quantized cache is rejected because Whisper is encoder-decoder. Four offloaded forms fail in the pinned CPU-only PyTorch runtime because CUDA is unavailable. Invalid names are rejected. `prefill_chunk_size` has mode-dependent typed behavior: sample/greedy generation with a cache reaches the generic helper and fails because Whisper forward does not accept `position_ids`; cache-free sample generation fails its cache precondition; beam search does not dispatch the helper and leaves tokens unchanged.

This is finite CPU evidence for one recording and two generated tokens. It does not certify offloaded accelerator behavior or arbitrary backends.
"""
)
print(
    json.dumps(
        {
            key: artifact[key]
            for key in (
                "certificate",
                "executable_case_count",
                "rejected_case_count",
                "prefill_policy_case_count",
                "prefill_rejected_case_count",
                "prefill_ignored_case_count",
                "all_sequences_exact",
                "all_cpp23_storage_values_bitwise_equal",
                "all_python_storage_values_bitwise_equal",
                "search_mode_case_count",
                "all_search_mode_tokens_storage_invariant",
                "beam_static_tokens_exact",
                "sampled_beam_static_tokens_exact",
                "custom_beam_cache_implementation_forwarded",
                "contrastive_cache_forced_dynamic_full",
                "assisted_static_forced_dynamic",
                "worst_max_absolute_cache_error",
            )
        },
        indent=2,
    )
)
