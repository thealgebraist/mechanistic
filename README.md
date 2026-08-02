# Mechanistic Whisper in C++23

This repository decompiles the pinned `openai/whisper-tiny.en` PyTorch checkpoint into an explicit C++23 probabilistic register graph.

The native runtime includes WAV parsing, Whisper log-Mel preprocessing, both encoder convolutions, four encoder and four decoder transformer blocks, explicit self/cross-attention K/V state, the tied 51,864-token readout, token decoding, timestamps, prompts, batches, and multiple generation/search policies. The generated graph has 74 named nodes, 168 weight references, and complete bindings for all 167 checkpoint tensors.

## Reproduce

On macOS arm64 with a C++23 compiler and Accelerate:

```sh
python3 bootstrap_whisper_cpp23.py --install-python
./verify_whisper_cpp23_graph.sh
```

The bootstrap downloads only commit-pinned assets listed in `whisper_cpp23_assets.json`, verifies every SHA-256 hash, and keeps checkpoints/environments under ignored `work/`. A non-mutating check is:

```sh
python3 bootstrap_whisper_cpp23.py --preflight
```

A successful full run ends with:

```text
WHISPER_CPP23_GRAPH_REGRESSION_OK
```

## Core files

- `whisper_graph_cpp23.cpp` — executable graph interpreter, native audio frontend, encoder/decoder, cache state, and generation algorithms.
- `whisper_interface_adt.hpp` — named algebraic data types for the PyTorch/GenerationConfig interface.
- `portable_backend.hpp` — vendor-neutral scalar binary32 GEMM/dot backend with primitive correspondence tests and a complete-model numerical certificate.
- `generated_whisper_graph.hpp` — generated 74-node graph and weight-reference table.
- `generated_whisper_generation_config.hpp` — generated pinned configuration representation.
- `generate_whisper_cpp23_graph.py` and `export_whisper_cpp23_encoder_fixture.py` — regenerate graph/config bindings and numerical fixtures.
- `verify_whisper_cpp23_graph.sh` — optimized, differential, coverage, and ASan/UBSan regression suite.
- `verify_whisper_cpp23_*.py` — source-pinned differential verifiers for forward variants, sampling, beam algorithms, constraints, watermarking, prompt lookup, batching, and model-specific boundaries.
- `verify_portable_backend.cpp` — bitwise dot/GEMM kernel checks.
- `verify_whisper_cpp23_portable_model.py` — complete 74-node Accelerate/portable differential run with transcript, error, speed, and memory evidence.

## Evidence

- `outputs/WHISPER_CPP23_COMPLETION_AUDIT.md` — requirement-by-requirement status and remaining boundary.
- `outputs/WHISPER_CPP23_CONVERSION_PROGRESS.md` — numerical and graph execution summary.
- `outputs/WHISPER_PYTORCH_TO_CPP23_COVERAGE.md` — bidirectional module/state/checkpoint/opcode coverage.
- `outputs/WHISPER_CPP23_GENERATION_EXTENSION_INVENTORY.md` — every pinned GenerationConfig field and non-default status.
- `outputs/WHISPER_CPP23_CACHE_IMPLEMENTATIONS.md` — dynamic/static cache storage, search-mode routing, and pinned rejection boundaries.
- `outputs/WHISPER_CPP23_EXTERNAL_ASSISTANT.md` — explicit target/assistant product state with proposal, acceptance, correction, schedule, confidence, and rollback traces.
- `outputs/WHISPER_CPP23_VS_TRANSFORMERS_BENCHMARK.md` — fresh-process latency and peak RSS comparison.
- `outputs/whisper_tiny_en_probabilistic_graph.svg` — readable graph overview.

The current measured sample result is exact at the token/transcript level. The explicit C++23 graph runs in 0.811 s median with 297.6 MiB peak RSS versus 3.295 s and 724.1 MiB for fresh Python/PyTorch processes on the measured Apple arm64 machine.

## Proof boundary

The pinned active checkpoint path is structurally complete under the declared macOS Accelerate binary32 ABI, and the repository contains finite differential evidence across multiple recordings and generation configurations. This is not a backend-independent proof for every waveform or every floating-point implementation. `WhisperAudioTokenEquivalence.lean` states the outer categorical-law theorem; its commuting hypotheses still require backend-specific discharge.

The remaining implementation work is tracked in [GitHub issue #10](https://github.com/thealgebraist/mechanistic/issues/10).
