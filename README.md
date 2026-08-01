# Tiny mechanistic text model

This is a deliberately small causal language model written in C++23. It has:

- a learned token embedding table;
- mean/sum context aggregation followed by a 12-unit `tanh` residual-like hidden state;
- a linear vocabulary readout and softmax;
- SGD training on synthetic text-completion examples;
- a forward trace and hidden-unit contribution report;
- a sentence-with-holes probe and candidate comparison.

Build and run:

```sh
c++ -std=c++23 -O2 -Wall -Wextra tiny_text_model.cpp -o tiny_text_model
mkdir -p outputs
./tiny_text_model
```

The concrete probe is:

```text
the keys to the cabinet ___
```

The model predicts `are` with probability 0.9997. It also predicts `are` after appending each of `are`, `is`, or `missing`; this is not a claim that those completions are semantically good. It is a probe of the next state after a candidate token.

The trace in `outputs/tiny_trace.tsv` is descriptive. The per-unit values are readout contributions `W2[are,j] * h[j]`, not causal effects. A genuine ablation experiment would rerun the forward pass with a unit removed and compare the resulting logit/probability. Therefore this toy result supports “the learned representation jointly encodes the training regularity,” but not “unit j represents plurality.”

This is an MSTL-style prototype rather than a transformer: the concrete state is `(token context, embedding sum, hidden vector)`, and an abstract state could quantize hidden activations. Since many contexts can share a quantized state while producing different future outputs, transition probabilities would be induced by the represented-state distribution, exactly as in the attached specification.

## Whisper audio and cochlear quotient experiment

The repository also contains a mechanistic Whisper Tiny English extraction and five 80-node audio frontends: the model's Mel bank, uniform DFT subbands, sparse Goertzel resonators, db4 wavelet packets, and the official Lyon CAR-FAC cochlear model. See:

- `outputs/TIME_FREQUENCY_COCHLEAR_DAG_THEORY.md` for the mathematical review and proof boundary;
- `outputs/AUDIO_FREQUENCY_QUOTIENT_COMPARISON.md` for measured transcription results;
- `outputs/audio_frequency_quotient_dags.svg` for the graph comparison;
- `outputs/audio_frequency_quotient_dags.json` and `outputs/audio_frequency_quotient_nodes.tsv` for machine-readable certificates.
- `outputs/HIERARCHICAL_DISCRETE_AUDIO_FILTER_GRAPH.md` and `outputs/hierarchical_audio_filter_probabilistic_graph.svg` for the nested discrete filter and feature-detector representation.
- `outputs/ACTUAL_AUDIO_FILTER_CIRQUENT_TEST.md` and `outputs/actual_audio_filter_cirquent_trace.svg` for a value-carrying execution trace on one real speech frame.
- `outputs/CARFAC_MEL_TRANSPORT_SUCCESS.md` and `outputs/carfac_mel_transport_success.svg` for the smallest tested temporal CAR-FAC adapter that returns the exact target transcript on this recording.
- `outputs/CARFAC_MEL_TRANSPORT_BENCHMARK.md` and `outputs/carfac_mel_transport_benchmark.json` for measured time, peak RSS, live adapter tensors, and stored-size comparisons on that recording.
- `outputs/WHISPER_CPP23_CONVERSION_PROGRESS.md` and `outputs/whisper_cpp23_conversion_manifest.json` for the executable C++23 encoder, decoder, greedy policy, token decoder, numerical comparison, and explicit remaining-completion ledger.
- `outputs/WHISPER_CPP23_COMPLETION_AUDIT.md` and `outputs/WHISPER_CPP23_MULTIAUDIO.md` for the requirement-by-requirement completion verdict under the declared Accelerate ABI and four additional exact token-sequence comparisons.
- `outputs/WHISPER_PYTORCH_TO_CPP23_COVERAGE.md` and `outputs/whisper_pytorch_to_cpp23_coverage.json` for the bidirectional 126-module, 168-state-name, 167-resource, and 74-opcode coverage ledger, including the tied readout/embedding alias.
- `outputs/WHISPER_CPP23_FORWARD_VARIANTS.md` for full graph-driven logits plus exact short- and long-form generation, both prompt scopes, previous-segment conditioning, progress and vocabulary callbacks, fallback thresholds, typed repetition/no-repeat/forbidden-sequence/minimum/maximum-length policies, timestamp segments, and eight-head cross-attention/DTW token timestamps. `outputs/WHISPER_CPP23_INTERFACE_ADT_LEDGER.md` classifies all 17 PyTorch forward and 27 top-level generation parameters with no unnamed rows.
- `outputs/WHISPER_CPP23_GENERATION_EXTENSION_INVENTORY.md` separates the 74 represented values of the pinned Transformers 4.57.3 generation configuration from alternate generic algorithms that are not yet lowered. The generated C++ table makes this versioned boundary executable rather than treating `**kwargs` as an unbounded claim.
- `outputs/WHISPER_CPP23_SAMPLING_FILTERS.md` compares complete 51,864-token categorical distributions for temperature, top-k, nucleus/top-p, min-p, typical-p, epsilon, eta, and a composed filter chain. It tests probability-law equivalence independently of differing random-number generators.
- `outputs/WHISPER_CPP23_SCORE_POLICIES.md` verifies exact altered generation paths for additive sequence bias, forced beginning/end tokens, exponential EOS decay, invalid-value repair, and final logit normalization.
- `outputs/WHISPER_CPP23_BEAM_SEARCH.md` verifies the explicit multi-hypothesis state graph: cumulative log probability, parent lineages, copied K/V caches, top-`2B` continuation frontiers, EOS finalization, length penalties, and three stopping modes.
- `outputs/WHISPER_CPP23_SYNTHID_BEAM.md` verifies SynthID's unusual persistent beam-row state: six ranked-search configurations with exact signed context hashes, repeated/skip decisions, ranked sequences, and source-equivalent slot behavior after hypothesis reordering.
- `outputs/WHISPER_CPP23_SYNTHID_GROUP_BEAM.md` verifies the source's shared within-group SynthID row tensor across sequential diverse-group calls, including exact state trajectories and processor ordering.
- `outputs/WHISPER_CPP23_SYNTHID_CONSTRAINED_BEAM.md` verifies that SynthID row-slot state remains distinct from constrained-search parent/KV ancestry and constraint-bank progress.
- `outputs/WHISPER_CPP23_GROUP_BEAM_SEARCH.md` verifies diverse-group search with typed per-group live/completed sets, Hamming token-transport penalties between groups, rectangular final tensorization, and global hypothesis ranking against a source-hashed Transformers 4.57-era community implementation.
- `outputs/WHISPER_CPP23_CONSTRAINED_BEAM_SEARCH.md` verifies phrase and disjunctive-trie constraints, replayable per-beam progress banks, forced-advance edges, EOS eligibility, bounded fallback, and rectangular ranking against a source-hashed Transformers 4.57-era community implementation.
- `outputs/WHISPER_CPP23_BEAM_SAMPLING.md` verifies the complete flattened beam-by-vocabulary probability law and the explicit ordered without-replacement candidate transport used by sampled beam search; random bitstreams remain an intentionally separate backend boundary.
- `outputs/WHISPER_CPP23_CONTRASTIVE_SEARCH.md` verifies typed top-k candidate branches, copied K/V state, final-hidden-state cosine edges, confidence-minus-degeneration scores, and six exact complete sequences against a source-pinned Transformers community implementation.
- `outputs/WHISPER_CPP23_MODEL_APPLICABILITY.md` preserves generic-generation boundaries: DoLa and unbatched guidance are rejected for Whisper's encoder-decoder/Mel interface, while encoder-token repetition processors warn and are ignored because audio has no encoder token IDs.
- `outputs/WHISPER_CPP23_STOP_STRINGS.md` verifies an explicit token-byte suffix-overlap matcher for whole-token, cross-token, final-token-overhang, alternative-stop, and ordinary-EOS cases.
- `outputs/WHISPER_CPP23_PROMPT_LOOKUP.md` verifies longest-first prompt n-gram proposals, target validation, accepted speculative prefixes, correction tokens, and dynamic cache transport against pinned Transformers assisted decoding.
- `outputs/WHISPER_CPP23_WATERMARK.md` verifies classic left/self-hash green-set graphs and the stateful SynthID graph: eleven full probability vectors, exact green masks or g-value tensors, exact signed context-hash/repetition/skip trajectories, exact real-audio sequences, and matched invalid-configuration rejection. Separate certificates cover standard, constrained, and diverse-group beam scheduling; sampled-beam SynthID remains open.
- `outputs/WHISPER_CPP23_BATCH.md` verifies one true five-recording Transformers batch against one C++23 process with shared immutable weights and isolated frontend, encoder, decoder-cache, and graph-audit state for every ordered item. This is semantic sequential batching, not vectorized batched matrix multiplication.
- `outputs/WHISPER_CPP23_VS_TRANSFORMERS_BENCHMARK.md` and `outputs/whisper_cpp23_vs_transformers_benchmark.json` for five-run fresh-process wall-time and peak-RSS measurements against the original Transformers execution.

Run `./fetch_google_carfac.sh` once to obtain the pinned official CAR-FAC source, then `./verify_prsl_all.sh` to reproduce and verify the complete experiment.
Run `PYTHONPATH=work/venv/lib/python3.14/site-packages python3 benchmark_whisper_cpp23_vs_transformers.py` to rebuild the native binary and repeat the performance comparison.
