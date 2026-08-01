# Explicit Whisper watermark graph

The C++23 graph now lowers Transformers `4.57.3` watermarking into three named ADT constructors. `LeftHashWatermark` maps the final context token through a keyed MT19937 permutation and adds a bias to a fixed-ratio green vocabulary. `SelfHashWatermark` constructs the pinned 1,000,003-entry key table, examines the top 40 candidates, and adds the bias only when a candidate belongs to its own candidate-conditioned green set. `SynthIDTextWatermark` carries a rolling n-minus-one token context, bounded context-hash history, call count, keyed sampling table, and depth-wise probability tournament.

The PyTorch `2.10.0` CPU permutation at commit `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0` is represented without PyTorch as a low-32-bit MT19937 seed followed by forward Fisher-Yates swaps. This is linear in vocabulary size and does not materialize a history-by-token transition table.

| case | typed scheme | first-step green tokens | maximum probability error | output tokens | exact sequence |
|---|---|---:|---:|---:|---:|
| `left_default` | `LeftHashWatermark` | 12966 | 2.69e-05 | 22 | yes |
| `left_context_two` | `LeftHashWatermark` | 17115 | 2.98e-05 | 22 | yes |
| `left_negative_key_bias` | `LeftHashWatermark` | 5186 | 6.47e-05 | 22 | yes |
| `self_default` | `SelfHashWatermark` | 5 | 5.87e-05 | 22 | yes |
| `self_context_two` | `SelfHashWatermark` | 12 | 4.92e-05 | 22 | yes |
| `self_delayed_context` | `SelfHashWatermark` | 0 | 6.09e-05 | 22 | yes |

All `6` complete real-audio token sequences and all first-step green masks match exactly. The worst full-distribution error is `6.47e-05` across all 51,864 probabilities; this includes the already-audited model-backend floating-point difference. All `9` invalid scheme, ratio, and context cases are rejected by both implementations.

## SynthID state graph

| case | n-gram | key depths | first g-values | state calls | repeated | skipped | maximum probability error | exact sequence/state |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `synthid_default` | 5 | 9 | 466776 | 22 | 0 | 0 | 3.46e-06 | yes |
| `synthid_small_table` | 3 | 3 | 155592 | 23 | 0 | 0 | 5.16e-05 | yes |
| `synthid_skip_initial` | 4 | 2 | 0 | 23 | 0 | 3 | 6.09e-05 | yes |
| `synthid_repeated_debug` | 2 | 1 | 51864 | 28 | 26 | 0 | 1.14e-09 | yes |
| `synthid_signed_seed_keys` | 2 | 2 | 103728 | 23 | 0 | 0 | 2.29e-05 | yes |

All `5` first-step g-value tensors, complete signed context-hash trajectories, repeated-context decisions, startup-skip decisions, and generated token sequences match exactly. The repeated-debug case deliberately reuses the same one-token context; the skip case verifies that no context hash enters history before the configured n-gram startup boundary. Zero-length context history, signed keys, signed sampling seeds, and non-power-of-two sampling tables are included; empty key depth is verified as a matched runtime rejection.

This closes both configuration families for batch-one greedy execution. Stateful SynthID transport through reordered beam rows remains separate cross-algorithm work and is not claimed here.
