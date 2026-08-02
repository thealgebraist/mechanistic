# External assistant product-state graph

The C++23 interpreter now runs target and assistant Whisper models as a typed product state: each owns an encoder memory and dynamic decoder cache; proposal tokens cross a common-vocabulary edge; target verification emits accepted-prefix or correction transitions; unused assistant positions are cropped before reconciliation. The source oracles are Transformers 4.57.3 `_assisted_decoding` (`95f034e33e7f441f1d2fbb0c27fdc50853ef991a27aeacd376e674850d5b65d0`) and `AssistedCandidateGenerator` (`7a1aecd451126c1a97eb354a7c31b28e036dc6c5b9351c74fe985d34059c1283`).

| case | schedule | initial budget | first proposal | accepted | class | rolled-back positions | final target/assistant cache | exact |
|---|---|---:|---|---:|---|---:|---:|---:|
| `full_constant` | `constant` | 3 | `[1770, 13, 2264]` | 3 | FULL | 0 | 11/10 | yes |
| `zero_heuristic` | `heuristic` | 3 | `[51083, 1770, 13]` | 0 | ZERO | 2 | 7/6 | yes |
| `partial_one_transient` | `heuristic_transient` | 3 | `[1770, 50256]` | 1 | PARTIAL | 0 | 9/8 | yes |
| `partial_two_confidence` | `constant` | 3 | `[1770, 13, 318]` | 2 | PARTIAL | 0 | 9/8 | yes |
| `eos_full_acceptance` | `constant` | 5 | `[1770, 13, 2264, 346, 353]` | 5 | FULL | 0 | 24/24 | yes |

All 5 complete sequences, every proposal stack, accepted-prefix length, correction token, and final target/assistant cache position match the pinned Python implementation. The cases include full, zero, and partial first-round acceptance plus maximum-length and EOS termination. The zero/partial fixtures alter one position-embedding row solely to force finite rollback branches; they are not claimed to be trained or useful assistants.

This is meaningful progress, not closure of assistant generation. Different-tokenizer UAG, sampled speculative rejection sampling, early-exit self-assistance, adaptive ROC confidence, a genuinely smaller trained checkpoint, and target multi-token forward fusion remain open.
