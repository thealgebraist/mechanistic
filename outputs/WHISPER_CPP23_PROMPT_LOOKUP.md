# Explicit prompt-lookup speculative graph

The typed `PromptLookupSearch` scans the current decoder token stack from the largest configured suffix n-gram down to one token and chooses the first earlier left-to-right occurrence with a continuation. That copied continuation becomes a proposal edge. Whisper evaluates each proposed token; the matching prefix is committed to the K/V state, the first mismatch is replaced by the target token, and a fully accepted proposal receives one extra target-model token.

The oracle is Transformers `4.57.3`. `candidate_generator.py` is hash `9809a08720d61cb0e7bb998685e6fb98c4b0e76a3f73cc0cf07603d69a00950b` and the assisted acceptance/cache implementation in `utils.py` is hash `a20024b1e82ed5361a524d238d2197be5407abc91297dd9888c57e8284d63fef`.

| case | proposal width | max n-gram | first proposal | first accepted | all accepted candidates | complete tokens exact |
|---|---:|---:|---|---:|---:|---:|
| `accepted_two_then_correct` | 5 | 1 | `[286, 262, 46329]` | 2 | 2 | yes |
| `single_token_proposals` | 1 | 1 | `[286]` | 1 | 1 | yes |
| `longest_ngram_scan` | 4 | 4 | `[286, 262, 46329]` | 2 | 2 | yes |
| `first_candidate_mismatch` | 5 | 2 | `[1770, 13, 2264, 346, 353]` | 0 | 0 | yes |
| `repeated_phrase` | 4 | 3 | `[3504, 286, 262]` | 0 | 0 | yes |
| `eos_termination` | 5 | 2 | `[286, 262, 46329]` | 2 | 2 | yes |

All `6` first proposals, first accepted-prefix lengths, and complete output sequences match. Across the cases, `7` of `35` proposed token occurrences were accepted. The remaining transitions were explicit target-model corrections rather than silently discarded speculative state.

This finite certificate covers batch-one greedy prompt lookup. Sampled speculative acceptance and an external assistant model are separate algorithms.
