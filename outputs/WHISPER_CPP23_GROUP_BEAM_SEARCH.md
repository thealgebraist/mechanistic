# Explicit C++23 diverse-group beam graph

The C++ graph partitions live hypotheses into typed groups. Each group owns branchable token/K/V states and a completed-hypothesis set. At every position, groups advance in order; a Hamming transport edge subtracts the configured penalty for tokens selected by earlier groups at that same position. Completion is tracked per group, then all group hypotheses are globally ranked.

The oracle is revision `1a281620f7c5fa711c6a44d61c42a4e3a9c2098b` of `transformers-community/group-beam-search`, whose compatible 4.57-era `generate.py` hash is `07cb918df0a9298b89debb926b672bf8fd688cc2e66ba97a756cd04c12d02b42`. Repository head is intentionally not used because its Transformers-v5 cache call is incompatible with the pinned 4.57.3 model runtime.

| case | beams | groups | returned | max positions | diversity | max score error | ranked sequences exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| `four_beams_two_groups` | 4 | 2 | 4 | 8 | 0.5 | 6.59823e-05 | yes |
| `strong_diversity` | 4 | 2 | 4 | 10 | 2.0 | 0.000116822 | yes |
| `six_beams_three_groups` | 6 | 3 | 6 | 8 | 1.0 | 6.59823e-05 | yes |
| `eos_finalization` | 4 | 2 | 4 | 448 | 0.5 | 5.69228e-05 | yes |
| `all_finished_stopping` | 4 | 2 | 2 | 448 | 0.5 | 0.000145584 | yes |
| `canonical_stopping` | 4 | 2 | 2 | 448 | 0.5 | 5.68254e-05 | yes |

All `6` complete ranked sequence sets match exactly. Worst normalized score error is `0.000145584139`. This finite certificate covers deterministic batch-one grouped beam search; constrained and sampled beam variants remain separate algorithms.
