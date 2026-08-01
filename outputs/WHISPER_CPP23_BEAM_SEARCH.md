# Explicit C++23 beam-state graph

Each live beam is represented by a token stack, cumulative log probability, parent lineage, and a branchable four-layer self/cross-attention cache. At each step the graph computes log-softmax mass, applies Whisper policy masks, constructs the global beam×vocabulary frontier, keeps the top `2B` candidates, separates completed from live states, copies only selected caches, and ranks completed hypotheses by length-normalized score.

| case | beams | returned | max positions | length penalty | cache branches | max score error | ranked sequences exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| `two_beam_ranked_frontier` | 2 | 2 | 8 | 1.0 | 10 | 6.59823e-05 | yes |
| `four_beam_ranked_frontier` | 4 | 4 | 8 | 1.0 | 20 | 6.61019e-05 | yes |
| `two_beam_eos_finalization` | 2 | 1 | 448 | 1.0 | 46 | 5.68254e-05 | yes |
| `all_finished_early_stop` | 2 | 1 | 448 | 0.7 | 46 | 0.000145584 | yes |
| `canonical_never_stop` | 2 | 1 | 448 | 1.0 | 240 | 5.68254e-05 | yes |

All ranked token sequences match the pinned Transformers GenerationMixin beam implementation exactly. Worst normalized sequence-score error is `0.000145584139`. The four-beam finite frontier contains genuinely different alternatives, so this is not merely replaying greedy output. Scope is standard, deterministic, batch-one beam search; grouped, constrained, and sampled beam variants remain separate work.
