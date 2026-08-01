# Explicit C++23 constrained-beam graph

Positive constraints are algebraic values: `ForcedPhrase` or `ForcedDisjunction`. Every beam carries an implicit replayable constraint state consisting of completed constraints, one optional in-progress machine, pending constraints, a progress bank, and the next tokens that advance it. Ordinary top-probability edges and constraint-advance edges are merged, deduplicated, bank-interleaved, and then used to branch token and K/V state.

The oracle is revision `57fb32700aa9933f2e5077030f479d4931e56267` of `transformers-community/constrained-beam-search`. All three Python source files are hash-checked before execution.

| case | beams | returned | max positions | constraint | satisfying outputs | max score error | ranked tensor exact |
|---|---:|---:|---:|---|---:|---:|---:|
| `single_token_force_words` | 4 | 4 | 12 | `p:25996` | 2/4 | 6.10347e-05 | yes |
| `multi_token_phrase` | 4 | 4 | 18 | `p:1976,37052` | 1/4 | 6.25819e-05 | yes |
| `two_required_phrases` | 6 | 4 | 18 | `p:25996;p:17180` | 0/4 | 0.000144002 | yes |
| `two_required_phrases_completed` | 6 | 1 | 30 | `p:25996;p:17180` | 1/1 | 0.000142809 | yes |
| `single_token_disjunction` | 4 | 4 | 18 | `d:25996|17180` | 2/4 | 6.25829e-05 | yes |
| `multi_token_disjunction` | 4 | 1 | 18 | `d:1976,37052|3504,6097` | 1/1 | 6.20183e-05 | yes |
| `direct_constraint_objects` | 4 | 4 | 18 | `p:10912` | 2/4 | 6.20183e-05 | yes |
| `natural_phrase_eos_finalization` | 4 | 1 | 448 | `p:21443` | 1/1 | 5.68254e-05 | yes |

All `8` ranked tensors match exactly, including the reference fallback at short bounds where some returned beams cannot yet satisfy every constraint. Worst normalized score error is `0.000144001862`. This finite certificate covers phrase constraints, multiple required phrases, disjunctions, direct constraint objects, maximum-length fallback, and canonical EOS finalization.
