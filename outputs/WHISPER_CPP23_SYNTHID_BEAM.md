# SynthID state through standard beam rows

Transformers' SynthID processor state follows beam-row slots rather than parent hypothesis ancestry. The C++23 graph represents that behavior explicitly: the first single live C++ frontier initializes all `4` logical processor rows identically, then every subsequent row appends the token currently occupying that slot without copying state from the selected parent beam.

| case | beams | calls | flattened row states | repeated | skipped | max score error | exact ranked sequence/state |
|---|---:|---:|---:|---:|---:|---:|---:|
| `two_rows_short` | 2 | 6 | 12 | 0 | 0 | 1.24e-05 | yes |
| `four_rows_short` | 4 | 6 | 24 | 0 | 0 | 0.000103 | yes |
| `startup_skip_rows` | 2 | 6 | 12 | 0 | 6 | 5.43e-05 | yes |
| `repeated_empty_context_rows` | 2 | 6 | 12 | 10 | 0 | 5.63e-05 | yes |
| `signed_zero_history` | 3 | 6 | 18 | 0 | 0 | 2.5e-05 | yes |
| `eos_finalization` | 2 | 22 | 44 | 0 | 0 | 5.16e-06 | yes |

All `6` ranked sequence tensors, signed context hashes, repeated-context decisions, and startup-skip decisions match. Worst normalized sequence-score error is `0.000103`. The one-token n-gram case forces the empty context to repeat on every later call without introducing score ties; the signed case uses negative keys, a negative sampling seed, zero history, and a non-power-of-two table.

This certificate covers standard deterministic beam search. Sampled, constrained, and diverse-group beam scheduling remain separate stateful cross-algorithm checks.
