# SynthID state through diverse-group beam calls

The pinned implementation calls the same stateful SynthID processor once per group at each generated position. Its state therefore has `beams / groups` rows and is shared across groups: group 1 updates the state left by group 0, rather than owning an independent watermark state. The C++23 graph makes this unusual sequencing explicit and applies Hamming diversity after SynthID, matching processor-list order.

| case | beams | groups | state rows | processor calls | max score error | ranked sequences exact | state exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| `two_groups` | 4 | 2 | 2 | 12 | 6.85486e-06 | yes | yes |
| `three_groups` | 6 | 3 | 2 | 18 | 3.80311e-05 | yes | yes |
| `startup_skip` | 4 | 2 | 2 | 12 | 6.90822e-05 | yes | yes |
| `repeated_empty` | 4 | 2 | 2 | 12 | 6.48494e-05 | yes | yes |
| `three_rows_per_group` | 6 | 2 | 3 | 12 | 2.50335e-05 | yes | yes |

All `5` ranked sequence tensors match. Every signed context hash, repetition decision, and startup-skip decision matches the pinned source. Worst normalized score error is `6.90821944e-05`.
