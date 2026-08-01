# SynthID state through constrained beam rows

The pinned constrained-search loop calls its logits processors on all beam rows before constraint-bank selection. The C++23 graph therefore keeps SynthID state by logical processor row, while token/K/V ancestry and constraint-machine progress follow selected parent edges. These are intentionally distinct state transports.

| case | beams | constraint | calls | row states | max score error | ranked sequences exact | state exact |
|---|---:|---|---:|---:|---:|---:|---:|
| `single_token_phrase` | 4 | `p:25996` | 10 | 40 | 0.00243354 | yes | yes |
| `multi_token_phrase` | 4 | `p:1976,37052` | 16 | 64 | 1.8701e-05 | yes | yes |
| `startup_skip` | 4 | `p:25996` | 10 | 40 | 4.0291e-05 | yes | yes |
| `repeated_empty_context` | 4 | `d:25996|17180` | 10 | 40 | 5.46016e-05 | yes | yes |
| `signed_hash_no_history` | 4 | `p:10912` | 10 | 40 | 0.00194836 | yes | yes |

All `5` configurations reproduce the complete ranked sequence tensor. Every signed context hash, repeated-context decision, and startup-skip decision matches the source processor exactly. Worst normalized score error is `0.00243353943`.

The oracle is revision `57fb32700aa9933f2e5077030f479d4931e56267` of `transformers-community/constrained-beam-search`. Its three relevant source files are hash-pinned in the JSON artifact.
