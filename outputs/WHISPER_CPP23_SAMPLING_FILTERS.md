# C++23 sampling-filter probability equivalence

The runtime lowers temperature, top-k, nucleus/top-p, min-p, typical-p, epsilon, and eta filters into named C++23 probability transforms. Each row compares the complete `51864`-token first-step categorical distribution with Transformers after the same Whisper suppression policy. Comparing distributions avoids conflating policy equivalence with unrelated PyTorch/C++ random-number generators.

| case | surviving support | argmax token | maximum probability error | L1 probability error | support exact |
|---|---:|---:|---:|---:|---:|
| `temperature` | 51772 | 1770 | 9.71556e-06 | 9.93382e-06 | yes |
| `top_k` | 20 | 1770 | 5.96046e-07 | 1.32195e-06 | yes |
| `top_p` | 1 | 1770 | 0 | 0 | yes |
| `min_p` | 2 | 1770 | 4.17233e-07 | 8.23289e-07 | yes |
| `typical_p` | 1 | 1770 | 0 | 0 | yes |
| `epsilon` | 18 | 1770 | 5.96046e-07 | 1.30892e-06 | yes |
| `eta` | 9 | 1770 | 6.55651e-07 | 1.2574e-06 | yes |
| `composed` | 31 | 1770 | 1.3113e-06 | 2.66812e-06 | yes |

All supports and argmax tokens match exactly. Worst maximum probability error is `9.7155571e-06` and worst L1 error is `9.93381986e-06`. This is a concrete finite equivalence check for one real Whisper state, not a universal floating-point theorem.
