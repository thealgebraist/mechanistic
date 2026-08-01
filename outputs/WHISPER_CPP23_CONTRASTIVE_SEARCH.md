# Explicit C++23 contrastive-search graph

Each decoding position expands a typed `ContrastiveSearch` state into `top_k` candidate K/V branches. Every candidate's final 384-dimensional decoder hidden state is connected to every prior context hidden state by a cosine edge. The transition score is `(1-alpha) * model_probability - alpha * maximum_context_cosine`; only the winning branch is retained.

The oracle is source-pinned revision `89ece6d21c47e6187e86d45d98fd495feadb33cb` of `transformers-community/contrastive-search`, with SHA-256 `ea33addf7128014a238f3210abba3df7f8343acfb3880383f38ce5b34881c3d3`. The C++ path uses the reference's low-memory sequential semantics, while making candidate branches and hidden-state edges explicit.

| case | top-k | alpha | max positions | output tokens | first probability error | first cosine-penalty error | first score error | complete tokens exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `canonical` | 4 | 0.6 | 32 | 23 | 6.0856e-05 | 5.0679e-07 | 2.46467e-05 | yes |
| `narrow_low_penalty` | 2 | 0.2 | 32 | 25 | 6.0856e-05 | 4.61826e-07 | 4.88163e-05 | yes |
| `wide_strong_penalty` | 8 | 0.8 | 18 | 18 | 6.0856e-05 | 5.0679e-07 | 1.25617e-05 | yes |
| `five_candidates` | 5 | 0.45 | 24 | 24 | 6.0856e-05 | 5.0679e-07 | 3.37063e-05 | yes |
| `pure_degeneration` | 3 | 1.0 | 8 | 8 | 6.0856e-05 | 5.0679e-07 | 5.0679e-07 | yes |
| `eos_termination` | 4 | 0.3 | 448 | 24 | 6.0856e-05 | 5.0679e-07 | 4.27364e-05 | yes |

All `6` complete token sequences match the pinned reference exactly. The first decision in every case also has the exact candidate-token set and selected rank. Worst first-decision errors are `6.08559953e-05` for probability, `5.0678952e-07` for degeneration penalty, and `4.88163044e-05` for the combined score.

This is a finite execution certificate under the declared C++23/Accelerate binary32 ABI, not a backend-independent proof for every waveform.
