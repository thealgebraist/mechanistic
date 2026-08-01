# Explicit C++23 sampled-beam probability graph

At each transition, the graph forms one categorical law over the complete live-beam × 51,864-token product. It draws `2B` distinct continuations without replacement, preserving draw order for completion eligibility, then score-ranks the nonfinished sampled states to retain `B` branchable token/K/V states.

| probability case | beams | product states | nonzero support | max probability error | L1 error | support exact |
|---|---:|---:|---:|---:|---:|---:|
| `two_beam_default_top_k` | 2 | 103728 | 50 | 8.9407e-07 | 1.72777e-06 | yes |
| `four_beam_cool_top_k` | 4 | 207456 | 20 | 1.17812e-07 | 1.50545e-07 | yes |
| `four_beam_composed` | 4 | 207456 | 44 | 1.49012e-06 | 3.03768e-06 | yes |
| `three_beam_full_vocabulary` | 3 | 155592 | 51772 | 6.08563e-05 | 6.71e-05 | yes |

| stochastic run | beams | returned | max positions | sampled candidates | max conditional mass error | unique draw sets | reproducible seed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `two_beam_short` | 2 | 2 | 8 | 24 | 1.11e-15 | yes | yes |
| `four_beam_top_k` | 4 | 4 | 12 | 80 | 1.55e-15 | yes | yes |
| `three_beam_warm` | 3 | 2 | 24 | 132 | 2e-15 | yes | yes |
| `two_beam_eos` | 2 | 2 | 448 | 92 | 1.22e-15 | yes | yes |

All four complete first-transition product distributions match Transformers support exactly. Worst maximum probability error is `6.08563423e-05` and worst L1 error is `6.70999504e-05`. All full runs visit 74 graph nodes, conserve each conditional mass within `2e-15`, and never repeat a candidate within a without-replacement draw set.

The RNG boundary is explicit: C++ uses `std::mt19937_64`; PyTorch uses its own CPU generator and multinomial kernel. Semantic equivalence is therefore asserted for the probability law and transition algorithm, not identical random bitstreams.
