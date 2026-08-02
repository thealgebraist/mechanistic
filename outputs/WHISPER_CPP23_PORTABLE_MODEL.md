# Portable C++23 whole-model validation

Both numerical backends execute the complete 74-node graph, validate all 167 checkpoint tensors, pass the embedded greedy and seeded-sampling token assertions, and produce the exact transcript.

| backend | wall time | peak RSS | worst stage max error | logit max error |
|---|---:|---:|---:|---:|
| Accelerate CBLAS binary32 | 1.93 s | 406.2 MiB | 0.00180054 | 8.96454e-05 |
| portable scalar binary32 | 80.92 s | 400.4 MiB | 0.000549316 | 0.000106812 |

On this run, the scalar backend took `41.93x` the wall time and `0.986x` the peak resident memory of Accelerate. These are local measurements, not portable performance guarantees.

The portable run's worst stage maximum absolute error against the PyTorch fixtures is `0.000549316`. This is finite evidence for one checkpoint, compiler, machine, and recording; it is not a universal floating-point equivalence theorem.
