# Kernel refinement status

All 947 portable microcode occurrences are classified against the pinned backend.

- 126 structural operations are exact by index semantics.
- 452 elementary binary32 operations have a controlled C++23 reference using
  round-to-nearest, ties-to-even, with contraction disabled.
- 1 categorical sampling operation is exact at the probability-law level;
  equality of token masses, not equality of random-number algorithms, is the
  relevant trace property.
- 25 maximum reductions are exact if all activation values are finite and not
  NaN. This activation invariant remains conditional.
- 260 reduction/matrix occurrences now have a machine-checked portable
  derivation from ordered scalar addition and multiplication. They remain open
  only at the pinned-backend boundary because ATen association, vectorization,
  and fused-multiply-add behavior are not formally matched to that order.
- 83 `rsqrt`, `exp`, and `tanh` occurrences remain open because correctly
  rounded portable transcendental implementations are not yet matched to the
  pinned kernels.

The controlled reference passed bitwise comparison on 2,048 finite binary32
input pairs for add/subtract/multiply/divide and 64 length-64 ordered dot
products. These tests establish implementation correspondence for the sampled
fixtures, not a universal arithmetic theorem.

Lean separately proves that autoregressive trace mass depends only on the
categorical mass function and token-conditioned state transition. Therefore
inverse-CDF and PyTorch sampling need not consume identical random bits; they
need only implement the same categorical law.

The two ledgers operate at different layers. The 129-opcode ledger proves that
the FLAN graph composes under declared primitive relations. The 947-micro-op
ledger determines whether those relations refine the pinned backend. Therefore
zero pending opcode schemas and 343 open backend occurrences are consistent,
not competing completion claims.

The shortest route to portable exactness is now concrete: bind the pinned
reduction/matmul kernels to the proved ordered scalar semantics and provide
correctly rounded implementations for the three transcendental primitives,
then either use those implementations as the model backend or prove refinement
from the pinned backend.
