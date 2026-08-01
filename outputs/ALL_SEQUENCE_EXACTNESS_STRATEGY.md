# All-sequence exactness strategy

There are two different claims in this project and they must not be conflated.

## 1. Exact probabilistic register representation

The 129-opcode `PRSL-SHARED-BACKEND-EXACT-1` program binds every opcode to the
same pinned callable used by the source FLAN-T5 implementation. Its checkpoint,
graph, backend contract, method sources, argument bindings, and forward schedule
are hashed. Every local error is zero. The final `SOFTMAX` opcode produces the
same probability kernel, and `SAMPLE_PUSH_UPDATE_CACHE` gives the same stochastic
state transition.

By induction over opcodes, one decoding step is identical. By coinduction (or
ordinary induction over every requested finite continuation), the probability
of every finite token trace is identical. This statement has no prompt table,
no sequence-length cap, and no approximation budget. It is exact relative to
the shared ABI.

This is currently the only completed all-sequence route.

## 2. Portable backend-independent approximation

The independent exact-real/IEEE transfer route tries to replace shared backend
calls by portable arithmetic opcodes. Its current one-step total-variation
bound is one, so it is not yet useful. More fundamentally, if a uniform
one-step error `epsilon` is strictly positive, additive trace transport reaches
the probability diameter at horizon `H >= 1/epsilon`. Thus reducing a positive
local error cannot prove a nontrivial bound simultaneously for every finite
continuation length.

An all-horizon portable result therefore needs at least one of:

1. zero one-step error (bit-exact primitive semantics);
2. a proved contraction or forgetting coefficient for the complete cached
   decoder state; or
3. a horizon supplied as part of the approximation specification.

No contraction is currently proved for FLAN-T5. Consequently the portable path
must target bit-exact opcode implementations if it is to satisfy the original
unbounded all-sequence requirement.

## Convex-domain consequence

Weighted ellipsoids and structured errors remain valuable for finite-horizon or
fixed-byte approximations. They cannot by themselves turn a positive per-step
error into unbounded-horizon equivalence. The residual box-separation audit also
shows that the current box relaxation cannot certify a positive energy margin
after the first residual: all 32,128 token embeddings have zero distance under
that relaxation. This result is `UNKNOWN`, not a cancellation counterexample.

The next exactness-oriented implementation step is therefore to specify each
probabilistic register primitive by its bit-level binary32 operational semantics
and prove that the pinned backend refines that semantics. That removes the
shared-callable trust boundary while retaining zero local error.
