# PRSL theorem statement for FLAN-T5

## Objects

Let `F` be a fixed finite-precision FLAN-T5 checkpoint. Let `D` be a finite prompt domain, `B` a finite branch policy, and `h` a decoder horizon.

`F` is a deterministic polynomial-time transducer when arithmetic, weights, context length, and precision are encoded as finite binary strings. If its decoder samples from its output distribution, it induces a polynomial-time probabilistic transducer.

`PRSL(D,B,h,k)` is a finite probabilistic stack program with:

```text
prompt register
decoder-prefix stack
fuel indexed by h
EMIT_TOPK_FIXED16
PUSH / BRANCH
OTHER
HALT
```

## Finite-domain approximation theorem

Suppose every reachable PRSL state `s` is paired with a concrete FLAN state `f(s)` and satisfies:

```text
TV( project_B(F_emit(f(s))), PRSL_emit(s) ) <= delta
```

and every explicit branch maps to the corresponding PRSL successor with no state-transition error. Then the projected output law for every continuation of length at most `h` satisfies:

```text
TV( F_projected(D, B, h), PRSL(D, B, h) ) <= min(1, h * delta).
```

The proof is by coupling/induction on remaining fuel. At one step the coupling fails with probability at most `delta`; if it succeeds, the successor relation is preserved and the induction hypothesis applies. A union bound gives `h * delta`.

## Current measured instance

The workspace instance uses:

```text
D = 8 prompts
B = top-2 branches, with absorbing OTHER
h = 3
source states = 56
approximate quotient states = 55
delta <= 0.02285801480125124
```

Therefore the certified bound is:

```text
TV <= 3 * 0.02285801480125124
   = 0.06857404440375371.
```

Independent enumeration against FLAN gives a measured maximum sequence-law TV of `0.00001951185705701673`, below the conservative bound.

## Complexity-class interpretation

This theorem is not a claim that all FLAN behavior belongs to a small finite-state graph. It says that a declared finite restriction has a terminating probabilistic transducer with a machine-checkable error bound.

For finite precision, the original FLAN inference computation is in deterministic polynomial time as a function of encoded input, weights, and precision; its sampling version is a polynomial-time probabilistic transducer. `BPP` applies only after converting the transducer into a bounded-error decision problem. PRSL is a compact, interpretable approximation language, not a new complexity class.

## Remaining universal gap

To extend the theorem beyond finite `D`, `B`, and `h`, one needs a distributional or grammar-based domain and an inductive abstraction invariant. Without that, no finite PRSL byte bound can certify arbitrary prompts and unbounded decoding.
