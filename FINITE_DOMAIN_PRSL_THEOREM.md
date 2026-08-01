# Finite-domain compilation theorem for FLAN-T5

This document states exactly what the current artifacts prove.

## Definitions

Let `D` be a finite set of tokenized prompts, `h` a finite decoding horizon,
and `F(s)` the exact FLAN-T5 next-token distribution for every reachable state

```text
s = (prompt, decoder_prefix),  |prefix| < h.
```

Choose a branch count `B`. At each state retain `T_s`, the `B` largest token
probabilities, and introduce one symbol `OTHER` with mass

```text
p_s(OTHER) = 1 - sum(t in T_s) p_s(t).
```

The finite oracle table contains all states reached by following retained
tokens. A quotient map `alpha` is support-preserving when any two states in one
block have the same labelled retained successors after applying `alpha`.

## Construction theorem

For every finite `D`, finite `h`, and `B`, the oracle table is finite and can be
serialized as a probabilistic stack program. Its machine state is

```text
(quotient_state, decoder_prefix, fuel)
```

and one step samples from `p_s`, pushes a retained token, follows the labelled
successor, or terminates on `OTHER`. If probabilities are quantized with
per-state total-variation error at most `delta`, and quotient emissions differ
from their concrete representatives by at most `delta_q`, then a coupling gives

```text
TV(output_exact[h], output_PRSL[h]) <= min(1, h * (delta + delta_q)).
```

### Proof sketch

Couple both machines at equal quotient states. At a live state, the probability
that their next observable event differs is at most the TV distance between
their emission measures. If the event is equal and not `OTHER`, support
preservation puts both machines in the same quotient block. Thus the coupling
survives one step with probability at least `1-e`, where
`e <= delta + delta_q`. Induction over `h` steps gives failure probability at
most `h*e`; coupling inequality gives the TV result. Fuel decreases at every
step, so the stack program terminates.

## Byte bound

For the current fixed-width binary encoding with two retained branches, a
program with `P` roots and `S` reachable quotient states occupies

```text
6 + 2P + 19S bytes.
```

The header is six bytes, each root is a 16-bit state index, and each state
stores depth, residual mass, two `(token,mass)` pairs, and two labelled edges.
Therefore a compiler that emits only domains satisfying

```text
6 + 2P + 19S <= n
```

has a machine-checkable `max n bytes` guarantee.

## FLAN instantiation

The current manifest instantiates this theorem with:

```text
P = 32
source states = 224
quotient states = 221
h = 3
local emission TV = 0.012649729152361318
union bound = 0.037949187457084005
measured replay TV = 0.0020689571230390917
binary size = 4275 bytes
```

The parameterized pipeline also produced a separate 48-prompt instantiation
(`outputs/flan_domain64_manifest.json`): 336 source states, 328 quotient
states, 6,340-byte binary, local TV `0.01965362020294499`, horizon bound
`0.05896086060883497`, and measured replay TV `0.005291106531076158`. The
larger domain increases the bound, as expected, but does not alter the proof
construction.

The byte-budget compiler can target this expanded quotient too. For example:

```text
3000-byte request -> 2944 bytes, 22 of 48 prompts, 152 states
6000-byte request -> 5954 bytes, 45 of 48 prompts, 308 states
```

Both binaries pass the structural loader and normalize every replayed law.

Horizon scaling is also parameterized. For the original 32 prompts at
`h=4`, the separate certificate has 480 source states, 460 quotient states,
an 8,816-byte binary, local TV `0.0429541466392004`, conservative bound
`0.1718165865568016`, and measured replay TV `0.003004759447945494`.

The theorem is about the declared finite prompt domain and the exact checkpoint
hashes in `outputs/flan_prsl_manifest.json`. It is not a claim of equivalence
for arbitrary natural-language inputs, arbitrary horizons, or unrestricted
floating-point/quantized arithmetic. Extending the theorem to those cases
requires a separately proved state abstraction and arithmetic error budget.
