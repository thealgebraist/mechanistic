# Four typed toy decompilation tests

`ToyDecompile.idr` is a first exact test bed for the proposed decompiler. It uses Idris2 ADTs for every alphabet, state space, output space, model, and corpus case. The generic model is:

```text
Model q x y = (q -> x -> q, q -> y)
```

The run function exposes the output trace for a finite continuation. The four cases are:

| model | concrete states | minimized states | purpose |
|---|---:|---:|---|
| constant | 2 | 1 | merge states with identical behavior |
| parity | 2 | 2 | retain a one-bit predictive memory |
| copy | 3 | 3 | retain the latest-symbol memory |
| hidden-redundant | 4 | 3 | merge one redundant pair while retaining two distinct modes |

The supplied behavior equalities are checked by Idris2's definitional equality (`Refl`). Thus those particular finite traces are not merely unit-test assertions.

Run:

```sh
idris2 ToyDecompile.idr -o toy_decompile
./build/exec/toy_decompile
```

The current `minimized states` numbers are explicit certificates for these four hand-enumerated finite models, not yet a generic partition-refinement proof. The next step is to represent a typed finite state set using `Fin n`, enumerate all continuations up to an indexed horizon `h`, compute typed behavioral signatures, and prove that the quotient preserves `run` for every continuation of length at most `h`.

These examples are deterministic transducers, which are degenerate Bayesian graphs. A fifth test should introduce rational emission distributions and probabilistic successor distributions; exact equality can then be represented by normalized integer counts rather than floating-point values.
