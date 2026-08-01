# Arithmetic bound ledger

The probabilistic quotient certificate and tensor arithmetic certificate are
separate. A replay is evidence, not an arithmetic proof. For every opcode the
compiler must certify

```text
|| opcode_exact(s) - opcode_fixed(s') || <= epsilon_opcode
```

for all reachable states in the declared input box.

## Primitive obligations

For rounding to grid `q = 2^-b`:

```text
|round_q(x) - x| <= q/2
```

For `y = W x`, using the infinity norm:

```text
epsilon_y <= ||W||inf epsilon_x
             + ||dW||inf (||x||inf + epsilon_x) + q/2
```

and `||dW||inf <= columns(W) q/2` for independently rounded weights.

For RMSNorm `R(x)=w*x/sqrt(mean(x^2)+eta)`, a conservative bound on a box is

```text
Lip(R) <= 2 ||w||inf / sqrt(eta)
epsilon_R <= Lip(R) epsilon_x + ||x||inf q/(2 sqrt(eta)) + q/2
```

For residual addition:

```text
epsilon_add <= epsilon_left + epsilon_right + q/2
```

For the tanh-approximate GELU gate, the checker must establish a derivative
bound `L_g` on the declared input interval and then use the product bound

```text
epsilon_gate <= (L_g ||v||inf) epsilon_u
               + sup|GELU(u)| epsilon_v + q/2
```

For attention, bound score error with the linear-map rule, use

```text
||softmax(s)-softmax(s')||inf <= 1/2 ||s-s'||inf
```

and bound the weighted value sum by

```text
epsilon_sum <= ||V||inf epsilon_weights
               + ||weights||inf epsilon_V + q/2
```

The output projection is another linear-map entry. Cache rows are ordinary
matrix operands and must be included in the range proof.

## Composition

If opcode `i` has certified Lipschitz factor `L_i` and local rounding budget
`epsilon_i`, the trace recurrence is

```text
e_0 = input_error
e_(i+1) = L_i * e_i + epsilon_i
```

The final arithmetic error is `e_n`. The complete output-law bound is then

```text
TV_total <= TV_quotient + TV_arithmetic
```

where `TV_arithmetic` must be derived from the logit-to-distribution map; it
must not be identified with a hidden-vector norm. `BudgetTrace` and
`TwoSourceBound` in `PRSLProof.idr` encode this composition interface.

The current results from `quantize_cached_step.py` are deliberately marked
`MEASURED`, not `CERTIFIED`, until these range and local-inequality checks are
implemented.

`interval_bound_cached_step.py` is the first attempted implementation. It
correctly propagates outward boxes through the opcode structure, but the
independent-coordinate boxes become unbounded (`Infinity`) after attention and
RMSNorm. This is a failed *tightness* strategy, not evidence of model failure:
the correlations between repeated values and matrix products are lost. A
useful certificate therefore needs centered/affine forms, interval subdivision,
or a local Jacobian bound around the reachable trace. Its output is not marked
`CERTIFIED`.
