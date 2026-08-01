# Softmax logit error to total variation

Let `p = softmax(x)` and `q = softmax(y)`, with
`max_i |x_i-y_i| <= delta`. Then

```text
exp(-delta) <= exp(x_i-y_i) <= exp(delta)
exp(-delta) <= Z_y/Z_x <= exp(delta)
exp(-2 delta) <= p_i/q_i <= exp(2 delta).
```

For probability laws whose likelihood ratio lies in `[1/M,M]`, total
variation is at most `(M-1)/(M+1)`. Taking `M = exp(2 delta)` gives

```text
TV(softmax(x), softmax(y)) <= tanh(delta) <= min(1, delta).
```

This bound is independent of vocabulary size. With `U` integer TV units and a
certified real logit bound `delta`, a conservative one-step integer budget is

```text
epsilon = min(U, ceil(U * tanh(delta))).
```

The current FLAN ledger does not yet contain a certified `delta`; numerical
replay values are not substituted for this universal premise.
