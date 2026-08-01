# Exact quotient result

`minimize_prsl_stack.py` performs backward behavioral partition refinement on the PRSL-STACK-1 program. A state signature contains:

```text
(quantized emission distribution,
 labelled successor quotient states)
```

For the current domain (8 prompts, top-2 branch expansion, horizon 3), the result is:

```text
source states:   56
quotient states: 56
```

No exact merges are possible because the prompt roots and their descendants have distinct quantized emission/continuation signatures. The emitted quotient is [flan_stack_quotient.json](/Users/anders/Documents/Codex/2026-07-31/dwl/outputs/flan_stack_quotient.json).

This is a useful negative certificate: exact finite-horizon minimization does not automatically produce a small graph for FLAN. The next reduction must be approximate partition refinement. If states are merged only when their emissions differ by at most δ in total variation and their successor blocks satisfy the same inductive condition, a horizon-​h error bound can be propagated by the probabilistic coupling inequality.
