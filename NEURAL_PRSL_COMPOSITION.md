# Neural register program composed with PRSL

The current project has two verified layers:

1. a deterministic neural register program for selected FLAN-T5 operations;
2. a probabilistic PRSL quotient whose emissions were obtained from the FLAN
   oracle table.

The missing composition is specified here.

## Machine state

```text
R = (prompt, decoder_stack, encoder_memory, self_KV_cache, fuel)
```

The neural register program `N` is deterministic and total at a fixed numeric
format:

```text
N : R -> (logits, next_KV_cache)
```

The probabilistic wrapper applies the readout measure

```text
E_R(t) = softmax(logits)(t)
```

then performs:

```text
sample t ~ E_R
if t = OTHER: terminate
else push t; replace cache by next_KV_cache; continue
```

This is a probabilistic pushdown/register machine. The stack is discrete and
the neural registers/cache are continuous or fixed-point tensors.

## Abstraction theorem

Let `alpha(R)` map concrete neural states to abstract PRSL states. A usable
certificate must provide, for every reachable state:

```text
TV(E_R, E_alpha(R)) <= delta_emit
alpha(next_R(t)) = next_alpha(R)(t)  for every retained token t
```

The second condition is the cache-aware successor invariant. If it holds, the
standard coupling induction gives

```text
TV(output_neural[h], output_PRSL[h]) <= min(1, h * delta_emit).
```

If tensor arithmetic introduces a logit error `||d logits||inf <= eta`, a
separate softmax stability lemma can contribute an additional
`delta_arithmetic`; the final bound is

```text
TV(output_FLAN[h], output_PRSL[h])
  <= min(1, h * (delta_emit + delta_arithmetic)).
```

## Current evidence and exact gap

`flan_decoder_block_parametric.json` and `flan_cached_step.json` instantiate
the deterministic `N` layer for one decoder block and explicit KV state, with
float32 replay errors below `1.3e-4` and `3.1e-5`, respectively.

`flan_domain32.prslb` and `flan_domain64.prslb` instantiate the probabilistic
quotient layer and have independently checked finite-domain TV bounds.

What is not yet proved is the bridge between these artifacts: a full
encoder/12-decoder-block register program, a cache-aware abstraction map, and
a sound fixed-point logit error bound. Until those are supplied, the PRSL
certificates are oracle-specialized finite-domain approximations, not claims
of unrestricted neural equivalence.

`outputs/flan_full_graph.json` now supplies the structural full-model graph for
the actual checkpoint: 8 encoder layers, 8 decoder layers, 129 opcodes, and
explicit tensor-name references for all norms, attention projections, gated
MLPs, readout, sampling, and per-layer KV-cache updates. Its checkpoint hash
matches the provenance manifest. This advances the graph specification, but it
is intentionally not labeled a numerical equivalence certificate until an
end-to-end interpreter replay is added.

`verify_full_graph.py` checks the graph against the actual safetensors
inventory. It verifies all 189 weight references (188 unique tensors), the
checkpoint hash, opcode count, and register dataflow; the checkpoint contains
190 tensors. It reports `FULL_GRAPH_TENSOR_REFERENCES_OK`. This is a stronger
structural certificate, but it still does not substitute for end-to-end numeric
replay.

The numeric interpreter is parameterized by prompt and decoder token. The
regression matrix now covers the original question prompt, a German translation
prompt, a summarization prompt, and a distinct decoder token. All four cases
report encoder error `0.0`, decoder-hidden error `0.0`, and maximum logit error
`0.0` against the checkpoint reference.

`run_full_graph_numeric.py` now executes all 8 encoder and 8 decoder layers
from the safetensors tensor references for the declared prompt and one decoder
position. It compares both encoder memory and final decoder hidden state with
the reference implementation; both errors are `0.0`, and final maximum logit
error is `0.0` in float32 (`FULL_GRAPH_NUMERIC_REPLAY`). This is an exact
single-example numeric validation of the full register graph, not yet a
quantitative generalization or fixed-point certificate.

The decoder-token regression case now uses a two-token causal prefix and
exercises sequence-shaped self-attention. It reports encoder/decoder-hidden
error `0.0` and maximum logit error `2.47955322265625e-05`, with the same top
token. This is the first full-graph numeric check beyond a one-position decoder.

The independent cached register interpreter is now also validated. Its cache
uses the native layout `(batch, heads, sequence, head_dim)`, and the decoder
relative-position bias computed by block 0 is reused by every decoder block,
as in T5's position-bias dataflow. `verify_kv_cache_register.py` reports
position errors below `5.4e-05` over a 17-position causal prefix, matching
the framework cache oracle. The layer-level diagnostics remain below `8e-04`
in the extended trace. This establishes the executable KV-cache transition
needed by the probabilistic register DSL for repeated cache appends; broader
arithmetic and abstraction bounds remain separate obligations.
