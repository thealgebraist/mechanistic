# Whole-model translation strategy

The complete FLAN-T5 checkpoint is translated into a typed register program,
while explicit probabilistic quotient graphs replace certified subdomains.
This is the dual fixed-point architecture from the supplied paper:

`whole behavior = certified graph patches + register-program residual`.

The register residual is not an opaque call to an unspecified neural service.
It is `outputs/flan_full_graph.json`, a 129-opcode program with explicit encoder
memory, decoder hidden registers, per-layer KV caches, relative bias, matrix
operations, nonlinearities, readout, and probabilistic token emission. All 189
weight references are bound to the checkpoint hash; 188 unique referenced
tensors are checked against the 190-tensor safetensors inventory.

The current graph patch is the nested finite-domain tower with 216, 221, 222,
and 224 states. The router uses a graph only when the exact prompt identity and
decoder-horizon guard hold. Every other input runs through the complete
register program. Consequently routing is total over tokenized inputs within
runtime resource limits, while graph exactness is claimed only where proved.

`outputs/flan_whole_model_hybrid.json` binds the checkpoint, tokenizer, full
register graph, quotient tower, guards, error contracts, verifier names, and
explicit unproved claims by SHA-256. `verify_whole_model_hybrid.py` independently
checks these links and emits `WHOLE_MODEL_HYBRID_OK`.

This translates the whole model into the register language now. It does not
yet translate the whole unrestricted behavior into a finite human-readable
graph; that would require the outer refinement process to cover all reachable
states or retain an explicitly classified infinite/register residual.

## Whole-domain byte selection

`compile_tower_budget.py --bytes n` chooses the finest nested quotient that
fits without dropping any of the 32 certified prompts. Measured examples:

| Budget | Actual | Level | States | Horizon TV bound |
|---:|---:|---:|---:|---:|
| 4200 | 4180 | 0 | 216 | 0.270497 |
| 4300 | 4294 | 2 | 222 | 0.012497 |
| 4400 | 4332 | 3 | 224 | 0 |

The binary stores the two explicit branch tokens and folds all remaining token
labels into `OTHER`. This projection cannot increase total variation, by the
data-processing inequality. The C++23 loader verifies structure and total
probability mass for every prompt.
