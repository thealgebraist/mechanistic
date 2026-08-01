# Fixed-point quotient tower for FLAN-T5

This implementation instantiates the coarse-to-fine construction from
*Certified Coarse-to-Fine Extraction of Neural Token Models into Explicit
Graphs* on the serialized 32-prompt, horizon-3 FLAN-T5 oracle domain.

## Implemented objects

The inner fixed point is a bottom-up transition-signature refinement. Because
the finite trace graph is acyclic by decoder depth, one reverse-depth pass is
the fixed point for a chosen tolerance. The outer stages use tolerances

`0.10 -> 0.05 -> 0.02 -> 0.00`.

They produce quotient sizes

`216 -> 221 -> 222 -> 224`.

Every fine block is contained in exactly one coarse block. The emitted maps
`pi_k : Q_(k+1) -> Q_k` satisfy both root and transition commutation:

`pi_k(delta_(k+1)(q,a)) = delta_k(pi_k(q),a)`.

The verified horizon bounds decrease monotonically:

`0.270497 -> 0.068574 -> 0.012497 -> 0`.

The final level is the exact 224-state serialized oracle graph. This is exact
only for the supplied finite prompt/horizon contract; behavior outside that
contract is explicitly classified as unexplored rather than silently claimed.

## Audit package

`outputs/flan_domain32_refinement_tower.json` contains:

- the source-table and FLAN checkpoint hashes;
- each quotient partition and its representative emissions;
- three explicit forgetful maps;
- counterexample witnesses for every proper split;
- per-edge token predicates, destinations, error contracts, residual routing,
  verifier assumptions, and certificate hashes;
- residual classification and coverage scope.

`verify_refinement_tower.py` independently checks partition coverage,
certificate hashes, every block's source-derived TV coherence, recomputed
horizon bounds, exact representative emissions and successor edges, block
containment, root commutation, edge commutation, monotone refinement, singleton
exactness at the final level, and split-witness membership. A passing run emits
`REFINEMENT_TOWER_OK`.

`PRSLProof.idr` mirrors the same structure in dependent types. `QEdge n` uses
`Fin n` endpoints, `RefinementMap c f` types every fine state with one coarse
parent, and `TowerStep c f` requires proofs that state cardinality increases
and epsilon decreases. `forgetEdge` transports a fine edge to its canonical
coarse edge, while `CommutingEdge` carries the equality proof. Idris2 checks
the source successfully, so out-of-range endpoints and directionally invalid
tower maps cannot inhabit these types.

## Proof boundary

This is a certified inverse system over a finite serialized neural oracle. It
does not prove that the unrestricted FLAN-T5 state space has a finite quotient,
nor that sampled prompts cover all natural-language behavior. Extending the
contract requires a sound reachability enclosure or a larger explicitly finite
domain and rerunning the same tower/verifier pair.
