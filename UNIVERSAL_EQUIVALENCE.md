# Universal all-token-sequence target

The unrestricted target is not a finite quotient of a fixed prompt corpus. It
is a probabilistic register transducer whose state contains:

- arbitrary finite encoder-token input;
- encoder memory;
- the complete decoder token stack;
- all eight layers of decoder K/V cache;
- the numerical mode and model parameters.

At each state the transition function appends an input or sampled decoder token
and updates the registers. The observation is the full next-token probability
function over the 32,128-token vocabulary. This state space grows with sequence
length and should not be forced into a finite graph.

## Universal theorem

`WholeModelEquivalence.lean` defines a source token transducer, a target register
transducer, and `RegisterCompilerCertificate`. The certificate has exactly two
universal obligations for every source state `h` and token `a`:

```text
target.observe (encode h) a = source.observe h a
target.step (encode h) a = encode (source.step h a)
```

Lean proves from these equations:

1. register and source states commute after every finite prefix;
2. next-token weights agree after every prefix;
3. products of conditional token weights agree for every finite sequence;
4. the result is polymorphic in the initial prompt, so it is not restricted to
   an enumerated prompt set.

This is the correct all-sequence equivalence theorem. The remaining FLAN-specific
obligation is to discharge both commuting equations for every one of the 129
lowered opcodes under a completely fixed floating-point execution semantics.
Tensor-reference coverage and multi-input numerical replay are evidence toward
that obligation, not a universal machine-checked proof of it.

`ProgramComposition.lean` proves that opcode-local preservation obligations
compose across an arbitrary finite register program and across any finite
number of autoregressive invocations. The generated ledger
`outputs/flan_universal_opcode_obligations.json` binds one obligation to each of
the 129 lowered opcode occurrences. `CacheSemantics.lean` proves token and K/V
append order plus synchronized cache-length growth for arbitrary finite
continuations. `RMSNormLowering.lean` proves one dimension- and weight-parametric
ordered-primitive lowering theorem and instantiates it across all 42 RMSNorm
occurrences. The exact Transformers T5 source file hash is recorded as the
backend semantics boundary. The current audit therefore has 45 definitionally
lowered schemas. `GatedMLPLowering.lean` similarly proves the ordered pair of
input projections, GELU gate, Hadamard product, output projection, and residual
compatibility for arbitrary dimensions and weights, covering all 16 MLP
occurrences. `AttentionLowering.lean` covers all 24 sequence, cached-self, and
cross-attention occurrences with one shape-, mask-, bias-, and cache-parametric
ordered-primitive theorem. `SoftmaxLowering.lean` covers the final vocabulary
softmax and every token-weight projection. The audit now has 84 machine-checked
complex composition occurrences and no pending opcode schemas. This is a
statement at the graph-composition layer, not a claim that the pinned numerical
backend has been refined.

`emit_flan_program_lean.py` now generates `GeneratedFlanProgram.lean` directly
from the full-graph JSON. It embeds the graph and checkpoint hashes, reproduces
the exact 129-tag order, proves the list length by reduction, instantiates every
tag as a paired opcode, and applies the composition theorem to the concrete
program. An independent parser checks exact JSON/Lean order before Lean checks
the generated theorem.

Universal completion remains false for one narrower reason: equality between
the declared ordered primitives and the independent floating-point backend is
still a hashed trust boundary rather than a machine proof.

The portable kernel ledger makes this layering explicit. Its 343 open backend
occurrences consist of 260 reduction/matrix calls and 83 transcendental calls.
`OrderedKernelLowering.lean` now proves that the portable reduction and each
matrix entry are derived programs over ordered scalar addition and
multiplication, reducing their portable trusted base to those scalar
operations. The remaining reduction obligation is specifically correspondence
of the pinned ATen execution order, including vectorization and FMA behavior.

## Certified finite-horizon approximation route

`AffineProgramComposition.lean` replaces exact opcode equality with the sound
local numerical-refinement contract `e' <= gain * e + bias`. Lean composes
these affine transfers in exact execution order. The older additive theorem is
retained as the special nonexpansive (`gain = 1`) case, but is not assumed for
ordinary neural sup norms. `GeneratedFlanProgram.lean` is generated from the
full graph and instantiates the affine theorem over the exact 129-opcode
occurrence list, not merely the twelve opcode kinds.

`ApproximateWholeModel.lean` then lifts a certified one-token distribution
bound through autoregressive execution. Its theorem
`all_prompts_bounded_horizon` quantifies over every prompt and every
continuation whose length is at most `horizon`, and proves

```text
trace_distance <= horizon * localBudget.
```

The analytic chain rule is explicit in the `SequentialDistance.trace_cons`
field, rather than hidden in a test. For total variation this field is the
standard sequential-kernel/coupling composition obligation. Thus the Lean
result is universal once its premises are supplied, while remaining honest
about which premises are still open.

`outputs/flan_backend_error_obligations.json` contains one hash-protected open
numerical-refinement obligation for every generated opcode occurrence. The
current local bounds are deliberately `null`: source correspondence and replay
measure discrepancies but do not certify a supremum rounding error.

The quantification domain is no longer arbitrary unbounded tensors.
`build_reachable_state_bounds.py` computes a checkpoint-specific sup-norm
envelope for all backend-valid finite encoder sequences and all finite decoder
continuations. It uses the finite embedding table, the dimension-only RMSNorm
bound, matrix infinity norms, the convex-combination property of attention,
the GELU magnitude bound, and residual triangle inequalities. Attention's
bound is independent of prompt/cache length. A separate C++23 verifier scans
the raw safetensors bytes, recomputes all 188 referenced tensor norms with
long-double accumulation, and independently repeats the eight encoder and
eight decoder recurrences. The conservative resulting bounds are approximately
`5.54` for normalized encoder memory, `182` for normalized decoder readout, and
`154310` for every logit. The much larger pre-normalization residual envelope
is retained in the certificate rather than hidden.

Every backend error record hashes this reachable-state certificate. Filling
the remaining local rounding bounds over this finite envelope and proving
the scaled pseudometric-to-TV readout contract is the remaining path to a
numerical all-prompt, horizon-parametric FLAN theorem.

The numerical ledger now closes five arithmetic-free transfers with zero bias
and unit gain: the two token-input operations, both embedding lookups, and the
sample/push/cache structural update. `EmbeddingLowering.lean` proves lookup
equality for every token and every finite token list. The other 124 occurrences
perform floating-point arithmetic and remain open rather than inheriting an
unsound zero-error label.

It also closes all 40 residual `ADD` occurrences against an exact-real target
register operation under an explicit IEEE-754 binary32 round-to-nearest-even
source contract. At error scale `2^40` units per real number, each occurrence
carries the conservative affine law

```text
e_out <= 2 e_in + ceil(2^40 * (2^-24 (B_left+B_right) + 2^-150)).
```

The `B` values come from the sequence-universal reachable-state certificate.
`build_ieee_add_transfers.py` emits all occurrence-specific transfers and a
C++23 verifier independently checks the scale, gain, rounded bias, and exact
40/40 coverage. These bounds are intentionally conservative; in particular,
large pre-normalization decoder envelopes produce large residual-add biases.

The final `lm_head` projection also has an occurrence-specific transfer. Its
32,128 rows each have dot length 512. Under the explicit contract that the
backend uses at most 512 rounded products and 512 rounded additions per output,
the standard `gamma_(2n)` dot-product analysis gives gain 848 and a dyadic
rounding bias. Python and C++23 independently check the checkpoint norm,
`gamma_1024`, scale, rounded gain, and bias. The original box/L1 calculation
gave about 9.42 real logit units with zero incoming discrepancy. The convex
outer RMS-ellipsoid/Cauchy-Schwarz certificate lowers this to about 0.524. The
exact diagonal RMS ellipsoid uses the sharp support function
`sqrt(512) * ||lm_row .* rms_weight||_2`, lowering the rounding term again to
about 0.0171. Its universal logit envelope is 280.12 versus 8,580.87 for the
outer Euclidean ball and 154,309.49 for the original box—a 550.9-fold geometric
improvement. The conservative incoming sup-error gain remains unchanged, and
upstream coarse gains still make the complete portable TV result saturate.

`outputs/flan_convex_geometry_certificate.json` records the checkpoint-specific
ellipsoid, maximum lm-head row L2 norm, convex attention hull, and log-partition
view of softmax. `ConvexTransport.lean` machine-checks that an arbitrary finite
scaled convex mixture remains in the value hull and exposes a potential/KL-to-TV
contract. `outputs/CONVEX_MODEL_TRANSPORT.md` describes the resulting route.

`outputs/flan_convex_reachable_bounds.json` propagates those weighted support
functions through all encoder and decoder blocks. Attention uses coordinatewise
V supports plus convex mixing; gated MLPs use paired weighted supports before
the Hadamard product; residuals use triangle inequalities. This reduces the
old decoder hidden envelope by about 15,822-fold and the encoder envelope by
about 37,118-fold. Transfer builders now consume occurrence-indexed bounds,
fixing the previous extra looseness caused by repeated register names such as
`enc_h` and `dec_h`. Maximum MLP bias falls about 2,300-fold and maximum
attention gain about 5,200-fold. The composed portable TV result nevertheless
remains saturated.

All 42 RMSNorm occurrences now carry the analogous conditional transfer. The
gain uses the global bound

```text
Lip_inf(w * x / sqrt(mean(x^2)+eps))
  <= ||w||_inf * (1 + sqrt(d_model)) / sqrt(eps),
```

and the bias explicitly propagates binary32 error through square/mean,
epsilon addition, reciprocal square root, normalization multiply, and weight
multiply. The contract requires RNE arithmetic, gradual underflow, no overflow,
and a correctly-rounded `rsqrt`; it is not silently asserted for an arbitrary
vendor approximation. The C++23 verifier independently recomputes every gain
and bias for exact 42/42 graph coverage. Because the global pre-normalization
envelope is extremely loose, some resulting biases are enormous. They prove
finiteness and composability but do not yet give a useful TV approximation.

All 16 gated-MLP occurrences are now covered using clipped register semantics.
The exact source states already lie inside the checkpoint-derived envelopes;
clipping therefore leaves source behavior unchanged. Projection onto each
sup-norm box is nonexpansive, so clipping the approximate target after an
operation cannot increase its error. This resolves the otherwise fatal fact
that the quadratic gated product has no global affine Lipschitz bound on an
unbounded approximate state space.

On a clipped input box `||x||_inf <= B`, with checkpoint row norms `A`, `C`,
and `O`, and conservative GELU Lipschitz constant 2, the generated transfer
uses gain `ceil(3 O A C B)`. Its bias propagates the two 512-term projections,
tanh-GELU, Hadamard product, and 1024-term output projection. The source
contract explicitly assumes RNE reductions and correctly-rounded tanh-GELU.
All 16 occurrence-specific records and their C++23 recomputation pass. These
are finite but highly conservative bounds, not a claim of useful final TV yet.

All 24 attention occurrences now have clipped affine transfers with no sequence
length cap. Entrywise floating-point softmax error would accumulate with cache
length and is therefore the wrong invariant. Instead, the contract treats both
source and target attention rows as probability laws: their L1 distance is at
most 2, and each weighted value is a convex combination. This bounds numerical
Q/K/score/softmax disagreement independently of the number of keys. A tighter
gain additionally uses the exact-softmax `l_inf -> l1` Lipschitz bound and
checkpoint Q/K/V/O row norms on clipped query and memory boxes. V and O
projection rounding are charged explicitly. The generated set covers eight
encoder self-attentions, eight cached decoder self-attentions, and eight cross
attentions, and C++23 verifies exact 24/24 coverage with `sequence_cap=none`.
The simplex-diameter bias is deliberately coarse but genuinely universal.

The final vocabulary softmax now also has a transfer, so every one of the 129
occurrences is instantiated. Its source contract says the backend softmax
denotes a probability law; against exact-real target softmax, TV is always at
most the simplex diameter 1. The emitted affine record therefore has unit gain
and bias equal to one full TV unit (`2^40` scaled units). This is universally
valid but deliberately labeled `TRIVIAL_DIAMETER_BOUND`.

Consequently “occurrence coverage complete” is now true while “nontrivial
universal approximation” remains false. The current composed result proves
only `TV <= 1`; moreover several primitive contracts (correctly-rounded rsqrt,
GELU, reduction behavior, probability-law softmax) remain assumptions about
the pinned backend rather than verified implementations. Completion requires
a tighter shared or formally refined backend that yields a TV bound strictly
below 1 for a useful horizon.

## Exact shared-backend route

There is now a second, non-vacuous route alongside the portable-error attempt.
`outputs/flan_shared_backend_exact_program.json` binds every named opcode
occurrence and its argument record to a shared callable ABI. Source FLAN and
the register interpreter must invoke the same callable object on bit-identical
tensors. Numerical error is then definitionally zero, while the outer program
remains the explicit 129-op register graph with a full probabilistic readout.

`SharedBackendExact.lean` constructs the source and target token transducers
from the same generated opcode kernels and proves trace-weight equality for
every initial prompt and every finite continuation. This theorem has no prompt
corpus or sequence-length cap. The manifest is hash-bound to the graph,
checkpoint, and backend contract and records 26,712 graph bytes plus the
307,867,048-byte checkpoint.

This result is exact relative to the shared ABI, not a portable independent
decompilation. Its trusted boundary is callable identity, argument identity,
the pinned backend/checkpoint, and correctness of the structural lowering from
the Transformers implementation to the 129-op graph. It is useful because it
isolates the only way to obtain exact floating behavior without formally
reimplementing PyTorch: share the numerical kernels and decompile the control,
state, cache, tensor references, and probability flow around them.

The structural lowering boundary is now narrower. A source-schedule checker
parses and hashes the installed `T5ForConditionalGeneration.forward`,
`T5Stack.forward`, `T5Block.forward`, self-attention wrapper, cross-attention
wrapper, and feed-forward wrapper. It verifies model encoder/decoder/lm-head
order; embedding, eight-block loop, and final norm; block self/cross/FF order;
pre-norm and residual placement; and exact equality with the generated encoder
and decoder opcode templates. The resulting six method hashes and certificate
hash are embedded in the shared-backend manifest. Softmax and sample/cache are
identified explicitly as the probabilistic autoregressive wrapper after source
logits. This is a strong drift-detecting AST/dataflow certificate, not a formal
semantics for arbitrary Python or PyTorch dispatch.

For readout, corresponding logit vectors within sup error `delta` have softmax
likelihood ratio between `exp(-2 delta)` and `exp(2 delta)`, hence total
variation at most `tanh(delta)`, independent of vocabulary size. The scaled
contract and capped horizon theorem are represented in
`SoftmaxTVTransport.lean` and `ApproximateWholeModel.lean`; the elementary
analytic derivation is recorded in `outputs/SOFTMAX_TV_TRANSPORT.md`.

`outputs/flan_backend_contract.json` now makes that boundary reproducible. It
pins CPU float32 execution, deterministic algorithms, highest matmul precision,
Torch and Transformers versions, ordered primitive definitions, and hashes of
the T5 implementation, activation implementation, checkpoint, tokenizer,
register interpreter, and full graph. This is independently rechecked by
`verify_backend_contract.py`; it deliberately reports the universal IEEE-754
proof as false.

`verify_backend_source_correspondence.py` additionally extracts the installed
`T5LayerNorm`, `T5DenseGatedActDense`, and `T5Attention` implementations and
checks their primitive order against the sequence-parametric register
interpreter, including K/V update versus append. It hashes the extracted method
bodies. This detects semantic drift in the pinned backend, but remains a
syntactic correspondence check rather than a proof about IEEE-754 operations.

## Role of finite quotient towers

The finite prompt towers remain useful as certified graph patches and regression
instances. Following the revised paper, every level now carries a direct trace
contract, not merely within-block coherence. Exhaustive trace-law comparison on
the 32-prompt, horizon-3 domain gives:

| Level | Status relative to uint16 oracle | States | Direct trace TV | Bound relative to projected neural law |
|---:|---|---:|---:|---:|
| 0 | certified approximate | 216 | 0.00532213 | 0.00550524 |
| 1 | certified approximate | 221 | 0.00407386 | 0.00425697 |
| 2 | certified approximate | 222 | 0.00407386 | 0.00425697 |
| 3 | certified exact | 224 | 0 | 0.000183108 |

The nonzero final neural bound is the accumulated analytic uint16 probability
quantization bound. “Exact” at level 3 therefore means exact relative to the
serialized uint16 oracle, not identical to unquantized neural probabilities.

## Honest current status

- Whole architecture translated: yes, as a parametric register program.
- Sequence-parametric executable: yes. `verify_kv_cache_register.py` accepts
  arbitrary tokenized prompts and either decoder text or explicit token IDs;
  tests currently cover 6, 19, and 31 decoder positions with maximum logit
  discrepancies below `6.9e-05` against full-prefix execution.
- Whole input domain routed: yes, through the register residual.
- Universal theorem relating one-step simulation to all-sequence probability:
  machine-checked in Lean.
- FLAN-specific universal one-step simulation proof: incomplete.
- Finite-domain quotient patches: directly trace-certified.
- Finite explicit graph equivalent on all unbounded token sequences: neither
  established nor expected to be small.
