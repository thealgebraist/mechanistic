# NEURAL-ALGEBRA-1 and PRSL composition

## Level 1 state

```text
RegisterState =
  vectors : Name -> Vect q Rational
  matrices : Name -> Matrix Rational
  heads : Name -> Vect n (Vect d Rational)
  cache : KVCache
  fuel : Nat
```

The arithmetic DSL has deterministic instructions:

```text
LOAD_VECTOR name values
MATMUL matrix input output
MATMUL_HEADS qkv weights heads head_width
SCALED_DOT_SELF q k scale score
SOFTMAX score weights
WEIGHTED_SUM weights values heads
CONCAT heads joined
GELU_GATE u v gated
DOT_ROW token_id weights logit
HALT
```

With finite rational or fixed-point values, each opcode denotes a total deterministic function on `RegisterState`. The `fuel` index gives a terminating semantics for bounded programs.

## Level 2 state

```text
ProbState =
  prompt_id : PromptId
  decoder_prefix : Stack Token
  belief : Measure RegisterState
  fuel : Nat
```

Level 2 invokes Level 1 to obtain logits, then applies:

```text
EMIT_DISTRIBUTION
SAMPLE
PUSH
BRANCH
OTHER
```

The emission distribution is the pushforward/mixture:

\[
E(y\mid a)=\int E_1(y\mid r)\,\rho_a(dr).
\]

The transition kernel is:

\[
K(a'\mid a,t)=
\frac{\int E_1(t\mid r)\mathbf 1[\alpha(T_1(r,t))=a']\,\rho_a(dr)}
{\int E_1(t\mid r)\,\rho_a(dr)}.
\]

This conditional form is essential. A state merge based only on unconditional emission similarity is unsound because the emitted token can correlate with different successor states.

## Composition certificate

For a finite prompt domain `D`, branch policy `B`, and horizon `h`, a certificate must contain:

```text
Level-1 checkpoint hash
opcode program hash
state-to-belief assignment
emission TV bound delta_E
conditional successor TV bound delta_K
byte length <= n
fuel termination proof
```

If the concrete and abstract transitions satisfy an inductive coupling invariant with per-step error `delta`, then:

\[
\operatorname{TV}(P_{\mathrm{concrete}}^{(h)},
P_{\mathrm{abstract}}^{(h)})
\le \min(1,h\delta).
\]

The current verified PRSL artifacts instantiate the Level-2 part using FLAN oracle evaluations. `flan_attention_level1.json`, `flan_mlp_level1.json`, and `flan_readout_level1.json` instantiate exact Level-1 subprograms for selected FLAN operations. The remaining integration work is to compile an entire encoder/decoder layer into one connected Level-1 trace and then replace oracle emissions with its interpreter output.

## Independent finite-domain check

`verify_domain32_certificate.py` reads only `flan_domain32_program.json.gz`
and `flan_domain32_quotient.json`; it does not load Python Transformers. It
checks probability normalization, complete non-overlapping state membership,
successor compatibility for every merged block, the recorded local TV value,
and the finite-horizon bound. The current certificate passes for 32 prompts,
224 source states, 221 quotient states, horizon 3, local TV
`0.012649729152361318`, and union bound `0.037949187457083956`.

This is a theorem about the supplied finite oracle table and prompt domain,
not yet a theorem about all FLAN-T5 inputs. Full-model validity requires a
connected opcode compilation of the encoder/decoder recurrence and a trusted
error bound for quantized tensor arithmetic.

`verify_domain32_replay.py` additionally replays the concrete and quotient
output laws for all 32 roots through three decoding steps, including the
aggregated `OTHER` event. It currently reports maximum sequence TV
`0.0010116591814860144` and mean sequence TV
`0.00008498501190308378`; both are below the recorded union bound.

## Connected FLAN decoder block

`lower_flant5_decoder_block.py` lowers one actual FLAN-T5 decoder block for a
declared prompt and one decoder position into the following register trace:

```text
x0 -> RMSNORM -> self attention -> ADD -> x1
x1 -> RMSNORM -> cross attention(memory) -> ADD -> x2
x2 -> RMSNORM -> wi_0, wi_1, GELU_GATE, wo -> ADD -> y
```

The serialized program is `outputs/flan_decoder_block_level1.json`. Its
standalone interpreter is `run_decoder_block.py`, which currently verifies
maximum float32 replay error `0.0001220703125` against the checkpoint trace.
The program is 63,593,940 bytes because it contains the explicit matrices and
one 11-by-512 encoder-memory snapshot. This is an exact trace certificate for
that prompt/position, not a general compact replacement for the checkpoint.

`make_parametric_decoder_block.py` removes those two activation constants and
emits `flan_decoder_block_parametric.json`. Its typed runtime inputs are
`x0 : Vect[512] Float32` and `memory : Matrix[11,512] Float32`; the captured
values are kept separately in `flan_decoder_block_fixture.json`. The
parametric interpreter `run_parametric_decoder_block.py` reproduces the same
fixture with maximum error `0.0001220703125`. This is the correct shape for a
general register-language compilation: activations are state, not program
text. It still specializes sequence lengths and decoder position, and does
not yet include a KV-cache transition or a quantization proof.

## Autoregressive KV-cache transition

`lower_flant5_cached_step.py` and `run_cached_step.py` add the missing
autoregressive state transition. For a two-token decoder prefix, the program
takes `x`, a six-head self-attention K/V cache, relative-position bias, and
encoder memory as explicit registers. The resulting artifacts are
`outputs/flan_cached_step.json` and `outputs/flan_cached_step_fixture.json`.
The checkpoint lowering and standalone replay both pass with maximum error
`0.000030517578125` (`CACHED_STEP_REPLAY_OK`).

The cache is therefore part of the machine state rather than hidden inside an
oracle call. In the probabilistic machine, the next-token sampler updates the
stack and appends the newly computed K/V row to this cache before the next
register-program invocation. Remaining work is to quantify fixed-point weight
and activation error and to prove the resulting cache-state abstraction over a
nontrivial prompt domain.

## Standalone C++23 probabilistic interpreter

`emit_cpp_prsl.py` compiles the 221-state certified quotient into
`flan_prsl_cxx23.cpp`. The C++23 interpreter implements fixed-point emissions,
explicit `OTHER` mass, stack transitions, and terminal-depth token emission.
It compiles with `-std=c++23 -Wall -Wextra -pedantic` and reports normalized
laws (`mass=1.000000000000`) for all 32 roots. The corrected independent replay
certificate reports maximum sequence TV `0.0020689571230390917` and mean
sequence TV `0.00019571697807858627`, below the 3-step bound
`0.037949187457084005`.

The executable also supports seeded sampling:

```text
./flan_prsl_cxx23 --sample 0 7
```

which performs categorical draws from the fixed-point measure, pushes sampled
tokens through the stack transition, and emits `OTHER` when the residual event
is drawn. Repeating the same prompt/seed reproduces the same path.

`emit_prsl_binary.py` also emits the fixed-width binary program
`outputs/flan_domain32.prslb`. Its `prsl_binary_cxx23.cpp` loader verifies the
magic, dimensions, roots, states, and two-branch tables before replay. The
current artifact is 4,275 bytes with SHA-256
`14c5aedd649c152d5046691db1f20630b589356ad4ddbb8b8a487a7643d23627`; all 32
binary-replayed laws normalize to one. This is the current concrete answer to
the `max n bytes` form of the objective for the declared 32-prompt, 3-step
domain. It is not a bound for unrestricted FLAN-T5 inputs.

`budget_prsl_binary.py --bytes n` selects the largest prefix of the declared
prompt domain whose reachable quotient graph fits the requested byte budget.
Current certificates are:

```text
n=1000  -> 919 bytes,  7 prompts,  47 states
n=2000  -> 1999 bytes, 15 prompts, 103 states
n=3000  -> 2944 bytes, 22 prompts, 152 states
n=4275  -> 4275 bytes, 32 prompts, 221 states
```

Each is replayed by `prsl_binary_cxx23` with normalized probability mass. The
compiler guarantees the byte bound for the selected finite domain; it does not
claim that omitted prompts are represented.

The binary loader now rejects malformed artifacts before replay: it checks
file consumption, root depth, fixed-point mass sums, token-labelled successor
edges, successor depth increments, terminal padding, and normalized replay
mass. The 919-, 1,999-, 2,944-, and 4,275-byte artifacts all report
`structural=OK`.

`verify_prsl_all.sh` is the single regression gate. It rebuilds the quotient,
runs the dependency-free certificate and finite-horizon replay checks, emits
the full and budgeted binaries, compiles the C++23 loader with warnings enabled,
validates every budget artifact, and typechecks `PRSLProof.idr`. The current
run ends with `PRSL_FULL_REGRESSION_OK`.

`make_prsl_manifest.py` emits `outputs/flan_prsl_manifest.json`, binding the
certificate to SHA-256 hashes of the FLAN config, 307,867,048-byte safetensors
checkpoint, tokenizer model, source oracle table, quotient, and binary PRSL
program. The manifest records 32 prompts, 224 source states, 221 quotient
states, horizon 3, local TV `0.012649729152361318`, and horizon bound
`0.037949187457084005`.

The general finite-domain construction and its byte/coupling proof are stated
in [FINITE_DOMAIN_PRSL_THEOREM.md]. The regression gate checks that this
theorem specification is present alongside the executable certificate.

## Fixed-point arithmetic experiment

`quantize_cached_step.py` rounds every input, weight, and intermediate register
to a grid of size `2^-b`. On the cached decoder-step fixture, the maximum
output errors for `b = 6, 8, 10, 12, 14, 16` fractional bits are respectively
`2.99945, 2.36664, 0.44489, 0.09515, 0.03439, 0.00496`. These numbers are
measurements, not a proof: a final approximation theorem must add a
sound interval or Lipschitz error ledger for RMSNorm, attention softmax, GELU,
and matrix products. They must also be kept separate from the probabilistic
total-variation bound for state merging.

`PRSLProof.idr` now contains a typed `BudgetTrace` assigning one certified
arithmetic budget to every opcode, plus a `TwoSourceBound` separating
probabilistic and arithmetic budgets. `idris2 --check PRSLProof.idr` succeeds.
This proves the composition interface and termination/indexing discipline; it
deliberately does not manufacture the missing numerical interval bounds.
