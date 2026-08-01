# Two-level interpreter description of FLAN-T5

The useful architecture is a composition of two interpreters rather than a claim that the neural computation is intrinsically Bayesian.

```text
token input
    |
    v
Level 2: Bayesian / probabilistic stack interpreter
    |
    | invokes named neural-algebra operations
    v
Level 1: matrix + nonlinear arithmetic interpreter
    |
    v
finite-precision FLAN-T5 state and output distribution
```

## Level 1: neural algebra DSL

The first DSL describes the concrete finite-precision computation:

```text
LOAD_MATRIX W[layer, row, col]
LOAD_VECTOR b[layer, row]
MATMUL_ACCUM W, activation
ADD_BIAS b
ACTIVATE RELU | TANH | SOFTMAX | ROUND(q)
READ_KV_CACHE layer, head, position
WRITE_KV_CACHE layer, head, position
ATTEND query, key, value, mask
EMIT_LOGITS
```

Its semantics is a deterministic state transition over encoder activations, decoder activations, and KV-cache registers:

```text
h' = Round_q(Nonlinear(W h + b))
s' = NeuralStep(s, token)
```

For a finite-precision checkpoint, this level is a deterministic arithmetic transducer. It can be simulated by a deterministic Turing machine in polynomial time in the encoded weights, precision, and input size.

## Level 2: Bayesian stack DSL

The second DSL describes an abstraction of Level 1:

```text
READ_PROMPT prompt_id
READ_STACK decoder_prefix
ABSTRACT concrete_state -> symbolic_state
EMIT_DISTRIBUTION token_weights
SAMPLE token using random_bits
PUSH token onto decoder_prefix
BRANCH token | OTHER
HALT when fuel = 0
```

Its state is a distribution over concrete Level-1 states:

```text
rho : AbstractState -> Measure(ConcreteState)
```

The induced transition kernel is:

\[
K(a' \mid a,u)
=
\int \mathbf{1}\{\alpha(T(s,u))=a'\}\,\rho_a(ds).
\]

The induced emission law is:

\[
E(y\mid a)
=
\int P_{\mathrm{neural}}(y\mid s)\,\rho_a(ds).
\]

This is why the Bayesian layer is the right representation after information-losing abstraction: a single symbolic state can stand for many distinct neural states.

## Exactness conditions

The two-level program is behaviorally equivalent on a declared domain if:

1. Level 1 exactly implements the finite-precision neural step.
2. Every concrete state in the domain has an abstract-state assignment.
3. The abstract emission law is within `delta` TV of the concrete law.
4. Every concrete successor maps to a permitted abstract successor.
5. The stack fuel bounds the continuation horizon.

Then the finite-horizon output-law error is at most:

\[
\min(1,h\delta),
\]

by induction/coupling over the remaining stack fuel.

## Relation to the current implementation

The current FLAN artifacts instantiate Level 2 directly from Level-1 oracle evaluations. `flan_stack_compile.py` records next-token emissions and prefix-stack transitions; `approximate_domain_quotient.py` merges states; `flan_stack_approx.prsl` is the readable Level-2 program; and `PRSLProof.idr` proves fuel termination and additive error composition.

The missing next stage is to replace the Level-1 oracle calls with explicit matrix/KV opcodes for selected FLAN layers. That would produce a true two-level compiler rather than a Bayesian program whose emissions are queried from FLAN during compilation.
