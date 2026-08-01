# PRSL-STACK-1

`PRSL-STACK-1` is a domain-specific probabilistic stack language for a bounded FLAN-T5 approximation.

Its state is:

```text
(prompt_id : Fin |D|, depth : Fin h, decoder_prefix : Vect depth Token)
```

The instructions are conceptually:

```text
READ_PROMPT prompt_id
READ_STACK decoder_prefix
EMIT_TOPK quantized_distribution
PUSH token
```

For every compiled state, the emitted distribution is FLAN-T5's first-next-token distribution conditioned on the prompt and exact decoder prefix. The compiler expands every top-2 output branch for three steps, giving 56 states over eight prompts. Probabilities are fixed-point integers with denominator 65535.

The checker recomputes FLAN-T5 from the local 230 MB checkpoint and evaluates total variation distance independently. `OTHER` is an explicit normalized outcome, so every emitted measure sums to one. Thus the reported error is a verified finite-domain bound:

```text
for every compiled state s:
  TV(PRSL(s), FLAN(s)) <= 0.3650418194413511
```

The 16 KiB frontier actually uses 2,850 bytes and top-8 emissions, with maximum TV error 0.36504 and mean TV error 0.15888. The program is not claimed equivalent on arbitrary prompts, arbitrary decoder prefixes, or beyond horizon 3. Extending the domain and horizon is mechanical but increases the number of states; a generalization bound would require a separately declared prompt distribution or verified abstraction invariant.

`PRSLProof.idr` supplies the first formal layer for that invariant. Its `MetricWitness` interface abstracts the observable distance, while `Chain` composes local approximation witnesses additively. The file typechecks in total Idris2 mode. The remaining model-specific obligation is to instantiate `Near` with a certified TV-distance relation for FLAN distributions and prove successor preservation for a nontrivial state quotient.
