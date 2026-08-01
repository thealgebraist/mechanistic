# Portable probabilistic binary32 microcode

The complete FLAN-T5 register graph now has two interpreter levels:

1. 129 human-readable model opcodes such as `RMSNORM`, `GATED_MLP`,
   `SELF_ATTENTION_SEQUENCE`, `SOFTMAX`, and `SAMPLE_PUSH_UPDATE_CACHE`;
2. 947 portable micro-operations from 23 primitive kinds.

Floating micro-operations explicitly round every scalar result using binary32
round-to-nearest, ties-to-even. Reductions use a specified index order. Tensor
dimensions remain symbolic finite loops, so the program is linear in graph
size rather than an unrolling of token states.

Probability normalization expands to maximum reduction, subtraction,
binary32 exponential, sum reduction, and division. Sampling is inverse CDF
with an explicit random bitstream. For fixed random bits the interpreter is
deterministic; quantifying over uniform bitstreams induces the categorical
transition kernel and hence token-trace probabilities.

Lean proves that executing a macro program equals executing the concatenation
of its microcode expansions. This is definitional semantic preservation: macro
semantics are specified by their expansion, so it introduces no numerical
error.

## Remaining refinement obligation

The microcode is a complete portable reference semantics, but it is not yet a
proof that the pinned PyTorch/Apple-arm64 kernels implement those exact scalar
rounding and reduction rules. In particular, matrix kernels may reassociate or
vectorize reductions, and system `exp`/`tanh` implementations are not generally
specified as correctly rounded binary32 functions.

There are therefore currently two exact statements:

- the readable 129-opcode program is exactly equivalent to its 947-opcode
  portable reference expansion;
- the shared-ABI 129-opcode program is exactly equivalent to the pinned source
  for all finite token traces.

Bridging these statements requires either a bit-exact reference implementation
used by both sides or a kernel-by-kernel refinement proof. Numeric replay alone
does not prove that bridge.
