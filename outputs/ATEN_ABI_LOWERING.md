# ATen ABI lowering

The exact shared-ABI route is now lowered below Python model methods. A dispatch
trace of the full encoder/decoder path and a one-token cached decoder path binds
the implementation to the actual ATen operator schemas invoked by PyTorch.

The schema set includes embedding, shape/index transformations, masking,
elementary binary32 arithmetic, `mm`, `bmm`, `mean`, `rsqrt`, `tanh`, and the
opaque `_softmax` operator. Each schema string is hashed. The exact PRSL target
contract is to dispatch the identical ATen schema with bit-identical tensor
arguments, device and dtype.

This removes `T5Attention.forward`, `T5LayerNorm.forward`, and related Python
methods from the innermost semantic identity assumption. The source schedule
certificate still establishes how those methods compose the fixed eight-layer
encoder and decoder, while traces cover both the no-cache and populated-cache
branches. ATen schemas are shape-polymorphic, so the binding itself does not
fix prompt or continuation length.

This remains a shared ABI result, not a portable kernel proof. CPU dispatcher
kernels for matrix reductions and transcendental operations are still opaque;
the target is exact because it invokes those same kernels, not because their
implementation has been independently formalized.

The independent complete graph evaluator is now the executable ATen PRSL
target. Its numerical core contains no Python `@` nodes and no calls to
high-level `torch.softmax`, `torch.rsqrt`, `torch.matmul`, or functional GELU.
It invokes explicit `torch.ops.aten` schemas for `mm`, `bmm`, `_softmax`,
`rsqrt`, `tanh`, means, powers, additions and multiplications. The regression
compares this evaluator with the original model across multiple encoder and
decoder lengths.

This evaluator is a candidate implementation, not yet an unconditional proof
witness. All four regression cases now match bit-for-bit, including the
two-token decoder case. The repaired readout applies `lm_head` to the complete
decoder sequence before selecting its last row, preserving the source matrix
shape and therefore its binary32 reduction order. A row-only matrix product had
previously produced a maximum discrepancy of about `2.48e-5`.

The zero-error regression matrix is evidence, not universal quantification over
all valid shapes. The abstract shared-ABI theorem remains exact only under its
explicit bit-identical-argument commuting assumptions.
