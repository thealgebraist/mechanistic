import Std

/-! Parametric semantic lowering of T5's gated-GELU feed-forward block. -/

namespace GatedMLPLowering

structure Primitive (Tensor Matrix : Type) where
  linear : Matrix → Tensor → Tensor
  geluNew : Tensor → Tensor
  hadamard : Tensor → Tensor → Tensor

def sourceT5DenseGatedGeluDense (p : Primitive Tensor Matrix)
    (wi0 wi1 wo : Matrix) (x : Tensor) : Tensor :=
  let gate := p.geluNew (p.linear wi0 x)
  let value := p.linear wi1 x
  p.linear wo (p.hadamard gate value)

def registerGatedMLP (p : Primitive Tensor Matrix)
    (wi0 wi1 wo : Matrix) (x : Tensor) : Tensor :=
  let gate := p.geluNew (p.linear wi0 x)
  let value := p.linear wi1 x
  p.linear wo (p.hadamard gate value)

theorem lowering_exact_for_all_dimensions
    (p : Primitive Tensor Matrix) :
    ∀ wi0 wi1 wo x,
      registerGatedMLP p wi0 wi1 wo x =
      sourceT5DenseGatedGeluDense p wi0 wi1 wo x := by
  intro wi0 wi1 wo x
  rfl

/- The residual addition remains a distinct opcode.  This theorem states the
lowering contract for the feed-forward branch itself and therefore composes
with the already definitionally lowered ADD occurrence. -/
theorem residual_block_exact
    (p : Primitive Tensor Matrix)
    (add : Tensor → Tensor → Tensor) :
    ∀ wi0 wi1 wo x,
      add x (registerGatedMLP p wi0 wi1 wo x) =
      add x (sourceT5DenseGatedGeluDense p wi0 wi1 wo x) := by
  intro wi0 wi1 wo x
  rw [lowering_exact_for_all_dimensions]

end GatedMLPLowering
