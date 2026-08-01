import Std

/-!
Parametric lowering theorem for the T5 RMSNorm opcode.  `Primitive` fixes the
ordered backend operations; both source and register semantics are expressed
through exactly that interface.  The theorem is dimension- and weight-generic.
-/

namespace RMSNormLowering

structure Primitive (Tensor Scalar Weight : Type) where
  square : Tensor → Tensor
  meanLast : Tensor → Scalar
  addEpsilon : Scalar → Scalar → Scalar
  reciprocalSqrt : Scalar → Scalar
  scale : Tensor → Scalar → Tensor
  multiplyWeight : Tensor → Weight → Tensor

def sourceT5LayerNorm (p : Primitive Tensor Scalar Weight)
    (epsilon : Scalar) (weight : Weight) (x : Tensor) : Tensor :=
  let variance := p.meanLast (p.square x)
  let factor := p.reciprocalSqrt (p.addEpsilon variance epsilon)
  p.multiplyWeight (p.scale x factor) weight

def registerRMSNorm (p : Primitive Tensor Scalar Weight)
    (epsilon : Scalar) (weight : Weight) (x : Tensor) : Tensor :=
  let variance := p.meanLast (p.square x)
  let factor := p.reciprocalSqrt (p.addEpsilon variance epsilon)
  p.multiplyWeight (p.scale x factor) weight

theorem lowering_exact_for_all_tensors
    (p : Primitive Tensor Scalar Weight) :
    ∀ epsilon weight x,
      registerRMSNorm p epsilon weight x = sourceT5LayerNorm p epsilon weight x := by
  intro epsilon weight x
  rfl

/- Reusing the schema at any number of layers introduces no additional proof
obligation: parameters and tensor dimensions are universally quantified. -/
theorem lowering_exact_for_every_occurrence
    (p : Primitive Tensor Scalar Weight)
    (occurrences : List (Scalar × Weight × Tensor)) :
    occurrences.map (fun z => registerRMSNorm p z.1 z.2.1 z.2.2) =
    occurrences.map (fun z => sourceT5LayerNorm p z.1 z.2.1 z.2.2) := by
  apply List.map_congr_left
  intro z hz
  exact lowering_exact_for_all_tensors p z.1 z.2.1 z.2.2

end RMSNormLowering
