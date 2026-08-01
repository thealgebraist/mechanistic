import Std

namespace IEEEMatmulTransfer

structure Contract (Source Target : Type) where
  distance : Source → Target → Nat
  sourceMatmul : Source → Source
  targetMatmul : Target → Target
  gain : Nat
  roundingBias : Nat
  transfer : ∀ s t,
    distance (sourceMatmul s) (targetMatmul t) ≤
      gain * distance s t + roundingBias

theorem certified_affine_transfer (c : Contract Source Target) : ∀ s t,
    c.distance (c.sourceMatmul s) (c.targetMatmul t) ≤
      c.gain * c.distance s t + c.roundingBias := by
  exact c.transfer

theorem zero_input_discrepancy (c : Contract Source Target) : ∀ s t,
    c.distance s t = 0 →
    c.distance (c.sourceMatmul s) (c.targetMatmul t) ≤ c.roundingBias := by
  intro s t h
  simpa [h] using c.transfer s t

end IEEEMatmulTransfer
