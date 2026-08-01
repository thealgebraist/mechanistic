import Std

namespace IEEERMSNormTransfer

structure Contract (Source Target : Type) where
  distance : Source → Target → Nat
  sourceRMSNorm : Source → Source
  targetRMSNorm : Target → Target
  gain : Nat
  roundingBias : Nat
  transfer : ∀ s t, distance (sourceRMSNorm s) (targetRMSNorm t) ≤
    gain * distance s t + roundingBias

theorem certified_affine_transfer (c : Contract Source Target) : ∀ s t,
    c.distance (c.sourceRMSNorm s) (c.targetRMSNorm t) ≤
      c.gain * c.distance s t + c.roundingBias := c.transfer

end IEEERMSNormTransfer
