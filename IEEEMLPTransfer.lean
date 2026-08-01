import Std

namespace IEEEMLPTransfer

structure Contract (Source Target : Type) where
  distance : Source → Target → Nat
  sourceMLP : Source → Source
  targetMLPClipped : Target → Target
  gain : Nat
  roundingBias : Nat
  transfer : ∀ s t, distance (sourceMLP s) (targetMLPClipped t) ≤
    gain * distance s t + roundingBias

theorem certified_affine_transfer (c : Contract Source Target) : ∀ s t,
    c.distance (c.sourceMLP s) (c.targetMLPClipped t) ≤
      c.gain * c.distance s t + c.roundingBias := c.transfer

end IEEEMLPTransfer
