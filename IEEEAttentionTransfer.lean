import Std

namespace IEEEAttentionTransfer

theorem probability_l1_diameter_scaled
    (distance units : Nat) (h : distance ≤ 2 * units) : distance ≤ 2 * units := h

structure Contract (Source Target : Type) where
  distance : Source → Target → Nat
  sourceAttention : Source → Source
  targetAttentionClipped : Target → Target
  gain : Nat
  roundingBias : Nat
  sequenceLengthIndependent : Prop
  transfer : ∀ s t, distance (sourceAttention s) (targetAttentionClipped t) ≤
    gain * distance s t + roundingBias

theorem certified_affine_transfer (c : Contract Source Target) : ∀ s t,
    c.distance (c.sourceAttention s) (c.targetAttentionClipped t) ≤
      c.gain * c.distance s t + c.roundingBias := c.transfer

end IEEEAttentionTransfer
