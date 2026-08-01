import Std

/-! Certified clipping keeps approximate register states inside the source envelope. -/

namespace ClippedRegisterSemantics

structure Contract (Source Target : Type) where
  distance : Source → Target → Nat
  reachable : Source → Prop
  clip : Target → Target
  clip_nonexpansive : ∀ s t, reachable s → distance s (clip t) ≤ distance s t

theorem clipping_preserves_any_affine_transfer
    (c : Contract Source Target)
    (sourceOp : Source → Source) (targetOp : Target → Target)
    (gain bias : Nat)
    (source_closed : ∀ s, c.reachable s → c.reachable (sourceOp s))
    (transfer : ∀ s t, c.reachable s →
      c.distance (sourceOp s) (targetOp t) ≤ gain * c.distance s t + bias) :
    ∀ s t, c.reachable s →
      c.distance (sourceOp s) (c.clip (targetOp t)) ≤
        gain * c.distance s t + bias := by
  intro s t hs
  exact Nat.le_trans (c.clip_nonexpansive (sourceOp s) (targetOp t) (source_closed s hs))
    (transfer s t hs)

end ClippedRegisterSemantics
