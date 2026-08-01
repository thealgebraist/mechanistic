import Std

/-!
Discrete affine wrapper for a binary32 addition certificate.  The analytic
IEEE premise supplies `roundingError ≤ bias`; triangle decomposition and the
global sup-error transfer are checked here.
-/

namespace IEEEAddTransfer

theorem two_inputs_plus_rounding
    (leftError rightError roundingError inputError bias : Nat)
    (hleft : leftError ≤ inputError)
    (hright : rightError ≤ inputError)
    (hround : roundingError ≤ bias) :
    leftError + rightError + roundingError ≤ 2 * inputError + bias := by
  omega

structure Contract (Source Target : Type) where
  distance : Source → Target → Nat
  sourceAdd : Source → Source → Source
  targetAdd : Target → Target → Target
  roundingBias : Nat
  decomposes : ∀ a b a' b',
    distance (sourceAdd a b) (targetAdd a' b') ≤
      distance a a' + distance b b' + roundingBias

theorem affine_gain_two
    (c : Contract Source Target) : ∀ a b a' b' e,
    c.distance a a' ≤ e → c.distance b b' ≤ e →
    c.distance (c.sourceAdd a b) (c.targetAdd a' b') ≤
      2 * e + c.roundingBias := by
  intro a b a' b' e ha hb
  exact Nat.le_trans (c.decomposes a b a' b') (by omega)

end IEEEAddTransfer
