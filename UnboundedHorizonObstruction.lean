import Std

/-! A positive per-step error cannot yield a nontrivial uniform trace bound
over every finite continuation length by additive transport alone. -/
namespace UnboundedHorizonObstruction

theorem additive_bound_eventually_saturates
    (diameter epsilon : Nat) (heps : 0 < epsilon) :
    diameter ≤ diameter * epsilon := by
  exact Nat.le_mul_of_pos_right diameter heps

theorem exists_horizon_reaching_diameter
    (diameter epsilon : Nat) (heps : 0 < epsilon) :
    ∃ horizon, diameter ≤ horizon * epsilon := by
  exact ⟨diameter, additive_bound_eventually_saturates diameter epsilon heps⟩

theorem zero_error_never_accumulates (horizon : Nat) : horizon * 0 = 0 := by
  simp

end UnboundedHorizonObstruction
