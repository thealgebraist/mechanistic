import Std

/-! Discrete convex-mixture and potential-divergence transport skeleton. -/
namespace ConvexTransport

def totalWeight : List (Nat × Nat) → Nat
  | [] => 0
  | (w, _) :: xs => w + totalWeight xs

def weightedSum : List (Nat × Nat) → Nat
  | [] => 0
  | (w, v) :: xs => w * v + weightedSum xs

theorem convex_mixture_stays_in_hull : ∀ (xs : List (Nat × Nat)) (bound : Nat),
    (∀ p ∈ xs, p.2 ≤ bound) →
    weightedSum xs ≤ totalWeight xs * bound := by
  intro xs bound h
  induction xs with
  | nil => simp [weightedSum, totalWeight]
  | cons p rest ih =>
      have hp : p.2 ≤ bound := h p (by simp)
      have hr : ∀ q ∈ rest, q.2 ≤ bound := by
        intro q hq; exact h q (by simp [hq])
      have hmul := Nat.mul_le_mul_left p.1 hp
      have hrest := ih hr
      simp only [weightedSum, totalWeight]
      calc
        p.1 * p.2 + weightedSum rest ≤ p.1 * bound + totalWeight rest * bound :=
          Nat.add_le_add hmul hrest
        _ = (p.1 + totalWeight rest) * bound := by rw [Nat.add_mul]

/- A convex log-partition potential supplies a divergence (for softmax this is
KL/Bregman divergence).  Analytic instances prove `pinsker`; trace composition
can then use divergence rather than repeatedly taking coarse TV diameters. -/
structure PotentialTransport (State : Type) where
  units : Nat
  divergence : State → State → Nat
  totalVariation : State → State → Nat
  pinsker : ∀ x y,
    totalVariation x y * totalVariation x y ≤ units * divergence x y

theorem tv_squared_controlled_by_potential
    (c : PotentialTransport State) : ∀ x y,
    c.totalVariation x y * c.totalVariation x y ≤
      c.units * c.divergence x y := c.pinsker

end ConvexTransport
