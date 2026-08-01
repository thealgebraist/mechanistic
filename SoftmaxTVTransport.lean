import Std

/-!
Interface between a certified logit error and scaled total variation.  The
analytic instance for softmax uses `TV ≤ tanh(δ) ≤ min 1 δ` when corresponding
logits differ by at most `δ` in sup norm.  This file isolates the numerical
scaling/rounding obligation from autoregressive trace composition.
-/

namespace SoftmaxTVTransport

structure Contract (Logits : Type) where
  units : Nat
  logitError : Logits → Logits → Nat
  totalVariation : Logits → Logits → Nat
  transport : ∀ x y,
    totalVariation x y ≤ min units (logitError x y)

theorem readout_is_capped_and_lipschitz
    (c : Contract Logits) : ∀ x y,
    c.totalVariation x y ≤ c.units ∧
    c.totalVariation x y ≤ c.logitError x y := by
  intro x y
  have h := c.transport x y
  rw [Nat.le_min] at h
  exact h

theorem uniform_logit_error_gives_uniform_tv
    (c : Contract Logits) (delta : Nat)
    (hlogit : ∀ x y, c.logitError x y ≤ delta) : ∀ x y,
    c.totalVariation x y ≤ min c.units delta := by
  intro x y
  rw [Nat.le_min]
  constructor
  · exact (readout_is_capped_and_lipschitz c x y).1
  · exact Nat.le_trans (readout_is_capped_and_lipschitz c x y).2 (hlogit x y)

end SoftmaxTVTransport
