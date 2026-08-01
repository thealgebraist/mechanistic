import Std
import WholeModelEquivalence

/-!
Finite-horizon approximation of a token transducer by a probabilistic register
machine.  Distances are represented in integer `Unit`s (for example uint16
total-variation units).  The only analytic premise is `trace_composes`: a
backend refinement must prove that its chosen trace distance is bounded by the
sum of the one-step conditional distances along a common token prefix.

The theorem itself is independent of prompts, vocabulary size, hidden-state
dimension, and model architecture.  In particular, it quantifies over every
initial prompt and every finite continuation up to the requested horizon.
-/

namespace ApproximateWholeModel

open WholeModel

variable {H Tok Weight R Prompt : Type}

structure SequentialDistance
    (source : TokenTransducer H Tok Weight)
    (target : TokenTransducer R Tok Weight) where
  /-- Certified distance between the two next-token distributions. -/
  oneStep : H → R → Nat
  /-- Certified distance between distributions on complete token traces. -/
  trace : H → R → List Tok → Nat
  trace_nil : ∀ h r, trace h r [] = 0
  /-- Sequential kernel composition (coupling/TV chain rule). -/
  trace_cons : ∀ h r a rest,
    trace h r (a :: rest) ≤
      oneStep h r + trace (source.step h a) (target.step r a) rest

structure ApproximateRegisterCompilerCertificate
    (source : TokenTransducer H Tok Weight)
    (target : TokenTransducer R Tok Weight)
    (metric : SequentialDistance source target) where
  encode : H → R
  localBudget : Nat
  one_step_bounded : ∀ h, metric.oneStep h (encode h) ≤ localBudget
  step_commutes : ∀ h a,
    target.step (encode h) a = encode (source.step h a)

structure BoundedTraceDistance
    {source : TokenTransducer H Tok Weight}
    {target : TokenTransducer R Tok Weight}
    (metric : SequentialDistance source target) where
  diameter : Nat
  trace_bounded : ∀ h r w, metric.trace h r w ≤ diameter

def accumulatedBudget
    {source : TokenTransducer H Tok Weight}
    {target : TokenTransducer R Tok Weight}
    (metric : SequentialDistance source target) : H → R → List Tok → Nat
  | _, _, [] => 0
  | h, r, a :: rest =>
      metric.oneStep h r +
        accumulatedBudget metric (source.step h a) (target.step r a) rest

theorem trace_le_accumulated
    {source : TokenTransducer H Tok Weight}
    {target : TokenTransducer R Tok Weight}
    (metric : SequentialDistance source target) :
    ∀ h r w, metric.trace h r w ≤ accumulatedBudget metric h r w := by
  intro h r w
  induction w generalizing h r with
  | nil => simp [accumulatedBudget, metric.trace_nil]
  | cons a rest ih =>
      exact Nat.le_trans (metric.trace_cons h r a rest)
        (Nat.add_le_add_left (ih (source.step h a) (target.step r a)) _)

theorem accumulated_le_length_mul
    {source : TokenTransducer H Tok Weight}
    {target : TokenTransducer R Tok Weight}
    {metric : SequentialDistance source target}
    (c : ApproximateRegisterCompilerCertificate source target metric) :
    ∀ (h : H) (w : List Tok),
      accumulatedBudget metric h (c.encode h) w ≤ w.length * c.localBudget := by
  intro h w
  induction w generalizing h with
  | nil => simp [accumulatedBudget]
  | cons a rest ih =>
      simp only [accumulatedBudget, List.length_cons]
      rw [c.step_commutes]
      calc
        metric.oneStep h (c.encode h) +
              accumulatedBudget metric (source.step h a)
                (c.encode (source.step h a)) rest
            ≤ c.localBudget + rest.length * c.localBudget :=
              Nat.add_le_add (c.one_step_bounded h) (ih (source.step h a))
        _ = (rest.length + 1) * c.localBudget := by
              rw [Nat.add_mul, Nat.one_mul, Nat.add_comm]

theorem every_finite_trace_approximated
    {source : TokenTransducer H Tok Weight}
    {target : TokenTransducer R Tok Weight}
    {metric : SequentialDistance source target}
    (c : ApproximateRegisterCompilerCertificate source target metric) :
    ∀ (h : H) (w : List Tok),
      metric.trace h (c.encode h) w ≤ w.length * c.localBudget := by
  intro h w
  exact Nat.le_trans (trace_le_accumulated metric h (c.encode h) w)
    (accumulated_le_length_mul c h w)

theorem all_prompts_bounded_horizon
    {source : TokenTransducer H Tok Weight}
    {target : TokenTransducer R Tok Weight}
    {metric : SequentialDistance source target}
    (c : ApproximateRegisterCompilerCertificate source target metric)
    (initial : Prompt → H) :
    ∀ (prompt : Prompt) (continuation : List Tok) (horizon : Nat),
      continuation.length ≤ horizon →
      metric.trace (initial prompt) (c.encode (initial prompt)) continuation ≤
        horizon * c.localBudget := by
  intro prompt continuation horizon hlen
  exact Nat.le_trans (every_finite_trace_approximated c (initial prompt) continuation)
    (Nat.mul_le_mul_right c.localBudget hlen)

theorem all_prompts_bounded_horizon_capped
    {source : TokenTransducer H Tok Weight}
    {target : TokenTransducer R Tok Weight}
    {metric : SequentialDistance source target}
    (bounded : BoundedTraceDistance metric)
    (c : ApproximateRegisterCompilerCertificate source target metric)
    (initial : Prompt → H) :
    ∀ (prompt : Prompt) (continuation : List Tok) (horizon : Nat),
      continuation.length ≤ horizon →
      metric.trace (initial prompt) (c.encode (initial prompt)) continuation ≤
        min bounded.diameter (horizon * c.localBudget) := by
  intro prompt continuation horizon hlen
  rw [Nat.le_min]
  constructor
  · exact bounded.trace_bounded _ _ _
  · exact all_prompts_bounded_horizon c initial prompt continuation horizon hlen

end ApproximateWholeModel
