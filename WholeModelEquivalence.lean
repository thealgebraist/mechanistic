import Std

/-!
Universal semantic theorem for compiling a token model to a probabilistic
register machine.  The observation is a deterministic next-token weight
function; sampling happens outside the state transition.  Thus equality of
the observation at every prefix implies equality of every autoregressive trace
weight, without restricting the initial prompt or continuation.
-/

namespace WholeModel

structure TokenTransducer (H Tok Weight : Type) where
  step : H → Tok → H
  observe : H → Tok → Weight

def run (m : TokenTransducer H Tok Weight) : H → List Tok → H
  | h, [] => h
  | h, a :: rest => run m (m.step h a) rest

def traceWeight [One Weight] [Mul Weight]
    (m : TokenTransducer H Tok Weight) : H → List Tok → Weight
  | _, [] => 1
  | h, a :: rest => m.observe h a * traceWeight m (m.step h a) rest

/- A compiler certificate relates every source state to one register state.
The two commuting equations are exactly the corrected paper's functional
quotient obligations, except the target state may be infinite/register-valued. -/
structure RegisterCompilerCertificate
    (source : TokenTransducer H Tok Weight)
    (target : TokenTransducer R Tok Weight) where
  encode : H → R
  observe_commutes : ∀ h a, target.observe (encode h) a = source.observe h a
  step_commutes : ∀ h a, target.step (encode h) a = encode (source.step h a)

theorem run_commutes
    (c : RegisterCompilerCertificate source target) :
    ∀ h w, run target (c.encode h) w = c.encode (run source h w) := by
  intro h w
  induction w generalizing h with
  | nil => rfl
  | cons a rest ih =>
      simp only [run]
      rw [c.step_commutes]
      exact ih (source.step h a)

theorem observation_after_every_prefix
    (c : RegisterCompilerCertificate source target) :
    ∀ h pref next,
      target.observe (run target (c.encode h) pref) next =
      source.observe (run source h pref) next := by
  intro h pref next
  rw [run_commutes c h pref]
  exact c.observe_commutes (run source h pref) next

theorem every_finite_trace_weight_equal
    {H Tok Weight R : Type} [One Weight] [Mul Weight]
    {source : TokenTransducer H Tok Weight}
    {target : TokenTransducer R Tok Weight}
    (c : RegisterCompilerCertificate source target) :
    ∀ h w, traceWeight target (c.encode h) w = traceWeight source h w := by
  intro h w
  induction w generalizing h with
  | nil => rfl
  | cons a rest ih =>
      simp only [traceWeight]
      rw [c.observe_commutes, c.step_commutes]
      exact congrArg (fun z => source.observe h a * z) (ih (source.step h a))

/- The theorem is polymorphic in the initial state.  For FLAN-T5, an initial
state contains arbitrary encoder tokens and an empty decoder cache; therefore
the quantifier covers every finite tokenized prompt, not a fixed corpus. -/
theorem all_prompts_all_continuations
    {H Tok Weight R Prompt : Type} [One Weight] [Mul Weight]
    {source : TokenTransducer H Tok Weight}
    {target : TokenTransducer R Tok Weight}
    (c : RegisterCompilerCertificate source target)
    (initial : Prompt → H) :
    ∀ prompt continuation,
      traceWeight target (c.encode (initial prompt)) continuation =
      traceWeight source (initial prompt) continuation := by
  intro prompt continuation
  exact every_finite_trace_weight_equal c (initial prompt) continuation

end WholeModel
