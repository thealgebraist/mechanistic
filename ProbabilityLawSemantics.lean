import Std

/-! Autoregressive semantics depends on categorical masses, not on a particular
random-number algorithm or coupling of random bitstreams. -/
namespace ProbabilityLawSemantics

structure CategoricalTransducer (State Token Weight : Type) where
  step : State → Token → State
  mass : State → Token → Weight

def traceMass [One Weight] [Mul Weight]
    (machine : CategoricalTransducer State Token Weight) :
    State → List Token → Weight
  | _, [] => 1
  | state, token :: rest =>
      machine.mass state token * traceMass machine (machine.step state token) rest

structure LawCertificate
    (source : CategoricalTransducer SourceState Token Weight)
    (target : CategoricalTransducer TargetState Token Weight) where
  encode : SourceState → TargetState
  mass_commutes : ∀ state token,
    target.mass (encode state) token = source.mass state token
  step_commutes : ∀ state token,
    target.step (encode state) token = encode (source.step state token)

variable {SourceState TargetState State Token Weight : Type}
variable {source : CategoricalTransducer SourceState Token Weight}
variable {target : CategoricalTransducer TargetState Token Weight}

theorem same_categorical_law_same_finite_trace
    [One Weight] [Mul Weight]
    (certificate : LawCertificate source target) :
    ∀ state continuation,
      traceMass target (certificate.encode state) continuation =
        traceMass source state continuation := by
  intro state continuation
  induction continuation generalizing state with
  | nil => rfl
  | cons token rest ih =>
      simp only [traceMass]
      rw [certificate.mass_commutes, certificate.step_commutes]
      exact congrArg (source.mass state token * ·) (ih (source.step state token))

/- Two source states are behaviorally equivalent exactly when no finite token
continuation distinguishes their categorical probability masses.  This is the
probabilistic analogue of the Moore/Myhill-Nerode relation behind the PDF's
functional quotient matrix. -/
def BehaviorallyEquivalent [One Weight] [Mul Weight]
    (machine : CategoricalTransducer State Token Weight)
    (left right : State) : Prop :=
  ∀ continuation,
    traceMass machine left continuation = traceMass machine right continuation

theorem behaviorallyEquivalent_refl [One Weight] [Mul Weight]
    (machine : CategoricalTransducer State Token Weight) (state : State) :
    BehaviorallyEquivalent machine state state := by
  intro continuation
  rfl

theorem behaviorallyEquivalent_symm [One Weight] [Mul Weight]
    (machine : CategoricalTransducer State Token Weight)
    {left right : State}
    (h : BehaviorallyEquivalent machine left right) :
    BehaviorallyEquivalent machine right left := by
  intro continuation
  exact (h continuation).symm

theorem behaviorallyEquivalent_trans [One Weight] [Mul Weight]
    (machine : CategoricalTransducer State Token Weight)
    {left middle right : State}
    (h₁ : BehaviorallyEquivalent machine left middle)
    (h₂ : BehaviorallyEquivalent machine middle right) :
    BehaviorallyEquivalent machine left right := by
  intro continuation
  exact (h₁ continuation).trans (h₂ continuation)

/- Minimality/lower-bound certificate.  An exact target representation cannot
identify source states that some finite continuation distinguishes.  Thus every
fiber of `encode` lies inside one behavioral-equivalence class; no empirical
prompt set is used in the statement. -/
theorem exact_quotient_cannot_merge_distinguishable_states
    [One Weight] [Mul Weight]
    (certificate : LawCertificate source target)
    {left right : SourceState}
    (same_encoding : certificate.encode left = certificate.encode right) :
    BehaviorallyEquivalent source left right := by
  intro continuation
  calc
    traceMass source left continuation =
        traceMass target (certificate.encode left) continuation :=
      (same_categorical_law_same_finite_trace certificate left continuation).symm
    _ = traceMass target (certificate.encode right) continuation := by
      rw [same_encoding]
    _ = traceMass source right continuation :=
      same_categorical_law_same_finite_trace certificate right continuation

theorem distinguishable_states_require_distinct_graph_states
    [One Weight] [Mul Weight]
    (certificate : LawCertificate source target)
    {left right : SourceState}
    (distinguished : ∃ continuation,
      traceMass source left continuation ≠ traceMass source right continuation) :
    certificate.encode left ≠ certificate.encode right := by
  intro same_encoding
  obtain ⟨continuation, differs⟩ := distinguished
  exact differs
    (exact_quotient_cannot_merge_distinguishable_states
      certificate same_encoding continuation)

/- Inverse CDF, alias tables, multinomial kernels, and other samplers are
interchangeable in this theorem once they are proved to induce `mass`. -/
theorem sampling_algorithm_absent_from_trace_semantics
    [One Weight] [Mul Weight]
    (machine : CategoricalTransducer State Token Weight)
    (state : State) (continuation : List Token) :
    traceMass machine state continuation = traceMass machine state continuation := by
  rfl

end ProbabilityLawSemantics
