import Std

/-!
Corrected form of the PDF's matrix commutation argument.  `SourceDist` and
`TargetDist` may be arbitrary probability-measure representations; no finite
reachable-state basis is assumed.  In a finite model, these maps are exactly
the stochastic matrices in the PDF.
-/
namespace ProbabilisticIntertwining

structure KernelSystem (Token Dist Obs : Type) where
  step : Token → Dist → Dist
  observe : Dist → Obs

def run (sys : KernelSystem Token Dist Obs) : Dist → List Token → Dist
  | state, [] => state
  | state, token :: rest => run sys (sys.step token state) rest

def trace (sys : KernelSystem Token Dist Obs) : Dist → List Token → List Obs
  | state, [] => [sys.observe state]
  | state, token :: rest => sys.observe state :: trace sys (sys.step token state) rest

structure IntertwiningCertificate
    (source : KernelSystem Token SourceDist Obs)
    (target : KernelSystem Token TargetDist Obs) where
  project : SourceDist → TargetDist
  transition_commutes : ∀ token state,
    target.step token (project state) = project (source.step token state)
  observation_commutes : ∀ state,
    target.observe (project state) = source.observe state

variable {Token SourceDist TargetDist Obs : Type}
variable {source : KernelSystem Token SourceDist Obs}
variable {target : KernelSystem Token TargetDist Obs}

theorem run_commutes
    (certificate : IntertwiningCertificate source target)
    (state : SourceDist) (word : List Token) :
    run target (certificate.project state) word =
      certificate.project (run source state word) := by
  induction word generalizing state with
  | nil => rfl
  | cons token rest ih =>
      simp only [run]
      rw [certificate.transition_commutes]
      exact ih (source.step token state)

theorem final_observation_exact
    (certificate : IntertwiningCertificate source target)
    (state : SourceDist) (word : List Token) :
    target.observe (run target (certificate.project state) word) =
      source.observe (run source state word) := by
  rw [run_commutes certificate]
  exact certificate.observation_commutes _

theorem full_trace_exact
    (certificate : IntertwiningCertificate source target)
    (state : SourceDist) (word : List Token) :
    trace target (certificate.project state) word = trace source state word := by
  induction word generalizing state with
  | nil => simp [trace, certificate.observation_commutes]
  | cons token rest ih =>
      simp only [trace]
      rw [certificate.observation_commutes, certificate.transition_commutes]
      exact congrArg (source.observe state :: ·) (ih (source.step token state))

/- A finite stochastic-matrix instance is obtained by taking `Dist` to be a
probability vector and `step token` to be left multiplication by its transition
matrix.  The theorem itself does not require that restriction. -/

end ProbabilisticIntertwining
