import Std
import WholeModelEquivalence
import GeneratedFlanProgram

/-! Exact all-sequence semantics when source and target share the same kernels. -/
namespace SharedBackendExact
open WholeModel

def machine {State Tok Weight : Type}
    (consume : State → Tok → State)
    (kernel : GeneratedFlanProgram.OpTag → State → State)
    (readout : State → Tok → Weight) : TokenTransducer State Tok Weight where
  step := fun state token =>
    ProgramComposition.runTarget (GeneratedFlanProgram.concreteProgram kernel)
      (consume state token)
  observe := readout

def identityCertificate
    {State Tok Weight : Type}
    (consume : State → Tok → State)
    (kernel : GeneratedFlanProgram.OpTag → State → State)
    (readout : State → Tok → Weight) :
    RegisterCompilerCertificate (machine consume kernel readout)
      (machine consume kernel readout) where
  encode := id
  observe_commutes := by intro h a; rfl
  step_commutes := by intro h a; rfl

theorem all_prompts_all_continuations_exact
    {State Tok Weight Prompt : Type} [One Weight] [Mul Weight]
    (consume : State → Tok → State)
    (kernel : GeneratedFlanProgram.OpTag → State → State)
    (readout : State → Tok → Weight)
    (initial : Prompt → State) : ∀ prompt continuation,
      traceWeight (machine consume kernel readout) (initial prompt) continuation =
      traceWeight (machine consume kernel readout) (initial prompt) continuation := by
  intro prompt continuation
  exact all_prompts_all_continuations
    (identityCertificate consume kernel readout) initial prompt continuation

end SharedBackendExact
