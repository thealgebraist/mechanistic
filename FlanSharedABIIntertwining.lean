import Std
import WholeModelEquivalence
import GeneratedFlanProgram

/-!
Nontrivial shared-ABI instantiation: source backend states and target register
states are distinct.  Per-opcode commuting obligations imply one-step and
all-finite-trace equality for the generated 129-opcode FLAN schedule.
-/
namespace FlanSharedABIIntertwining
open WholeModel GeneratedFlanProgram

def runTags (primitive : OpTag → State → State) : List OpTag → State → State
  | [], state => state
  | tag :: rest, state => runTags primitive rest (primitive tag state)

structure SharedABICertificate
    (SourceState RegisterState Tok Weight Prompt : Type) where
  encode : SourceState → RegisterState
  sourceConsume : SourceState → Tok → SourceState
  registerConsume : RegisterState → Tok → RegisterState
  sourcePrimitive : OpTag → SourceState → SourceState
  registerPrimitive : OpTag → RegisterState → RegisterState
  sourceReadout : SourceState → Tok → Weight
  registerReadout : RegisterState → Tok → Weight
  sourceInitial : Prompt → SourceState
  registerInitial : Prompt → RegisterState
  consume_commutes : ∀ state token,
    registerConsume (encode state) token = encode (sourceConsume state token)
  primitive_commutes : ∀ tag state,
    registerPrimitive tag (encode state) = encode (sourcePrimitive tag state)
  readout_commutes : ∀ state token,
    registerReadout (encode state) token = sourceReadout state token
  initial_commutes : ∀ prompt,
    registerInitial prompt = encode (sourceInitial prompt)

theorem runTags_commutes
    (c : SharedABICertificate SourceState RegisterState Tok Weight Prompt) :
    ∀ tags state,
      runTags c.registerPrimitive tags (c.encode state) =
        c.encode (runTags c.sourcePrimitive tags state) := by
  intro tags state
  induction tags generalizing state with
  | nil => rfl
  | cons tag rest ih =>
      simp only [runTags]
      rw [c.primitive_commutes]
      exact ih (c.sourcePrimitive tag state)

def sourceMachine
    (c : SharedABICertificate SourceState RegisterState Tok Weight Prompt) :
    TokenTransducer SourceState Tok Weight where
  step state token :=
    runTags c.sourcePrimitive programTags (c.sourceConsume state token)
  observe := c.sourceReadout

def registerMachine
    (c : SharedABICertificate SourceState RegisterState Tok Weight Prompt) :
    TokenTransducer RegisterState Tok Weight where
  step state token :=
    runTags c.registerPrimitive programTags (c.registerConsume state token)
  observe := c.registerReadout

def compilerCertificate
    (c : SharedABICertificate SourceState RegisterState Tok Weight Prompt) :
    RegisterCompilerCertificate (sourceMachine c) (registerMachine c) where
  encode := c.encode
  observe_commutes := c.readout_commutes
  step_commutes := by
    intro state token
    simp only [sourceMachine, registerMachine]
    rw [c.consume_commutes]
    exact runTags_commutes c programTags (c.sourceConsume state token)

theorem generated_129_opcode_step_commutes
    (c : SharedABICertificate SourceState RegisterState Tok Weight Prompt)
    (state : SourceState) (token : Tok) :
    (registerMachine c).step (c.encode state) token =
      c.encode ((sourceMachine c).step state token) := by
  exact (compilerCertificate c).step_commutes state token

theorem all_prompts_all_finite_continuations_exact
    [One Weight] [Mul Weight]
    (c : SharedABICertificate SourceState RegisterState Tok Weight Prompt) :
    ∀ prompt continuation,
      traceWeight (registerMachine c) (c.registerInitial prompt) continuation =
      traceWeight (sourceMachine c) (c.sourceInitial prompt) continuation := by
  intro prompt continuation
  rw [c.initial_commutes]
  exact all_prompts_all_continuations
    (compilerCertificate c) c.sourceInitial prompt continuation

theorem generated_schedule_has_129_opcodes : programTags.length = 129 :=
  programTags_length

end FlanSharedABIIntertwining
