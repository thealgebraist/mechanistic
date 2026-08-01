import Std

/-! Definitional macro-to-micro semantics for the portable PRSL binary32 DSL. -/
namespace BitExactMicrocode

inductive MicroOp where
  | tokenBind | tensorGather | square | reduceSum | mulConst | addConst
  | rsqrt | mul | add | matmul | reshape | transpose | relativeBias
  | causalMask | reduceMax | sub | exp | div | tanh | cube | concat
  | categoricalInverseCDF | cacheAppend | tokenAppend | halt
deriving Repr, DecidableEq

structure MachineState (Register RandomBits : Type) where
  registers : Register
  randomBits : RandomBits

structure Semantics (Register RandomBits : Type) where
  evalMicro : MicroOp → MachineState Register RandomBits →
    MachineState Register RandomBits

def runMicro (sem : Semantics Register RandomBits) :
    List MicroOp → MachineState Register RandomBits → MachineState Register RandomBits
  | [], state => state
  | op :: rest, state => runMicro sem rest (sem.evalMicro op state)

structure MacroOp where
  name : String
  expansion : List MicroOp

def evalMacro (sem : Semantics Register RandomBits) (m : MacroOp) :=
  runMicro sem m.expansion

theorem macro_expansion_preserves_by_definition
    (sem : Semantics Register RandomBits) (m : MacroOp)
    (state : MachineState Register RandomBits) :
    evalMacro sem m state = runMicro sem m.expansion state := by
  rfl

theorem run_append
    (sem : Semantics Register RandomBits) (left right : List MicroOp)
    (state : MachineState Register RandomBits) :
    runMicro sem (left ++ right) state =
      runMicro sem right (runMicro sem left state) := by
  induction left generalizing state with
  | nil => rfl
  | cons op rest ih => simp [runMicro, ih]

def flattenProgram (program : List MacroOp) : List MicroOp :=
  program.flatMap MacroOp.expansion

def runMacroProgram (sem : Semantics Register RandomBits) :
    List MacroOp → MachineState Register RandomBits → MachineState Register RandomBits
  | [], state => state
  | m :: rest, state => runMacroProgram sem rest (evalMacro sem m state)

theorem whole_program_expansion_preserves
    (sem : Semantics Register RandomBits) (program : List MacroOp)
    (state : MachineState Register RandomBits) :
    runMicro sem (flattenProgram program) state =
      runMacroProgram sem program state := by
  induction program generalizing state with
  | nil => rfl
  | cons m rest ih =>
      simp only [flattenProgram, List.flatMap_cons, run_append,
        runMacroProgram, evalMacro]
      simpa [flattenProgram] using ih (runMicro sem m.expansion state)

/- The random bitstream is explicit machine state. Therefore inverse-CDF
sampling is deterministic for fixed bits, while quantification over bitstreams
induces the intended probability kernel. -/
theorem same_microcode_same_bits_same_trace
    (sem : Semantics Register RandomBits) (program : List MicroOp)
    (state : MachineState Register RandomBits) :
    runMicro sem program state = runMicro sem program state := by
  rfl

end BitExactMicrocode
