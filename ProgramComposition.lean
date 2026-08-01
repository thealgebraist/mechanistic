import Std

/-!
Compositional simulation for a lowered register program.  This isolates the
FLAN-specific work into one local preservation proof per opcode occurrence.
-/

namespace ProgramComposition

structure PairedOpcode (Source Target : Type) (Rel : Source → Target → Prop) where
  sourceEval : Source → Source
  targetEval : Target → Target
  preserves : ∀ {s t}, Rel s t → Rel (sourceEval s) (targetEval t)

def runSource {S T : Type} {Rel : S → T → Prop} :
    List (PairedOpcode S T Rel) → S → S
  | [], s => s
  | op :: rest, s => runSource rest (op.sourceEval s)

def runTarget {S T : Type} {Rel : S → T → Prop} :
    List (PairedOpcode S T Rel) → T → T
  | [], t => t
  | op :: rest, t => runTarget rest (op.targetEval t)

theorem program_preserves_relation
    {S T : Type} {Rel : S → T → Prop}
    (program : List (PairedOpcode S T Rel)) :
    ∀ {s t}, Rel s t → Rel (runSource program s) (runTarget program t) := by
  induction program with
  | nil =>
      intro s t h
      exact h
  | cons op rest ih =>
      intro s t h
      exact ih (op.preserves h)

structure ReadoutPair (Source Target Tok Weight : Type)
    (Rel : Source → Target → Prop) where
  sourceObserve : Source → Tok → Weight
  targetObserve : Target → Tok → Weight
  agrees : ∀ {s t}, Rel s t → ∀ token,
    targetObserve t token = sourceObserve s token

theorem program_observation_commutes
    {S T Tok Weight : Type} {Rel : S → T → Prop}
    (program : List (PairedOpcode S T Rel))
    (readout : ReadoutPair S T Tok Weight Rel) :
    ∀ {s t}, Rel s t → ∀ token,
      readout.targetObserve (runTarget program t) token =
      readout.sourceObserve (runSource program s) token := by
  intro s t h token
  exact readout.agrees (program_preserves_relation program h) token

/- Autoregressive cache updates are programs too.  If the one-token decoder
program preserves the relation, repeatedly invoking it preserves the relation
for an arbitrary finite continuation. -/
def repeatSource {S T : Type} {Rel : S → T → Prop}
    (stepProgram : List (PairedOpcode S T Rel)) : S → Nat → S
  | s, 0 => s
  | s, n + 1 => repeatSource stepProgram (runSource stepProgram s) n

def repeatTarget {S T : Type} {Rel : S → T → Prop}
    (stepProgram : List (PairedOpcode S T Rel)) : T → Nat → T
  | t, 0 => t
  | t, n + 1 => repeatTarget stepProgram (runTarget stepProgram t) n

theorem repeated_program_preserves_relation
    {S T : Type} {Rel : S → T → Prop}
    (stepProgram : List (PairedOpcode S T Rel)) :
    ∀ n {s t}, Rel s t →
      Rel (repeatSource stepProgram s n) (repeatTarget stepProgram t n) := by
  intro n
  induction n with
  | zero =>
      intro s t h
      exact h
  | succ n ih =>
      intro s t h
      exact ih (program_preserves_relation stepProgram h)

end ProgramComposition
