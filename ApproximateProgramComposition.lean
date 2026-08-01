import Std

/-! Additive error refinement for an ordered source/register opcode list. -/

namespace ApproximateProgramComposition

structure PairedOpcode (Source Target : Type) (distance : Source → Target → Nat) where
  sourceEval : Source → Source
  targetEval : Target → Target
  localBudget : Nat
  refines : ∀ s t,
    distance (sourceEval s) (targetEval t) ≤ distance s t + localBudget

def runSource {S T : Type} {distance : S → T → Nat} :
    List (PairedOpcode S T distance) → S → S
  | [], s => s
  | op :: rest, s => runSource rest (op.sourceEval s)

def runTarget {S T : Type} {distance : S → T → Nat} :
    List (PairedOpcode S T distance) → T → T
  | [], t => t
  | op :: rest, t => runTarget rest (op.targetEval t)

def totalBudget {S T : Type} {distance : S → T → Nat} :
    List (PairedOpcode S T distance) → Nat
  | [] => 0
  | op :: rest => op.localBudget + totalBudget rest

theorem program_error_adds
    {S T : Type} {distance : S → T → Nat}
    (program : List (PairedOpcode S T distance)) :
    ∀ s t,
      distance (runSource program s) (runTarget program t) ≤
        distance s t + totalBudget program := by
  induction program with
  | nil => intro s t; exact Nat.le_refl _
  | cons op rest ih =>
      intro s t
      simp only [runSource, runTarget, totalBudget]
      exact Nat.le_trans (ih (op.sourceEval s) (op.targetEval t))
        (by
          have hop := op.refines s t
          omega)

theorem zero_initial_error
    {S T : Type} {distance : S → T → Nat}
    (program : List (PairedOpcode S T distance))
    (s : S) (t : T) (initial : distance s t = 0) :
    distance (runSource program s) (runTarget program t) ≤ totalBudget program := by
  simpa [initial] using program_error_adds program s t

end ApproximateProgramComposition
