import Std

/-! Compositional affine error transport for numerical register lowering. -/

namespace AffineProgramComposition

structure PairedOpcode (Source Target : Type) (distance : Source → Target → Nat) where
  sourceEval : Source → Source
  targetEval : Target → Target
  gain : Nat
  bias : Nat
  refines : ∀ s t,
    distance (sourceEval s) (targetEval t) ≤ gain * distance s t + bias

def runSource {S T : Type} {distance : S → T → Nat} :
    List (PairedOpcode S T distance) → S → S
  | [], s => s
  | op :: rest, s => runSource rest (op.sourceEval s)

def runTarget {S T : Type} {distance : S → T → Nat} :
    List (PairedOpcode S T distance) → T → T
  | [], t => t
  | op :: rest, t => runTarget rest (op.targetEval t)

def totalGain {S T : Type} {distance : S → T → Nat} :
    List (PairedOpcode S T distance) → Nat
  | [] => 1
  | op :: rest => totalGain rest * op.gain

def totalBias {S T : Type} {distance : S → T → Nat} :
    List (PairedOpcode S T distance) → Nat
  | [] => 0
  | op :: rest => totalGain rest * op.bias + totalBias rest

theorem program_error_affine
    {S T : Type} {distance : S → T → Nat}
    (program : List (PairedOpcode S T distance)) :
    ∀ s t,
      distance (runSource program s) (runTarget program t) ≤
        totalGain program * distance s t + totalBias program := by
  induction program with
  | nil => intro s t; simp [runSource, runTarget, totalGain, totalBias]
  | cons op rest ih =>
      intro s t
      simp only [runSource, runTarget, totalGain, totalBias]
      apply Nat.le_trans (ih (op.sourceEval s) (op.targetEval t))
      have hscaled := Nat.mul_le_mul_left (totalGain rest) (op.refines s t)
      calc
        totalGain rest * distance (op.sourceEval s) (op.targetEval t) + totalBias rest
            ≤ totalGain rest * (op.gain * distance s t + op.bias) + totalBias rest :=
              Nat.add_le_add_right hscaled _
        _ = (totalGain rest * op.gain) * distance s t +
              (totalGain rest * op.bias + totalBias rest) := by
                rw [Nat.mul_add, Nat.mul_assoc, Nat.add_assoc]

theorem zero_initial_error
    {S T : Type} {distance : S → T → Nat}
    (program : List (PairedOpcode S T distance))
    (s : S) (t : T) (initial : distance s t = 0) :
    distance (runSource program s) (runTarget program t) ≤ totalBias program := by
  simpa [initial] using program_error_affine program s t

end AffineProgramComposition
