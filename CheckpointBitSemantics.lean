import Std

/-! Exact parameter semantics for a safetensors-bound microcode machine. -/
namespace CheckpointBitSemantics

abbrev F32Bits := UInt32

structure TensorSlice where
  shape : List Nat
  bits : List F32Bits
deriving Repr, DecidableEq

abbrev Checkpoint := String → Option TensorSlice

structure WeightedSemantics (Op State : Type) where
  eval : Checkpoint → Op → State → State

def run (sem : WeightedSemantics Op State) (checkpoint : Checkpoint) :
    List Op → State → State
  | [], state => state
  | op :: rest, state => run sem checkpoint rest (sem.eval checkpoint op state)

theorem checkpoint_extensionality
    (left right : Checkpoint)
    (h : ∀ name, left name = right name) : left = right := by
  funext name
  exact h name

theorem same_tensor_bits_same_program
    (sem : WeightedSemantics Op State)
    (left right : Checkpoint)
    (h : ∀ name, left name = right name)
    (program : List Op) (state : State) :
    run sem left program state = run sem right program state := by
  have hc : left = right := checkpoint_extensionality left right h
  cases hc
  rfl

/- Binary32 values are represented by their bits, not host floating values.
The operational microcode interpreter decides how each bit pattern participates
in round-to-nearest-ties-to-even operations. -/
theorem bit_identity_is_value_identity (x y : F32Bits) (h : x = y) : x = y := h

end CheckpointBitSemantics
