import Std

/-!
Structured error semantics for the probabilistic register language.

`RMSRadial` preserves the correlated radial part of RMS normalization instead
of immediately widening it to an arbitrary box.  Opcode certificates may
pattern-match on constructors; `fuseRMSAttention_exact` is the first such rule.
-/

namespace StructuredErrorTransport

inductive ErrorShape where
  | exact
  | rmsRadial (radial additive : Nat)
  | box (radius : Nat)
  | simplexTV (units : Nat)
deriving Repr, DecidableEq

def radius : ErrorShape → Nat
  | .exact => 0
  | .rmsRadial radial additive => radial + additive
  | .box r => r
  | .simplexTV u => u

structure OpcodeTransfer where
  run : ErrorShape → ErrorShape
  gain : Nat
  bias : Nat
  sound : ∀ e, StructuredErrorTransport.radius (run e) ≤
    gain * StructuredErrorTransport.radius e + bias

def compose (first second : OpcodeTransfer) : OpcodeTransfer where
  run e := second.run (first.run e)
  gain := second.gain * first.gain
  bias := second.gain * first.bias + second.bias
  sound := by
    intro e
    calc
      radius (second.run (first.run e))
          ≤ second.gain * radius (first.run e) + second.bias := second.sound _
      _ ≤ second.gain * (first.gain * radius e + first.bias) + second.bias := by
          exact Nat.add_le_add_right
            (Nat.mul_le_mul_left second.gain (first.sound e)) second.bias
      _ = (second.gain * first.gain) * radius e +
          (second.gain * first.bias + second.bias) := by
          simp [Nat.mul_add, Nat.mul_assoc, Nat.add_assoc]

def rmsStructured (radial additive fallbackGain fallbackBias : Nat) :
    ErrorShape → ErrorShape
  | .exact => .rmsRadial radial additive
  | e => .box (fallbackGain * radius e + fallbackBias)

def attentionStructured
    (expectedRadial expectedAdditive fusedBias fallbackGain fallbackBias : Nat) :
    ErrorShape → ErrorShape
  | .rmsRadial r a =>
      if r = expectedRadial ∧ a = expectedAdditive then .box fusedBias
      else .box (fallbackGain * (r + a) + fallbackBias)
  | e => .box (fallbackGain * radius e + fallbackBias)

theorem fuseRMSAttention_exact
    (radial additive fusedBias fallbackGain fallbackBias : Nat) :
    attentionStructured radial additive fusedBias fallbackGain fallbackBias
      (rmsStructured radial additive fallbackGain fallbackBias .exact) =
      .box fusedBias := by
  simp [rmsStructured, attentionStructured]

theorem fused_exact_radius
    (radial additive fusedBias fallbackGain fallbackBias : Nat) :
    radius
      (attentionStructured radial additive fusedBias fallbackGain fallbackBias
        (rmsStructured radial additive fallbackGain fallbackBias .exact)) =
      fusedBias := by
  simp [fuseRMSAttention_exact, radius]

theorem boxFallback_is_explicit
    (radial additive fusedBias fallbackGain fallbackBias r : Nat) :
    attentionStructured radial additive fusedBias fallbackGain fallbackBias
      (.box r) = .box (fallbackGain * r + fallbackBias) := by
  rfl

end StructuredErrorTransport
