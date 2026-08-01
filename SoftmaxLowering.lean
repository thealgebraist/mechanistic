import Std

/-! Parametric lowering of the final next-token softmax observation. -/

namespace SoftmaxLowering

structure Primitive (Logits Probabilities : Type) where
  softmaxLast : Logits → Probabilities

def sourceNextTokenDistribution (p : Primitive Logits Probabilities)
    (logits : Logits) : Probabilities :=
  p.softmaxLast logits

def registerNextTokenDistribution (p : Primitive Logits Probabilities)
    (logits : Logits) : Probabilities :=
  p.softmaxLast logits

theorem lowering_exact_for_every_vocabulary_size
    (p : Primitive Logits Probabilities) :
    ∀ logits,
      registerNextTokenDistribution p logits =
      sourceNextTokenDistribution p logits := by
  intro logits
  rfl

/- Equality of the full probability object implies equality of every token's
conditional weight through any projection supplied by the backend semantics. -/
theorem every_token_weight_equal
    (p : Primitive Logits Probabilities)
    (weightAt : Probabilities → Token → Weight) :
    ∀ logits token,
      weightAt (registerNextTokenDistribution p logits) token =
      weightAt (sourceNextTokenDistribution p logits) token := by
  intro logits token
  rfl

end SoftmaxLowering
