import Std

/-!
Parametric lowering of T5 attention.  Tensor shapes, head count, sequence/cache
length, masks, bias, and matrices are all abstract parameters.  `Primitive`
fixes the backend operation order shared by source and register semantics.
-/

namespace AttentionLowering

structure Primitive (Tensor Matrix Bias Mask Scores Probabilities : Type) where
  project : Matrix → Tensor → Tensor
  splitHeads : Tensor → Tensor
  dotScores : Tensor → Tensor → Scores
  addBias : Scores → Bias → Scores
  addMask : Scores → Mask → Scores
  softmaxLast : Scores → Probabilities
  weightedValues : Probabilities → Tensor → Tensor
  mergeHeads : Tensor → Tensor

def sourceT5Attention (p : Primitive Tensor Matrix Bias Mask Scores Probabilities)
    (wq wk wv wo : Matrix) (queryInput keyValueInput : Tensor)
    (bias : Bias) (mask : Mask) : Tensor :=
  let q := p.splitHeads (p.project wq queryInput)
  let k := p.splitHeads (p.project wk keyValueInput)
  let v := p.splitHeads (p.project wv keyValueInput)
  let scores := p.addMask (p.addBias (p.dotScores q k) bias) mask
  let probabilities := p.softmaxLast scores
  p.project wo (p.mergeHeads (p.weightedValues probabilities v))

def registerAttention (p : Primitive Tensor Matrix Bias Mask Scores Probabilities)
    (wq wk wv wo : Matrix) (queryInput keyValueInput : Tensor)
    (bias : Bias) (mask : Mask) : Tensor :=
  let q := p.splitHeads (p.project wq queryInput)
  let k := p.splitHeads (p.project wk keyValueInput)
  let v := p.splitHeads (p.project wv keyValueInput)
  let scores := p.addMask (p.addBias (p.dotScores q k) bias) mask
  let probabilities := p.softmaxLast scores
  p.project wo (p.mergeHeads (p.weightedValues probabilities v))

theorem lowering_exact
    (p : Primitive Tensor Matrix Bias Mask Scores Probabilities) :
    ∀ wq wk wv wo queryInput keyValueInput bias mask,
      registerAttention p wq wk wv wo queryInput keyValueInput bias mask =
      sourceT5Attention p wq wk wv wo queryInput keyValueInput bias mask := by
  intro wq wk wv wo queryInput keyValueInput bias mask
  rfl

end AttentionLowering
