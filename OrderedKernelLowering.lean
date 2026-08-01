import Std

/-!
Ordered scalar lowering for portable binary32 reductions and matrix entries.

This theorem deliberately separates two questions:

1. Is the PRSL reduction/matmul semantics fully defined by scalar operations?
2. Does a pinned vendor backend refine that ordered scalar semantics?

The first question is discharged here.  The second remains an explicit backend
obligation and cannot be inferred merely from the existence of an ATen schema.
-/
namespace OrderedKernelLowering

structure ScalarCompilerCertificate (Source Target : Type) where
  encode : Source → Target
  sourceZero : Source
  targetZero : Target
  sourceAdd : Source → Source → Source
  targetAdd : Target → Target → Target
  sourceMul : Source → Source → Source
  targetMul : Target → Target → Target
  zero_commutes : targetZero = encode sourceZero
  add_commutes : ∀ left right,
    targetAdd (encode left) (encode right) = encode (sourceAdd left right)
  mul_commutes : ∀ left right,
    targetMul (encode left) (encode right) = encode (sourceMul left right)

def sourceReduceAux
    (c : ScalarCompilerCertificate Source Target) : Source → List Source → Source
  | accumulator, [] => accumulator
  | accumulator, value :: rest =>
      sourceReduceAux c (c.sourceAdd accumulator value) rest

def targetReduceAux
    (c : ScalarCompilerCertificate Source Target) : Target → List Target → Target
  | accumulator, [] => accumulator
  | accumulator, value :: rest =>
      targetReduceAux c (c.targetAdd accumulator value) rest

theorem ordered_reduce_aux_commutes
    (c : ScalarCompilerCertificate Source Target) :
    ∀ values accumulator,
      targetReduceAux c (c.encode accumulator) (values.map c.encode) =
        c.encode (sourceReduceAux c accumulator values) := by
  intro values
  induction values with
  | nil =>
      intro accumulator
      rfl
  | cons value rest ih =>
      intro accumulator
      simp only [List.map_cons, targetReduceAux, sourceReduceAux]
      rw [c.add_commutes]
      exact ih (c.sourceAdd accumulator value)

def sourceReduce
    (c : ScalarCompilerCertificate Source Target) (values : List Source) : Source :=
  sourceReduceAux c c.sourceZero values

def targetReduce
    (c : ScalarCompilerCertificate Source Target) (values : List Target) : Target :=
  targetReduceAux c c.targetZero values

theorem ordered_reduce_commutes
    (c : ScalarCompilerCertificate Source Target) (values : List Source) :
    targetReduce c (values.map c.encode) = c.encode (sourceReduce c values) := by
  simp only [targetReduce, sourceReduce, c.zero_commutes]
  exact ordered_reduce_aux_commutes c values c.sourceZero

def sourceDotAux
    (c : ScalarCompilerCertificate Source Target) :
    Source → List Source → List Source → Source
  | accumulator, left :: leftRest, right :: rightRest =>
      sourceDotAux c
        (c.sourceAdd accumulator (c.sourceMul left right)) leftRest rightRest
  | accumulator, _, _ => accumulator

def targetDotAux
    (c : ScalarCompilerCertificate Source Target) :
    Target → List Target → List Target → Target
  | accumulator, left :: leftRest, right :: rightRest =>
      targetDotAux c
        (c.targetAdd accumulator (c.targetMul left right)) leftRest rightRest
  | accumulator, _, _ => accumulator

theorem ordered_dot_aux_commutes
    (c : ScalarCompilerCertificate Source Target) :
    ∀ left right accumulator,
      targetDotAux c (c.encode accumulator)
          (left.map c.encode) (right.map c.encode) =
        c.encode (sourceDotAux c accumulator left right) := by
  intro left
  induction left with
  | nil =>
      intro right accumulator
      simp [targetDotAux, sourceDotAux]
  | cons leftValue leftRest ih =>
      intro right accumulator
      cases right with
      | nil => simp [targetDotAux, sourceDotAux]
      | cons rightValue rightRest =>
          simp only [List.map_cons, targetDotAux, sourceDotAux]
          rw [c.mul_commutes, c.add_commutes]
          exact ih rightRest
            (c.sourceAdd accumulator (c.sourceMul leftValue rightValue))

def sourceDot
    (c : ScalarCompilerCertificate Source Target)
    (left right : List Source) : Source :=
  sourceDotAux c c.sourceZero left right

def targetDot
    (c : ScalarCompilerCertificate Source Target)
    (left right : List Target) : Target :=
  targetDotAux c c.targetZero left right

theorem ordered_matmul_entry_commutes
    (c : ScalarCompilerCertificate Source Target)
    (row column : List Source) :
    targetDot c (row.map c.encode) (column.map c.encode) =
      c.encode (sourceDot c row column) := by
  simp only [targetDot, sourceDot, c.zero_commutes]
  exact ordered_dot_aux_commutes c row column c.sourceZero

end OrderedKernelLowering
