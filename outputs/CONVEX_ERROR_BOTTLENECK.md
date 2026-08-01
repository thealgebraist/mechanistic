# Convex DAG error audit

The whole-program affine theorem is deliberately generic: it transports one
global error through a linear opcode list.  FLAN-T5 is more structured.  A
residual addition reads two named registers, attention reads a normalized query
and possibly encoder memory, and the final readout reads only the final decoder
register.  The convex DAG audit therefore propagates occurrence-level errors
through those actual dependencies.

For every produced real register with certified reachable sup bound `B`, the
clipped source/target contract gives the convex diameter cap

`d_infinity(x,y) <= 2 B`.

For a probability vector, total variation has diameter one.  Residual addition
uses the sharper recurrence `e_out <= e_left + e_right + bias`; unary and
multi-input primitives use their certified `gain * max(input errors) + bias`.
The generated JSON records the value before and after every cap, making the
first loss of a nontrivial estimate explicit.

This audit cannot by itself repair a saturated primitive certificate.  Its
purpose is to select the next convex refinement.  If saturation first occurs
inside RMS normalization, the useful object is a weighted projective metric on
the RMS ellipsoid.  If it occurs in attention, one should replace the simplex
diameter fallback by a log-sum-exp/Bregman transport certificate.  If only the
last softmax saturates, a direct logit-to-TV inequality is the right target.

## Current refinement result

The scale-relative RMS theorem removes the false `1/sqrt(epsilon)` numerical
explosion: its largest local roundoff bias is about `0.00560` real units, and
the first cap moves from RMSNorm (opcode 2) to attention (opcode 3).

Attention now retains head- and coordinate-specific weighted-ellipsoid
supports. It uses `||softmax(a)-softmax(b)||_1 <= ||a-b||_infinity` and shares
the pinned probability-normalization opcode between source and target. At
opcode 3 this reduces the collapsed gain from about `2.35e9` to `1.11e9` and
the local bias from about `14995` to `7126.52`.

The remaining first saturation is an interface loss. RMS roundoff is not an
arbitrary vector in a sup-norm ball: it has a coordinate-weighted, correlated
form inherited from normalization. Converting it to one scalar before the
Q/K/V maps permits impossible adversarial sign choices. The next language
revision should therefore make errors an algebraic datatype, for example
`Exact | RMSRadial(weight, relative, additive) | Box(radius) | SimplexTV(tv)`,
and give each opcode a constructor-specific transport rule. A fused
`RMS_ATTENTION` proof is the first concrete instance.

The tensor support arrays are checkpoint-derived in Python. The C++23 checker
independently verifies the emitted aggregate gain/bias inequalities, but does
not yet independently rescan every tensor entry for this attention refinement.
This remains a conditional numerical certificate, not a completed portable
universal approximation theorem.

## Constructor-aware audit

The corrected `RMSRadial` constructor separates a radial coefficient from an
additive coordinate box. The radial part is transported through Q/K/V using
the weighted ellipsoid support, while only the genuinely additive remainder is
multiplied by absolute matrix entries. For an exact incoming register, the
machine-checked composition rule permits an `RMSNORM; ATTENTION` pair to use a
single fused certificate.

This rule applies universally—not to sampled prompts—whenever the incoming
error constructor is `Exact`. In the current graph that happens for the first
encoder and first decoder RMS-attention pairs. Their bounds are respectively
`10727.53` and `7295.83`, both below their reachable output diameters. The first
active cap consequently moves from opcode 3 to opcode 5, the RMSNorm after the
first encoder residual. Active caps decrease from 82 to 80.

The next required constructor is a residual-aware error form. After attention,
`ADD` currently merges the exact embedding path and structured attention error
into an arbitrary box. Preserving that sum as an affine zonotope or support
function is necessary before the following RMSNorm can exploit radial
insensitivity.
