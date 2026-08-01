# Convex transport route for the FLAN register program

The portable proof should track convex sets and potentials rather than one
global componentwise box.

1. **RMS ellipsoids.** After RMSNorm,
   `sum_i (y_i / w_i)^2 <= d_model`. This is much tighter than bounding every
   component independently and then multiplying by a matrix row L1 norm.

2. **Attention convex hulls.** A softmax row is a probability law, so every
   attention head lies in the convex hull of its projected values. This removes
   sequence length from the state bound. `ConvexTransport.lean` proves the
   finite scaled convex-mixture lemma by induction over an arbitrary list.

3. **Log-partition potentials.** `softmax(z) = grad logsumexp(z)`. The associated
   Bregman divergence is KL divergence. Tracking KL before converting to TV via
   Pinsker avoids inserting the full simplex diameter at every attention.

4. **Weighted-ellipsoid support functions.** RMSNorm gives the sharper set
   `sum_i (h_i / rms_weight_i)^2 <= d`. Therefore the exact support bound for
   an lm-head row `a` is `sqrt(d) * ||a .* rms_weight||_2`. On the checkpoint,
   the outer Euclidean ball first lowers the old `154309.49` logit envelope to
   `8580.87`; retaining the diagonal weights lowers it again to `280.12`.
   The generic 512-term rounding contribution falls from about `9.42` to
   `0.524` and finally to `0.0171`.

The improvement is real but not yet enough for a complete nontrivial
all-sequence TV certificate. The next target is a layerwise ellipsoid or
weighted quadratic invariant whose induced matrix gains remain below the very
large box gains currently used for MLP and attention.

## Whole-stack propagation result

The same weighted support functions are now propagated through all 16 blocks.
For attention, each V coordinate gets its exact RMS-ellipsoid support and the
softmax output remains a convex mixture. For a gated MLP, both input projections
get weighted supports `U_j,V_j`, and each output is bounded by
`sum_j |Wo_ij| U_j V_j`. Residuals use the triangle inequality.

Compared with the former global boxes, the resulting source envelopes improve
encoder hidden state by about `37118x`, decoder hidden state by `15822x`, and
logits by `551x`. Feeding occurrence-local bounds into the IEEE transfers cuts
the maximum MLP bias by about `2300x` and the maximum attention gain by about
`5200x`. The final TV bound still saturates because the remaining worst-case
gains and the softmax diameter fallback are too large.
