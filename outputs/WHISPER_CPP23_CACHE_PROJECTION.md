# Generation cache output projection

Transformers 4.57.3 returns an `EncoderDecoderCache` when `return_legacy_cache` is `None` or `False`, and a four-tensor tuple per decoder layer when it is `True`. The C++23 ADT represents both public forms over one internal cache recurrence.

Both C++23 projections contain 4,617,216 binary32 values across four layers, three self-attention positions, and 1,500 cross-attention positions. Their bytes are identical. Compared with the corresponding PyTorch tensors, the worst maximum absolute error is `1.97887421e-05`. Generated token sequences match exactly, and an unknown representation is rejected.

This proves a finite output-container correspondence; it does not change or broaden the numerical backend proof boundary.
