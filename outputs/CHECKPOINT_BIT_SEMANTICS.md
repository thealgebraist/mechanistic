# Exact checkpoint-bit semantics

The portable PRSL program no longer treats model weights as host-language
floating values loaded through PyTorch. Every safetensors tensor is bound by:

- tensor name, shape and element count;
- absolute and relative byte offsets;
- exact little-endian binary32 slice bytes;
- per-slice SHA-256 and CRC32;
- all graph opcode references to that tensor.

All 190 checkpoint tensors are binary32 and contain finite values. The 129-opcode
graph contains 189 weight-reference occurrences to 188 unique tensors. Every
referenced tensor is present, and the 190 slices form one contiguous partition
of the checkpoint data region through byte 307,867,048.

The Python generator computes cryptographic hashes and reference bindings. An
independent C++23 verifier reads the raw file directly, checks every shape/byte
count, recomputes each CRC, rejects nonfinite binary32 patterns, and verifies
contiguous full-file coverage. It does not call PyTorch or safetensors libraries.

Lean represents tensor values as lists of `UInt32` bit patterns. It proves that
two checkpoints with extensionally identical named slices produce identical
execution under any checkpoint-parametric microcode semantics. This is a
parameter-identity theorem; the individual binary32 arithmetic operations are
still specified separately by the 947-opcode operational DSL.

This removes Python tensor loading from the parameter trust boundary. It does
not yet prove that PyTorch's compute kernels use the same reduction order and
transcendental implementations as the portable microcode.
