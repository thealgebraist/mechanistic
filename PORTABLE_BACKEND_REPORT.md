# Portable Whisper backend report

The graph executor now has a `WHISPER_PORTABLE_BACKEND` build mode. It supplies
the exact row-major binary32 `sgemm` and `sdot` subset used by all 74 generated
Whisper nodes, with deterministic left-to-right accumulation and checked
shape/stride/transpose/scalar contracts. The default build remains the
Accelerate implementation.

## Verification

```sh
c++ -std=c++23 -O3 -DWHISPER_PORTABLE_BACKEND whisper_graph_cpp23.cpp -lz \
  -o work/whisper_graph_cpp23_portable
c++ -std=c++23 -O3 verify_portable_backend.cpp -o work/verify_portable_backend
work/verify_portable_backend
```

`verify_portable_backend.cpp` is an executable correspondence check for dot
and GEMM, including all matrix values used in the check and a timed 20x128³
GEMM workload. A full graph binary was also compiled successfully on a system
without Accelerate headers or libraries.

## Boundary

The portable slice covers the graph's vendor BLAS dependency only. It is not a
claim of universal bitwise equivalence across compilers, vectorization, or
math libraries. FFT, quantized/int8 GEMM, vendor vector intrinsics, alternate
generation algorithms, and PyTorch/Accelerate transcript comparisons require
their existing fixtures and remain explicit unsupported extensions in this
backend contract. Full end-to-end comparison is therefore conditional on the
checkpoint/audio fixture bundle being present.
