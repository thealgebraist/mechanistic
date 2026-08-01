# C++23 graph versus Transformers: speed and memory

Both paths use the same `144.1 MiB` Whisper Tiny English binary32 checkpoint and the same `whisper_sample.wav` waveform. Every one of the 5 + 5 runs returned exactly:

> Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.

| implementation | fresh-process wall time, median (range) | peak RSS, median (range) |
|---|---:|---:|
| Explicit C++23 graph, Accelerate | 0.811 s (0.806–1.342) | 297.6 MiB (297.6–302.0) |
| Hugging Face Transformers, Python/PyTorch | 3.295 s (3.113–4.506) | 724.1 MiB (722.4–724.2) |

For this sample and machine, the C++23 graph is **4.06× faster** end to end and the Transformers process uses **2.43×** its peak memory; equivalently, C++23 reduces whole-process peak RSS by **58.9%**.

After Python/PyTorch/Transformers imports have completed inside the worker, the Transformers medians are model/processor loading `0.152 s`, frontend `0.004 s`, and autoregressive generation `0.197 s`. Their sum is much smaller than fresh-process wall time because importing the Python ML stack dominates startup. The C++ number is the complete executable from process launch through WAV parsing, lazy checkpoint loading, Mel computation, encoder, cached decoder, and byte-token decoding. A persistent already-loaded C++ mode was not measured, so the `0.201 s` loaded Transformers pipeline must not be compared directly with the `0.811 s` fresh C++ process.

## Measurement boundary

- Each observation is a new process, but the operating-system file cache is deliberately not flushed. This is a fresh-process benchmark, not a cold-disk benchmark.
- Peak RSS is the process high-water mark. It includes runtime and library overhead, temporary activations, caches, and the checkpoint mapping/copy; it is not parameter bytes alone.
- The C++ production path uses only incremental decoder K/V caches. The separate regression path still compares cached logits against full causal-prefix recomputation; that expensive verification is excluded here.
- Both implementations retain the same checkpoint, so this benchmark demonstrates a smaller execution runtime, not compression of the model's information.
