# Speed and memory: explicit adapter versus Whisper

Measured on the same actual LibriSpeech waveform. Times are wall-clock seconds. Peak RSS is whole-process memory and therefore includes Python, NumPy, PyTorch, and Transformers; the algorithmic tensor figure isolates the adapter's live arrays.

| path | measured time | peak RSS | stored coefficients |
|---|---:|---:|---:|
| CAR-FAC→Mel affine adapter only | 0.236 ms median (20 warm runs) | 44.2 MiB process; 6.54 MiB live tensors | 125.3 KiB |
| Native Mel + Whisper decoder | 0.192 s loaded; 3.335 s cold | 722.3 MiB | 144.1 MiB checkpoint |
| CAR-FAC graph + adapter + same Whisper decoder | 4.683 s loaded; 7.679 s cold | 972.2 MiB | adapter plus same checkpoint |

The adapter coefficient file is **1,177.2× smaller** than the Whisper checkpoint. The graph path is not a replacement for Whisper: it replaces only the audio frontend/interface and then invokes the unchanged neural encoder-decoder. Consequently its end-to-end peak RSS remains model-dominated. The adapter-only process RSS is also not the adapter's intrinsic memory requirement; its explicit live float arrays total 6.54 MiB, including the materialized 3000×401 design matrix.

On this run, the complete Python CAR-FAC graph path was **24.3× slower** and used **1.35× the peak RSS** of native Mel + Whisper. Almost all of that time is the unoptimized reference CAR-FAC frontend (4.486 s), not the affine graph adapter (0.236 ms median).

Both complete paths returned exactly: “Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.”
