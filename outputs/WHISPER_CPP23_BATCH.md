# C++23 Whisper batch semantics

One Transformers invocation received a real tensor of shape `[5, 80, 3000]` with five independently masked recordings. One C++23 process loaded the checkpoint and token vocabulary once, then executed five isolated item states in the same order. Every unpadded generated token and decoded transcript matched exactly.

| item | recording | valid feature frames | tokens | cached/full max error | transcript |
|---:|---|---:|---:|---:|---|
| 0 | `whisper_sample.wav` | 586 | 22 | 4.2e-05 | Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel. |
| 1 | `1272-128104-0001.wav` | 482 | 15 | 4.58e-05 | Nor is Mr. Quilter's manner less interesting than his matter. |
| 2 | `1272-128104-0002.wav` | 1249 | 36 | 4.15e-05 | He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similes drawn from eating and its results occur most readily to the mind. |
| 3 | `1272-128104-0003.wav` | 990 | 29 | 3.81e-05 | He has grave doubts whether Sir Frederick Layton's work is really Greek after all and can discover in it but little of rocky Ithaca. |
| 4 | `1272-128104-0010.wav` | 560 | 19 | 5.15e-05 | near the fire, and the ornaments Fred brought home from India on the mental board. |

The maximum incremental-cache versus full-prefix logit error was `5.14984e-05`. Every item independently visited all 74 graph nodes. The C++ execution is deliberately described as **sequential semantic batching with shared immutable weights**, not vectorized batched matrix multiplication. Timings in the JSON are diagnostic only because the C++ run enabled expensive full-prefix cache recomputation while Transformers did not.

This finite certificate checks batch ordering, variable valid lengths, state isolation, exact tokens, and text for these five recordings. It is not a proof for every possible batch or waveform.
