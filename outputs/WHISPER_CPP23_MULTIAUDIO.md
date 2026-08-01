# C++23 Whisper validation on additional audio

Four additional deterministic records from the same public LibriSpeech validation corpus were run independently through Transformers and the native C++23 WAV-to-text graph. Every generated token ID and every decoded transcript matched exactly.

| record | tokens | cached/recomputed max logit error | WER vs corpus reference | C++23 transcript |
|---|---:|---:|---:|---|
| `1272-128104-0001` | 15 | 4.58e-05 | 0.091 | Nor is Mr. Quilter's manner less interesting than his matter. |
| `1272-128104-0002` | 36 | 4.15e-05 | 0.000 | He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similes drawn from eating and its results occur most readily to the mind. |
| `1272-128104-0003` | 29 | 3.81e-05 | 0.040 | He has grave doubts whether Sir Frederick Layton's work is really Greek after all and can discover in it but little of rocky Ithaca. |
| `1272-128104-0010` | 19 | 5.15e-05 | 0.067 | near the fire, and the ornaments Fred brought home from India on the mental board. |

Maximum cached-versus-recomputed logit error was `5.14984e-05`. Reference WER measures model transcription against the corpus text; it is separate from the exact C++23-versus-Transformers equivalence check.

This broadens concrete evidence to five speech recordings when combined with the original Quilter sample. It is not a universal numerical-equivalence proof for every possible waveform or platform.
