# Whisper Tiny English audio-to-text deconstruction

The pinned `openai/whisper-tiny.en` checkpoint is represented as a 74-opcode probabilistic register graph. The binary contains 167 tensors and has SHA-256 `db59695928ded6043adaef491a53ef4e12da9611184d77c53baa691a60b958ad`. Every tensor is referenced by at least one graph opcode; tied token embeddings are also the language-model readout.

## Concrete audio

The input is a 5.855-second LibriSpeech utterance (93,680 mono 16 kHz samples). Whisper produced:

> Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.

Reference transcription: `MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL`.

The processor emitted shape `[1, 80, 3000]` with 586 active feature frames before padding. The encoder memory has shape `1 x 1500 x 384`. Greedy replay after the forced no-timestamps prefix matched the processed-logit argmax at 22/22 recorded text positions.

| position | emitted token | conditional mass | greedy argmax |
|---:|---|---:|---|
| 0 | `ĠMr` | 0.926911 | True |
| 1 | `.` | 0.968144 | True |
| 2 | `ĠQu` | 0.753293 | True |
| 3 | `il` | 0.918178 | True |
| 4 | `ter` | 0.992879 | True |
| 5 | `Ġis` | 0.991173 | True |
| 6 | `Ġthe` | 0.994233 | True |
| 7 | `Ġapostle` | 0.801416 | True |
| 8 | `Ġof` | 0.997599 | True |
| 9 | `Ġthe` | 0.996033 | True |
| 10 | `Ġmiddle` | 0.768884 | True |
| 11 | `Ġclasses` | 0.932443 | True |

## Probabilistic graph state

The graph state is the ADT `(encoder memory, finite decoder token stack, decoder position, four-layer K/V cache, generation policy)`. Its readout is a categorical mass function over 51,864 tokens. A transition samples or selects one token, appends it to the stack and cache, and repeats the decoder schedule.

This is not a finite-state quotient: audio and tensor registers are continuous, while token/cache length varies up to the configured target limit. The explicit graph is linear in model layers and tensors instead of enumerating exponentially many activation states.

## PDF commutation certificate

For a projection `Q` from the source Whisper execution state to typed graph registers, the required obligations are the probabilistic forms of the PDF equations:

```text
target.mass(Q(state), token) = source.mass(state, token)
target.step(Q(state), token) = Q(source.step(state, token))
```

Once these hold for the frontend and every opcode, induction gives equal probability for every finite transcript continuation for every valid audio input. The current artifact proves graph structure, complete checkpoint coverage, concrete execution, and the generic induction theorem. Universal numerical equality remains conditional on a shared primitive ABI; an independent portable proof would additionally have to refine STFT, convolution, matrix reductions, layer normalization, GELU, and softmax.
