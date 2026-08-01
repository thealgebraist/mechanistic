# Actual audio filter-cirquent test

The input is the real 5.855-second, 16 kHz LibriSpeech waveform `whisper_sample_1272-128104-0000.wav`. Frame **95** at nominal time **0.950 s** was selected deterministically as the active frame with the largest sum of pre-log Whisper Mel energy.

## Whisper path

- Frame RMS: `0.18389742`; Q15 peak: `12579`.
- Direct serialized-coefficient DFT versus PyTorch FFT relative error: `1.25177522e-07`.
- Direct serialized-coefficient Mel energy versus model Mel energy relative error: `1.36258976e-07`.
- The complete 80-channel `Fin 16` state exactly matches the packed state blob.
- Highest-mass Mel node: `11`, interval `420.0–500.0 Hz`, mass `0.205659`, class `15`.
- Whole-recording transcript: “Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.”

## CAR-FAC path

- The same PCM stream was executed through all 81 official CAR-FAC sections with IHC and closed-loop AGC.
- At sample `15280`, the strongest selected place uses section `73` with pole frequency `112.48 Hz.
- The 81 section energies were quotiented into 80 explicit place nodes; the complete selected-frame `Fin 16` vector exactly matches the packed state blob.
- Highest-mass cochlear node: `7`, mass `0.036343`, class `15`.
- Uncalibrated alternate transcript: “I'm gonna go to the next one.” (`WER=0.941`).

This is a concrete finite execution certificate. It demonstrates that nested nodes, coefficient-resource edges, state-delay edges, quotient edges, and probability nodes can carry real values. It does not prove that the CAR-FAC interface is semantically equivalent to Whisper's Mel interface.
