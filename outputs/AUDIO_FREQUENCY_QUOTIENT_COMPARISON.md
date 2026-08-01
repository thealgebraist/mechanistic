# Five frequency/place quotient models for Whisper audio

All five methods analyze the same 5.855-second, 16 kHz LibriSpeech waveform and feed an 80-channel representation into the unchanged Whisper Tiny English encoder. Each active frame becomes a probabilistic DAG layer: explicit frequency/place nodes carry normalized energy masses, their log energies are quotiented into 16 discrete classes, and the resulting 80-channel register is passed to Whisper.

| frontend | nodes | cosine vs Mel | mean JS bits | WER vs reference | exact transcript | transcription |
|---|---:|---:|---:|---:|---|---|
| Triangular Mel FIR bank | 80 | 1.0000 | 0.0000 | 0.000 | True | Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel. |
| Uniform ideal subbands | 80 | 0.5440 | 0.4595 | 1.412 | False | I think I'll stop this and leave for a bit more. Yes, I'll get it there. I think I'll stop this. |
| Sparse Goertzel resonators | 80 | 0.7569 | 0.0302 | 0.000 | True | Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel. |
| db4 wavelet packet | 80 | 0.7581 | 0.4786 | 1.059 | False | I think I was happy to meet my teachers. I know that I'm happy to meet my students. |
| Lyon CAR-FAC cochlea | 80 | 0.0791 | 0.6437 | 0.941 | False | I'm gonna go to the next one. |

The triangular Mel method is the model's actual frontend and is bit-close to the Transformers processor (`max error < 2e-6`). The other four are controlled replacements, not claimed equivalent. Their WER and divergence values are concrete evidence for this waveform only.

The cochlear lane executes Google's official NumPy implementation of Lyon's CAR-FAC at pinned commit `c74663cc7d05713ae2f2308765eb040530a81c7f`. It uses 81 nonlinear asymmetric-resonator sections, the two-capacitor inner-hair-cell stage, and closed-loop multi-timescale AGC. Adjacent cochlear places are explicitly quotiented to 80 nodes. Frame energy is the mean squared neural-activity-pattern output, giving a positive measure suitable for normalization; this probability mass is a constructed graph measure, not a claim that CAR-FAC firing rates themselves are categorical probabilities.

Each quantized frame uses 80 four-bit classes (40 bytes) instead of 80 binary32 values (320 bytes), an 8x representation reduction before graph/temporal compression. The finite utterance gives a DAG over 586 time layers. A universal audio model is a parametric DAG schema because the frame count varies.

The quotient certificate records every node's frequency interval and exact DFT-bin, wavelet-packet, or CAR-FAC-section membership. Probability mass is `band_energy / total_frame_energy`; normalization error is checked numerically. Exact semantic equivalence to Whisper is proved only for the Mel frontend under the shared numerical ABI. Proving an alternate frontend equivalent would require a uniform bound connecting its frequency masses to Whisper logits for every waveform.
