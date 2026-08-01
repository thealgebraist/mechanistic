# CAR-FAC temporal transport: exact target on the actual sample

The native Mel path already returned the target. The open task was to make the explicit CAR-FAC branch return the same text without substituting the Mel features. A deterministic least-squares search fitted affine transports from temporal CAR-FAC log-place features to the fixed 80-channel Whisper interface.

| radius | offsets | parameters | bytes | feature RMSE | WER | exact | transcript |
|---:|---|---:|---:|---:|---:|---|---|
| 0 | `[0]` | 6,480 | 25,920 | 0.088660 | 0.294 | False | Mr. Krueger is an apostle of the middle class, and we're going to welcome his gospel. |
| 1 | `[-1, 0, 1]` | 19,280 | 77,120 | 0.065542 | 0.059 | False | Mr. Crowder is the apostle of the middle classes and we are glad to welcome his gospel. |
| 2 | `[-2, -1, 0, 1, 2]` | 32,080 | 128,320 | 0.043196 | 0.000 | True | Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel. |

The smallest successful tested member is radius 2: offsets `[-2,-1,0,1,2]`, a `401 x 80` binary32 affine matrix, 32,080 parameters, and 128,320 bytes. It returns exactly:

> Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.

The two positive offsets are explicit batch-lookahead edges. Boundary indices are clamped. The graph remains acyclic over the already-available finite audio window, although this adapter is not zero-latency streaming.

This is deliberately labeled **single-sample specialization**: both the transport and its target Mel tensor were fitted from this recording. It proves that the actual CAR-FAC execution can be compiled through a small explicit graph to the requested output on this waveform. It does not establish generalization or universal semantic equivalence.
