# Hierarchical discrete audio filter graph

## Construction

The expanded representation is a graph-of-graphs. A hypernode contains a reusable detector template and an indexed instance family instead of copying its scalar arithmetic for every bin, band, cochlear place, sample, or frame.

The finite ADTs are `SampleQ15 = Fin 65536`, `F32Bits = BitVec 32`, `EnergyClass = Fin 16`, `FrequencyNode = Fin 80`, `CochlearPlace = Fin 81`, and `Token = Fin 51864`. The coefficient blob contains the actual 400-point Hann window, the actual 80x201 Whisper Mel matrix, explicit 201x400 direct-DFT real and imaginary tables, and every numeric coefficient array designed by the pinned official CAR-FAC implementation.

## Relation to established visual calculi

The proposed notation is a **typed hierarchical probabilistic signal-flow hypergraph**. “Filter cirquent” is a convenient project nickname, not an established mathematical term.

- Classical signal-flow graphs contribute gain, sum, delay, and feedback nodes. Mason's original formulation relates graphs directly to systems of equations.
- Categorical signal-flow/string diagrams contribute typed open boxes, composition by wiring, copying/merging generators, feedback as trace, and equational graph rewriting. Interacting Hopf-algebra calculi give a complete graphical language for important classes of linear relations and signal-flow systems.
- Forney normal/factor graphs contribute variables as edges, local constraints as vertices, and exact sum-product inference on cycle-free graphs.
- Cirquent calculus contributes the key resource intuition: graph-shaped expressions can explicitly share a subexpression or resource instead of duplicating a proof-tree branch.

Our additions are finite bit-vector types, nested template instances, coefficient-bit resource nodes, quotient maps, positive-measure nodes, and a distinction between deterministic data edges and stochastic-kernel edges. Thus the graph can be read both operationally as a filter program and probabilistically as a factorization.

## Special edges

| edge | meaning |
|---|---|
| `GRAPH_INSTANCE` | invoke a nested graph template with an index parameter |
| `PARAMETER_BITS` | immutable finite coefficient bits |
| `STATE_DELAY` | recurrent state from sample `t` to `t+1` |
| `CASCADE_PLACE` | same-sample basal-to-apical cochlear propagation |
| `AGC_FEEDBACK` / `OHC_FEEDBACK` | biological gain-control dependencies |
| `QUOTIENT_MEMBERSHIP` | merge fine bins, packets, or sections into an interpretable set |
| `POSITIVE_MEASURE` | map activity to nonnegative energy |
| `PROBABILITY_NORMALIZE` | normalize energies to a categorical mass |
| `MODEL_INTERFACE` | pass 80 channels to the fixed Whisper suffix |

## Why nesting matters

The direct Mel graph has about 323,303 scalar primitive nodes per analyzed frame, 189,455,558 for this active utterance, and 970,232,303 for the padded 30-second window. Center padding produces 3,001 analyzed frames; the final frame is explicitly dropped to produce Whisper's 3,000 output frames. The log-energy floor depends on the maximum over the complete 80x3,000 window. The CAR-FAC estimate is 3,188,160,000 scalar nodes for 30 seconds. These counts are useful flattening estimates, not backend instruction counts.

The hierarchical JSON is 56,588 bytes and its coefficient blob is 712,684 bytes, compared with the 151,060,136-byte Whisper checkpoint. The representation stays small because templates and indexed instance families share repeated structure. It still references the checkpoint for the 74-op neural suffix, so the standalone exact package remains checkpoint-sized.

Including the 117,200-byte trajectory blob, the hierarchical addition is 886,472 raw bytes (0.587% of the checkpoint) or 262,594 bytes when its three components are gzip-compressed independently. This is plausible overhead for an explicit frontend graph; it is not a replacement for the checkpoint parameters.

The five finite sample trajectories are now genuinely serialized in `audio_frequency_quotient_states.bin`: 117,200 bytes, exactly 586 frames x 80 four-bit classes x five methods.

## Proof boundary

- Coefficient blocks, offsets, hashes, finite types, hierarchy, memberships, and packed sample states are explicit and mechanically checked.
- Representative direct-DFT and Mel detector instances are replayed from the serialized coefficient bits against PyTorch, and all 586 stored Mel `Fin16` frame states are reconstructed from the actual processor output.
- The direct DFT graph defines an ordered finite-bit implementation, but bit-for-bit equality with PyTorch's FFT kernel is not proved.
- CAR-FAC coefficient arrays are extracted from the pinned official implementation and the graph mirrors its update dependencies, but an opcode-level source equivalence proof remains open.
- The Mel path is the intended Whisper interface. The CAR-FAC-to-Whisper edge is marked `MODEL_INTERFACE_CANDIDATE_UNPROVED`; the observed transcription already refutes naïve equivalence.

## Visual-calculus sources

- S. J. Mason, “Feedback Theory—Some Properties of Signal Flow Graphs,” 1953, DOI `10.1109/JRPROC.1953.274449`.
- F. Bonchi, P. Sobociński, and F. Zanasi, “Interacting Hopf Algebras,” [arXiv:1403.7048](https://arxiv.org/abs/1403.7048), and “The Calculus of Signal Flow Diagrams I,” [author repository](https://eprints.soton.ac.uk/396532/).
- G. D. Forney Jr., “Codes on Graphs: Normal Realizations,” 2001, DOI `10.1109/18.910573`.
- G. Japaridze and B. Lamichhane, “Cirquent Calculus in a Nutshell,” [arXiv:2108.12552](https://arxiv.org/abs/2108.12552).
