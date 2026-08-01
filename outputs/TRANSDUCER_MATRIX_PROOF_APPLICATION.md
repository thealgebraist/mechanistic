# Applying the transducer matrix proof to FLAN-T5

The PDF's central commutation idea is sound after two repairs. First, the
induction theorem must quantify over every reachable starting state, not only
the initial state. Second, probabilistic generation requires stochastic
operators on distributions (or explicit random-bit states), rather than binary
deterministic transition matrices.

The repaired theorem uses an arbitrary source distribution carrier, arbitrary
target distribution carrier, a projection `Q`, token-indexed transition
operators, and observation operators. If transition and observation
intertwining hold, it proves equality of the complete observation trace for
every finite token word. No finite-state assumption is needed. For finite
carriers this reduces to the PDF's stochastic matrix equations.

For the shared-ABI PRSL program, `Q` is the extraction of typed registers from
the pinned FLAN execution state. All 129 source and target operations invoke the
same callable with identical tensor arguments, so every local error is zero.
The corrected intertwining theorem therefore recovers exact equality of all
finite token traces relative to that ABI.

The probability-law formalization also supplies the missing minimality test.
Two source states are behaviorally equivalent when every finite continuation
has equal categorical mass from both states. Lean proves that every fiber of an
exact graph encoding lies inside one such equivalence class. Equivalently, if
any finite continuation distinguishes two states, every exact graph must assign
them distinct graph states. This is a lower-bound theorem: it identifies the
only legal merges, but does not assume that FLAN has many such merges.

The concrete Lean instantiation uses distinct source-backend and target-register
state types. It proves that the 129 generated opcode schedule commutes when each
hashed shared-callable binding satisfies its local commuting equation, then
derives equality for every prompt value and every finite continuation. This
replaces the earlier identity-machine proof, which compared one machine with
itself and therefore did not realize a nontrivial projection `Q`.

The argument does not close the portable proof. The 947-opcode binary32 program
is definitionally equivalent to its 129 macro opcodes, but PyTorch's matrix and
transcendental kernels have not been proved to refine those exact operational
rules.

Several claims in the PDF cannot be used: finiteness of the neural reachable
state space is assumed rather than proved; the proposed quotient is not smaller
than the source; 10,000 prompt matches do not imply universal equality; and the
reported entropy is inconsistent because `log2(81208) = 16.309334...`, not
`15.6565` bits. SentencePiece Viterbi algebra concerns tokenizer path selection,
not the FLAN decoder's stochastic kernel.
