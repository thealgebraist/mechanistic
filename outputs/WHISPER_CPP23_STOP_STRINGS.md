# Explicit token-byte stop-string automaton

The C++23 matcher stores a non-empty `StopStringSet`. After each selected token, it joins that token's exact byte-BPE payload to the previously decoded bytes and accepts when any stop string occurrence overlaps the newly selected token. This captures whole-token matches, strings crossing token boundaries, and the reference criterion's deliberate overhang behavior where a stop string ends inside the final indivisible token.

The oracle is Transformers `4.57.3` `StopStringCriteria`, source hash `495628e4c877f667fbbd9ed4cc83b680de9eac2abc97ec3903f6e9219af3ecd2`.

| case | stops | emitted tokens | termination | tokens exact |
|---|---|---:|---|---:|
| `whole_token` | `[' apostle']` | 8 | stop | yes |
| `three_token_span` | `['ilter is']` | 6 | stop | yes |
| `ends_inside_final_token` | `['middle cl']` | 12 | stop | yes |
| `starts_inside_previous_token` | `['dle classes']` | 12 | stop | yes |
| `multiple_alternatives` | `['absent', 'glad to']` | 18 | stop | yes |
| `period_completion` | `['gospel.']` | 22 | stop | yes |
| `no_match_eos` | `['this string is absent']` | 22 | EOS | yes |

All `7` complete token sequences match. The `no_match_eos` case proves ordinary EOS remains distinct from stop-string acceptance.

This certificate currently covers greedy batch-one generation. The matcher itself is selection-independent, but plumbing it through every beam and long-form branch remains necessary before `stop_strings` can be called a full generic override. Transformers 4.57.3's tokenizer keyword leaks into the Whisper forward call on the direct configuration route, so the oracle invokes the same pinned `StopStringCriteria` directly.
