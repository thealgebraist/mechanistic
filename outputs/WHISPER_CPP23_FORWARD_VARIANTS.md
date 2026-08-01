# Graph-driven Whisper forward variants

The C++23 graph now exposes a full no-cache forward constructor for arbitrary valid decoder token-ID sequences. Each case executes generated nodes 0–69 directly and writes every `[position, 51864]` logit for comparison with `WhisperForConditionalGeneration.forward(..., use_cache=False)`.

| case | graph nodes | decoder positions | max absolute logit error | RMSE | cosine | top-token sequence exact |
|---|---:|---:|---:|---:|---:|---|
| `forced_prefix` | 70 | 2 | 4.48227e-05 | 1.30156e-05 | 0.999999999994 | yes |
| `observed_speech_prefix` | 70 | 6 | 3.91006e-05 | 1.01773e-05 | 0.999999999992 | yes |
| `arbitrary_valid_tokens` | 70 | 4 | 5.91278e-05 | 9.8887e-06 | 0.999999999998 | yes |
| `encoder_decoder_cross_head_masks` | 70 | 4 | 5.91278e-05 | 1.28201e-05 | 0.999999999999 | yes |
| `complete_hidden_state_tuples` | 70 | 4 | 3.91006e-05 | 9.43449e-06 | 0.999999999995 | yes |
| `complete_attention_tuples` | 70 | 4 | 3.19481e-05 | 9.77406e-06 | 0.999999999996 | yes |
| `supplied_encoder_memory` | 40 | 4 | 4.04119e-05 | 8.53223e-06 | 0.999999999996 | yes |
| `supplied_decoder_embeddings` | 40 | 4 | 4.04119e-05 | 8.53223e-06 | 0.999999999996 | yes |
| `decoder_mask_and_position_ids` | 40 | 4 | 3.24249e-05 | 8.62864e-06 | 0.999999999997 | yes |
| `supplied_key_value_cache` | 40 | 1 | 1.3113e-05 | 2.99807e-06 | 0.999999999999 | yes |

Worst maximum absolute logit error is `5.91278076e-05`, worst imported/updated cache error is `4.20212746e-06`, worst complete hidden-state tuple error is `0.000137329102`, and worst complete attention-tuple error is `8.19563866e-06`. The labelled objective, including a `-100` ignored position and Whisper's decoder-right-shift rule, has absolute loss error `4.53422247e-05`. Timestamp generation emits the exact Transformers token sequence and segment boundaries; on this recording the explicit segment is `0.00`–`5.44` seconds. The eight-head cross-attention transport, reflected median filter, and DTW path reproduce all `25` token timestamps with maximum error `1.71661377e-07` seconds. A 35.13-second, 3,513-frame recording executes two windows and six segments; both unconditioned and previous-segment-conditioned runs exactly match Transformers tokens, boundaries, and seek transitions. Concrete first-segment and all-segments prompt placement also exactly match, including the 448-position stopping path. This verifies explicit input features or encoder memory, token IDs or decoder embeddings, supplied position IDs, decoder/head masks, no-cache or supplied-cache execution, hidden-state and eager-attention outputs, labelled cross-entropy, prompts, segment/token timestamps, and long-form state transport. It is evidence, not a universal floating-point proof.
