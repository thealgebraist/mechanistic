# Deterministic score-policy lowering

Seven named C++23 policy constructors were exercised on the real Whisper sample and compared with the corresponding Transformers GenerationConfig override.

| case | C++23 ADT constructor | output tokens | exact Transformers sequence |
|---|---|---:|---:|
| `single_token_sequence_bias` | `SequenceBiasPolicy.AdditiveSequenceBias` | 21 | yes |
| `contextual_sequence_bias` | `SequenceBiasPolicy.AdditiveSequenceBias` | 21 | yes |
| `forced_bos_model_noop` | `ForcedBeginningPolicy.ForcedBeginningToken` | 22 | yes |
| `forced_eos_at_maximum` | `ForcedEndingPolicy.ForcedEndingTokens` | 5 | yes |
| `exponential_eos_decay` | `ExponentialEosPolicy.ExponentialEosDecay` | 2 | yes |
| `repair_invalid_finite_model_path` | `InvalidLogitPolicy.RepairInvalidLogits` | 22 | yes |
| `renormalize_logits` | `LogitNormalizationPolicy.NormalizeLogProbabilities` | 22 | yes |

The contextual sequence-bias case verifies suffix-sensitive state rather than only a vocabulary-wide token bias. Forced EOS and exponential decay alter termination at different positions. `forced_bos_token_id` is an intentional model-path no-op because Whisper Tiny English begins with two prepared decoder tokens, so Transformers' `cur_len == 1` condition is false; the C++ constructor retains the same conditional semantics. Invalid-value repair is executable but is an identity on this checkpoint because all raw model logits are finite. This is finite behavioral evidence, not universal floating-point equivalence.
