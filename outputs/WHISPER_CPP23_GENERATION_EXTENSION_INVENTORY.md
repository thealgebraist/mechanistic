# Whisper generation `kwargs` and configuration inventory

This inventory is derived from Transformers `4.57.3` at verification time. It closes the top-level `**kwargs` row for the pinned model: all 74 generation-configuration values are represented, all 17 model-forward kwargs route to previously audited ADTs, and unknown kwargs are rejected by the installed framework.

This does **not** claim every non-default generic GenerationMixin algorithm is converted. Categorical and contrastive sampling plus prompt-lookup speculation, monotonic-deadline stopping across every converted search interpreter, classic left-hash/self-hash watermarking, all implemented SynthID schedulers, and standard, sampled, diverse-group, and phrase/disjunction-constrained beam search are lowered. Common-vocabulary deterministic external assistance now has an explicit two-model product state, but its four configuration rows remain `CPP23_PARTIAL_OVERRIDE` until different-tokenizer UAG, sampled speculative rejection, early exit, adaptive ROC thresholds, cross-call schedule persistence, and a useful smaller checkpoint are covered. Contrastive candidate evaluation has the explicit low-memory sequential schedule. `return_legacy_cache` is a tested output-container projection over identical cache tensors. Dynamic/static cache allocation is threaded through greedy and beam-family state branches; contrastive and assisted generation preserve their source-mandated dynamic-full/dynamic overrides. The non-quantized `cache_config` no-op is an explicit ADT. Generic prefill chunking has mode-dependent typed semantics: sample search requires a cache and then passes unsupported `position_ids` to Whisper, while beam search ignores the field because only `_sample` dispatches the helper. DoLa, quantized cache, and unbatched classifier-free guidance are explicitly model-rejected for Whisper's encoder-decoder/Mel interface; `bos_token_id` and encoder-token repetition processors are explicitly model-ignored by the pinned Whisper source paths.

## GenerationConfig fields

| field | pinned value | C++ constructor | non-default override status |
|---|---|---|---|
| `_from_model_config` | `false` | `FromModelConfig` | METADATA_FIXED |
| `alignment_heads` | `[[1,0],[2,0],[2,5],[3,0],[3,1],[3,2],[3,3],[3,4]]` | `AlignmentHeads` | CPP23_NAMED_OVERRIDE |
| `assistant_confidence_threshold` | `0.4` | `AssistantConfidenceThreshold` | CPP23_PARTIAL_OVERRIDE |
| `assistant_early_exit` | `null` | `AssistantEarlyExit` | PINNED_INACTIVE_GENERIC_EXTENSION |
| `assistant_lookbehind` | `10` | `AssistantLookbehind` | PINNED_INACTIVE_GENERIC_EXTENSION |
| `bad_words_ids` | `null` | `BadWordsIds` | CPP23_NAMED_OVERRIDE |
| `begin_suppress_tokens` | `[220,50256]` | `BeginSuppressTokens` | CPP23_NAMED_OVERRIDE |
| `bos_token_id` | `50257` | `BosTokenId` | CPP23_MODEL_IGNORED |
| `cache_config` | `null` | `CacheConfig` | CPP23_NAMED_OVERRIDE |
| `cache_implementation` | `null` | `CacheImplementation` | CPP23_NAMED_OVERRIDE |
| `constraints` | `null` | `Constraints` | CPP23_NAMED_OVERRIDE |
| `decoder_start_token_id` | `50257` | `DecoderStartTokenId` | CPP23_NAMED_OVERRIDE |
| `disable_compile` | `false` | `DisableCompile` | PINNED_INACTIVE_GENERIC_EXTENSION |
| `diversity_penalty` | `0.0` | `DiversityPenalty` | CPP23_NAMED_OVERRIDE |
| `do_sample` | `false` | `DoSample` | CPP23_NAMED_OVERRIDE |
| `dola_layers` | `null` | `DolaLayers` | CPP23_MODEL_REJECTED |
| `early_stopping` | `false` | `EarlyStopping` | CPP23_NAMED_OVERRIDE |
| `encoder_no_repeat_ngram_size` | `0` | `EncoderNoRepeatNgramSize` | CPP23_MODEL_IGNORED |
| `encoder_repetition_penalty` | `1.0` | `EncoderRepetitionPenalty` | CPP23_MODEL_IGNORED |
| `eos_token_id` | `50256` | `EosTokenId` | CPP23_NAMED_OVERRIDE |
| `epsilon_cutoff` | `0.0` | `EpsilonCutoff` | CPP23_NAMED_OVERRIDE |
| `eta_cutoff` | `0.0` | `EtaCutoff` | CPP23_NAMED_OVERRIDE |
| `exponential_decay_length_penalty` | `null` | `ExponentialDecayLengthPenalty` | CPP23_NAMED_OVERRIDE |
| `force_words_ids` | `null` | `ForceWordsIds` | CPP23_NAMED_OVERRIDE |
| `forced_bos_token_id` | `null` | `ForcedBosTokenId` | CPP23_NAMED_OVERRIDE |
| `forced_decoder_ids` | `[[1,50362]]` | `ForcedDecoderIds` | CPP23_NAMED_OVERRIDE |
| `forced_eos_token_id` | `null` | `ForcedEosTokenId` | CPP23_NAMED_OVERRIDE |
| `guidance_scale` | `null` | `GuidanceScale` | CPP23_MODEL_REJECTED |
| `is_assistant` | `false` | `IsAssistant` | CPP23_PARTIAL_OVERRIDE |
| `is_multilingual` | `false` | `IsMultilingual` | PINNED_INACTIVE_GENERIC_EXTENSION |
| `length_penalty` | `1.0` | `LengthPenalty` | CPP23_NAMED_OVERRIDE |
| `low_memory` | `null` | `LowMemory` | CPP23_NAMED_OVERRIDE |
| `max_initial_timestamp_index` | `50` | `MaxInitialTimestampIndex` | CPP23_NAMED_OVERRIDE |
| `max_length` | `448` | `MaxLength` | CPP23_NAMED_OVERRIDE |
| `max_matching_ngram_size` | `null` | `MaxMatchingNgramSize` | CPP23_NAMED_OVERRIDE |
| `max_new_tokens` | `null` | `MaxNewTokens` | CPP23_NAMED_OVERRIDE |
| `max_time` | `null` | `MaxTime` | CPP23_NAMED_OVERRIDE |
| `min_length` | `0` | `MinLength` | CPP23_NAMED_OVERRIDE |
| `min_new_tokens` | `null` | `MinNewTokens` | CPP23_NAMED_OVERRIDE |
| `min_p` | `null` | `MinP` | CPP23_NAMED_OVERRIDE |
| `no_repeat_ngram_size` | `0` | `NoRepeatNgramSize` | CPP23_NAMED_OVERRIDE |
| `no_timestamps_token_id` | `50362` | `NoTimestampsTokenId` | CPP23_NAMED_OVERRIDE |
| `num_assistant_tokens` | `20` | `NumAssistantTokens` | CPP23_PARTIAL_OVERRIDE |
| `num_assistant_tokens_schedule` | `"constant"` | `NumAssistantTokensSchedule` | CPP23_PARTIAL_OVERRIDE |
| `num_beam_groups` | `1` | `NumBeamGroups` | CPP23_NAMED_OVERRIDE |
| `num_beams` | `1` | `NumBeams` | CPP23_NAMED_OVERRIDE |
| `num_return_sequences` | `1` | `NumReturnSequences` | CPP23_NAMED_OVERRIDE |
| `output_attentions` | `false` | `OutputAttentions` | CPP23_NAMED_OVERRIDE |
| `output_hidden_states` | `false` | `OutputHiddenStates` | CPP23_NAMED_OVERRIDE |
| `output_logits` | `null` | `OutputLogits` | CPP23_NAMED_OVERRIDE |
| `output_scores` | `false` | `OutputScores` | CPP23_NAMED_OVERRIDE |
| `pad_token_id` | `50256` | `PadTokenId` | CPP23_NAMED_OVERRIDE |
| `penalty_alpha` | `null` | `PenaltyAlpha` | CPP23_NAMED_OVERRIDE |
| `prefill_chunk_size` | `null` | `PrefillChunkSize` | CPP23_NAMED_OVERRIDE |
| `prev_sot_token_id` | `50360` | `PrevSotTokenId` | CPP23_NAMED_OVERRIDE |
| `prompt_lookup_num_tokens` | `null` | `PromptLookupNumTokens` | CPP23_NAMED_OVERRIDE |
| `remove_invalid_values` | `false` | `RemoveInvalidValues` | CPP23_NAMED_OVERRIDE |
| `renormalize_logits` | `false` | `RenormalizeLogits` | CPP23_NAMED_OVERRIDE |
| `repetition_penalty` | `1.0` | `RepetitionPenalty` | CPP23_NAMED_OVERRIDE |
| `return_dict_in_generate` | `false` | `ReturnDictInGenerate` | CPP23_NAMED_OVERRIDE |
| `return_legacy_cache` | `null` | `ReturnLegacyCache` | CPP23_NAMED_OVERRIDE |
| `return_timestamps` | `false` | `ReturnTimestamps` | CPP23_NAMED_OVERRIDE |
| `sequence_bias` | `null` | `SequenceBias` | CPP23_NAMED_OVERRIDE |
| `stop_strings` | `null` | `StopStrings` | CPP23_NAMED_OVERRIDE |
| `suppress_tokens` | `[1,2,7,8,9,10,14,25,26,27,28,29,31,58,59,60,61,62,63,90,91,92,93,357,…` | `SuppressTokens` | CPP23_NAMED_OVERRIDE |
| `target_lookbehind` | `10` | `TargetLookbehind` | PINNED_INACTIVE_GENERIC_EXTENSION |
| `temperature` | `1.0` | `Temperature` | CPP23_NAMED_OVERRIDE |
| `token_healing` | `false` | `TokenHealing` | PINNED_INACTIVE_GENERIC_EXTENSION |
| `top_k` | `50` | `TopK` | CPP23_NAMED_OVERRIDE |
| `top_p` | `1.0` | `TopP` | CPP23_NAMED_OVERRIDE |
| `transformers_version` | `"4.57.3"` | `TransformersVersion` | METADATA_FIXED |
| `typical_p` | `1.0` | `TypicalP` | CPP23_NAMED_OVERRIDE |
| `use_cache` | `true` | `UseCache` | CPP23_NAMED_OVERRIDE |
| `watermarking_config` | `null` | `WatermarkingConfig` | CPP23_NAMED_OVERRIDE |

## Generic explicit extensions carried through Whisper `kwargs`

| parameter | pinned default | pinned status | override status |
|---|---|---|---|
| `assistant_model` | `null` | DISABLED_IN_PINNED_MODEL | CPP23_PARTIAL_EXTERNAL_ASSISTANT |
| `streamer` | `null` | DISABLED_IN_PINNED_MODEL | EXTERNAL_GENERATION_ALGORITHM |
| `negative_prompt_ids` | `null` | DISABLED_IN_PINNED_MODEL | EXTERNAL_GENERATION_ALGORITHM |
| `negative_prompt_attention_mask` | `null` | DISABLED_IN_PINNED_MODEL | EXTERNAL_GENERATION_ALGORITHM |
| `use_model_defaults` | `null` | DISABLED_IN_PINNED_MODEL | EXTERNAL_GENERATION_ALGORITHM |
| `custom_generate` | `null` | DISABLED_IN_PINNED_MODEL | EXTERNAL_GENERATION_ALGORITHM |

Pinned-value closure and full reconfiguration closure are separate claims. The former is proved structurally by the generated C++ table; the latter is false and remains future work outside the checkpoint-defined graph.
