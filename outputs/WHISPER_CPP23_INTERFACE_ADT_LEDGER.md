# Whisper PyTorch interface → C++23 ADT ledger

This ledger is derived from the installed `WhisperForConditionalGeneration.forward` and `.generate` signatures. All 17 forward parameters and 27 generation parameters are classified. There are **0 pending rows**; they are explicit remaining work rather than being counted as converted.

The C++23 constructors live in `whisper_interface_adt.hpp`. Dimension-bearing constructors include `MatrixRef<80,3000>`, `MatrixRef<1500,384>`, and `BoundedIndexSequence<448,51864>`.

## Forward

| parameter | status | ADT constructor | evidence or gap |
|---|---|---|---|
| `input_features` | EXECUTABLE_GRAPH | `EncoderInput.InputFeatures` | nodes 1-29 |
| `attention_mask` | EXECUTABLE_MODEL_NOOP | `AttentionMask.EncoderAttentionMask` | current WhisperEncoder accepts but does not consume this argument |
| `decoder_input_ids` | EXECUTABLE_GRAPH | `DecoderInput.TokenIds` | nodes 30-69; arbitrary valid IDs verified |
| `decoder_attention_mask` | EXECUTABLE_GRAPH | `AttentionMask.DecoderAttentionMask` | arbitrary binary key mask combined with causal mask and verified |
| `head_mask` | EXECUTABLE_GRAPH_EAGER | `HeadSelection.EncoderHeadMask` | 24 head masses verified against eager PyTorch attention |
| `decoder_head_mask` | EXECUTABLE_GRAPH_EAGER | `HeadSelection.DecoderHeadMask` | 24 head masses verified against eager PyTorch attention |
| `cross_attn_head_mask` | EXECUTABLE_GRAPH_EAGER | `HeadSelection.CrossAttentionHeadMask` | 24 head masses verified against eager PyTorch attention |
| `encoder_outputs` | EXECUTABLE_GRAPH | `EncoderInput.SuppliedEncoderMemory` | typed MatrixRef<1500,384> decoder-only graph entry verified |
| `past_key_values` | EXECUTABLE_GRAPH | `CacheMode.SuppliedKeyValueCache` | typed four-layer self/cross cache import, step, and export verified |
| `decoder_inputs_embeds` | EXECUTABLE_GRAPH | `DecoderInput.SuppliedDecoderEmbeddings` | typed bounded embedding-matrix graph entry verified |
| `decoder_position_ids` | EXECUTABLE_GRAPH | `PositionInput.SuppliedPositionIds` | arbitrary bounded position lookup verified |
| `labels` | EXECUTABLE_GRAPH | `Objective.LabelledCrossEntropy` | decoder right-shift, -100 ignore rule, and mean cross-entropy verified |
| `use_cache` | EXECUTABLE_GRAPH | `CacheMode.NoCache|InternalIncrementalCache` | both full causal and incremental graph paths verified |
| `output_attentions` | EXECUTABLE_GRAPH_EAGER | `Objective.EvalWithAttentions` | four encoder, four decoder, and four cross-attention tensors exported and verified |
| `output_hidden_states` | EXECUTABLE_GRAPH | `Objective.EvalWithHiddenStates` | complete five encoder plus five decoder hidden-state tuples exported and verified |
| `return_dict` | ABI_PROJECTION | `ForwardRequest` | C++ returns typed values rather than Python ModelOutput |
| `cache_position` | EXECUTABLE_GRAPH | `CacheMode.InternalIncrementalCache` | explicit imported position drives positional lookup and cache append |

## Generation

| parameter | status | ADT constructor | evidence or gap |
|---|---|---|---|
| `input_features` | EXECUTABLE_GRAPH | `GenerationRequest.input` | nodes 1-29 |
| `generation_config` | PINNED_EXECUTABLE_PLUS_POLICIES | `GenerationRequest` | pinned English policy plus typed cache-allocation/rejection behavior, sequence bias, repetition, no-repeat n-gram, forbidden-sequence, boundary/decay/length, repair/normalization, six sampling filters, contrastive hidden-state ranking, greedy stop-string and prompt-lookup speculation, common-vocabulary external-assistant product state, standard/sampled/diverse-group beam search, and phrase/disjunction-constrained beam search |
| `logits_processor` | EXECUTABLE_NAMED_SUBSET | `GenerationLogitPolicies` | forced/suppressed tokens, additive sequence bias, repetition and n-gram policies, EOS decay, invalid repair, normalization, and six sampling filters; arbitrary user processor classes remain external |
| `stopping_criteria` | EXECUTABLE_BUILTIN_SUBSET | `GenerationLengthLimit` | EOS, typed max_length, typed max_new_tokens, and the 448-position model bound; arbitrary user callbacks remain external |
| `prefix_allowed_tokens_fn` | EXECUTABLE_GRAPH_CALLBACK | `VocabularyConstraint.PrefixAllowedTokensFn` | step/stack/token predicate masks the policy node and an altered first-token constraint exactly matches Transformers |
| `synced_gpus` | SINGLE_PROCESS_NOOP | `GenerationRequest` | C++ runtime is single-process |
| `return_timestamps` | EXECUTABLE_GRAPH | `TimeOutput.TimestampTokens` | timestamp-pair, monotonicity, initial bound, and aggregate-mass policy exactly verified |
| `task` | PINNED_EXECUTABLE_SUBSET | `GenerationRequest` | English transcription task |
| `language` | PINNED_EXECUTABLE_SUBSET | `GenerationRequest` | tiny.en English only |
| `is_multilingual` | PINNED_EXECUTABLE_SUBSET | `GenerationRequest` | false for tiny.en |
| `prompt_ids` | EXECUTABLE_GRAPH | `PromptCondition.PromptTokens` | prompt prefix before decoder start/no-timestamps prefix exactly matches Transformers tokens |
| `prompt_condition_type` | EXECUTABLE_GRAPH_LONGFORM | `PromptConditionType` | first-segment and all-segments history placement exactly verified with a concrete prompt |
| `condition_on_prev_tokens` | EXECUTABLE_GRAPH_LONGFORM | `PromptCondition.PreviousSegmentTokens` | 223-token cutoff, previous-start token, double-timestamp elision, and changed second-window output exactly verified |
| `temperature` | EXECUTABLE_GRAPH | `Selection.CategoricalSample` | positive temperature scales logits before six typed sampling filters; complete first-step categorical distributions verified |
| `compression_ratio_threshold` | EXECUTABLE_GRAPH_FALLBACK | `FallbackPolicy.FallbackThresholds` | zlib token compression ratio forces and records two attempts per long-form window |
| `logprob_threshold` | EXECUTABLE_GRAPH_FALLBACK | `FallbackPolicy.FallbackThresholds` | processed-policy average selected logprob forces and records two attempts per window |
| `no_speech_threshold` | EXECUTABLE_GRAPH_FALLBACK | `FallbackPolicy.FallbackThresholds` | raw no-speech probability combined with average logprob exactly skips the final silent window |
| `num_segment_frames` | EXECUTABLE_MODEL_PINNED | `GenerationWindowing.LongFormWindowing` | installed implementation overwrites this argument with 3000; two-window 3513-frame execution verified |
| `attention_mask` | EXECUTABLE_CONTIGUOUS_AUDIO_MASK | `GenerationWindowing.ContiguousGenerationAttentionMask` | raw-audio-derived contiguous valid extent drives max frames, seeks, and DTW crop |
| `time_precision` | PINNED_EXECUTABLE_SUBSET | `TimeOutput.TimestampTokens` | pinned 0.02 seconds per timestamp token is represented and verified |
| `time_precision_features` | PINNED_EXECUTABLE_SUBSET | `TimeOutput.Segments` | pinned 0.01 seconds per feature frame is represented for segment arithmetic |
| `return_token_timestamps` | EXECUTABLE_GRAPH_DTW | `TimeOutput.TokenTimestamps` | eight configured cross-attention heads, normalization, reflected median filter, head mean, and DTW exactly verified |
| `return_segments` | EXECUTABLE_GRAPH_SHORTFORM | `TimeOutput.Segments` | timestamp-delimited short-form segment tokens and boundaries exactly verified |
| `return_dict_in_generate` | ABI_PROJECTION | `GenerationRequest` | C++ GenerationResult replaces Python GenerateOutput |
| `force_unique_generate_call` | SINGLE_PROCESS_NOOP | `GenerationRequest` | one graph invocation per request |
| `monitor_progress` | EXECUTABLE_GRAPH_CALLBACK | `ProgressMonitor.MonitorProgress` | typed seek/max callback emits the exact two PyTorch long-form progress states |
| `kwargs` | VERSION_PINNED_CLOSED_INVENTORY | `GenerationExtensionInventory` | 74 GenerationConfig fields, 6 generic extensions, and 17 forward kwargs are named; unknown keys are rejected |

There are no unclassified top-level signature rows for the pinned model. This is narrower than full generic reconfiguration: common-vocabulary deterministic external assistance is partially executable, while different-tokenizer/sampled/early-exit assistant variants and other external hooks remain explicit in `WHISPER_CPP23_GENERATION_EXTENSION_INVENTORY.md`. Model-inapplicable DoLa/guidance paths and ignored encoder-token processors have explicit typed classifications.
