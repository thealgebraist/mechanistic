#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace whisper_interface {

template <std::size_t Rows, std::size_t Columns> struct MatrixRef {
  std::span<const float> values;
  static constexpr std::size_t rows = Rows, columns = Columns;
  bool valid() const { return values.size() == Rows * Columns; }
};

template <std::size_t MaximumLength, std::size_t Cardinality>
struct BoundedIndexSequence {
  std::vector<std::int32_t> values;
  bool valid() const {
    if (values.empty() || values.size() > MaximumLength)
      return false;
    for (auto value : values)
      if (value < 0 || value >= static_cast<std::int32_t>(Cardinality))
        return false;
    return true;
  }
};

struct InputFeatures {
  MatrixRef<80, 3000> mel;
};

// A batch is not represented by smuggling a leading dimension into MatrixRef.
// Each item has its own dependent frontend/encoder/decoder state, while the
// checkpoint and token vocabulary are immutable values shared by the executor.
struct Pcm16WavInput {
  std::string path;
  bool valid() const { return !path.empty(); }
};
struct NonEmptyAudioBatch {
  std::vector<Pcm16WavInput> items;
  bool valid() const {
    if (items.empty())
      return false;
    for (const auto &item : items)
      if (!item.valid())
        return false;
    return true;
  }
};
struct SharedWeightsIsolatedItemState {
  bool valid() const { return true; }
};
struct ShortFormGreedyBatchRequest {
  NonEmptyAudioBatch audio;
  SharedWeightsIsolatedItemState execution;
  bool valid() const { return audio.valid() && execution.valid(); }
};
struct SuppliedEncoderMemory {
  MatrixRef<1500, 384> hidden;
};
using EncoderInput = std::variant<InputFeatures, SuppliedEncoderMemory>;

struct TokenIds {
  BoundedIndexSequence<448, 51864> tokens;
};
struct SuppliedDecoderEmbeddings {
  std::span<const float> values;
  std::size_t positions;
  bool valid() const {
    return positions > 0 && positions <= 448 &&
           values.size() == positions * 384;
  }
};
using DecoderInput = std::variant<TokenIds, SuppliedDecoderEmbeddings>;

struct DerivedPositionIds {};
struct SuppliedPositionIds {
  BoundedIndexSequence<448, 448> positions;
};
using PositionInput = std::variant<DerivedPositionIds, SuppliedPositionIds>;

struct NoCache {};
struct InternalIncrementalCache {};
struct SuppliedKeyValueCache {
  std::span<const float> serialized;
  std::size_t position;
  bool valid() const {
    return position <= 448 &&
           serialized.size() == 4 * (2 * position * 384 + 2 * 1500 * 384);
  }
};
using CacheMode =
    std::variant<NoCache, InternalIncrementalCache, SuppliedKeyValueCache>;

// `return_legacy_cache` is a public-output projection. It does not change the
// decoder's internal cache recurrence or any tensor value.
struct EncoderDecoderCacheObject {
  bool valid() const { return true; }
};
struct LegacyFourTuplePerLayer {
  bool valid() const { return true; }
};
using GenerationCacheRepresentation =
    std::variant<EncoderDecoderCacheObject, LegacyFourTuplePerLayer>;
struct GenerationCacheProjection {
  GenerationCacheRepresentation representation;
  std::span<const float> serialized;
  std::size_t position;
  bool valid() const {
    return position <= 448 &&
           serialized.size() == 4 * (2 * position * 384 + 2 * 1500 * 384) &&
           std::visit([](const auto &value) { return value.valid(); },
                      representation);
  }
};

// Time is an execution-state input, not a model tensor. The decision uses the
// pinned stopping criterion's strict `elapsed > maximum` boundary and is
// evaluated only after a token transition has been selected.
struct UnlimitedGenerationTime {
  bool valid() const { return true; }
};
struct FiniteMonotonicDeadline {
  double maximum_seconds;
  bool valid() const { return std::isfinite(maximum_seconds); }
};
using GenerationDeadline =
    std::variant<UnlimitedGenerationTime, FiniteMonotonicDeadline>;
struct ContinueAtOrBeforeDeadline {
  double elapsed_seconds;
  double maximum_seconds;
  bool valid() const {
    return std::isfinite(elapsed_seconds) && elapsed_seconds >= 0.0 &&
           std::isfinite(maximum_seconds) &&
           elapsed_seconds <= maximum_seconds;
  }
};
struct StopAfterDeadline {
  double elapsed_seconds;
  double maximum_seconds;
  bool valid() const {
    return std::isfinite(elapsed_seconds) && elapsed_seconds >= 0.0 &&
           std::isfinite(maximum_seconds) && elapsed_seconds > maximum_seconds;
  }
};
using DeadlineTransition =
    std::variant<ContinueAtOrBeforeDeadline, StopAfterDeadline>;
inline DeadlineTransition deadline_transition(
    const FiniteMonotonicDeadline &deadline, double elapsed_seconds) {
  if (!deadline.valid() || !std::isfinite(elapsed_seconds) ||
      elapsed_seconds < 0.0)
    throw std::invalid_argument("generation deadline domain");
  if (elapsed_seconds > deadline.maximum_seconds)
    return StopAfterDeadline{elapsed_seconds, deadline.maximum_seconds};
  return ContinueAtOrBeforeDeadline{elapsed_seconds,
                                    deadline.maximum_seconds};
}

struct NoAttentionMask {};
struct EncoderAttentionMask {
  BoundedIndexSequence<3000, 2> keep;
};
struct DecoderAttentionMask {
  BoundedIndexSequence<448, 2> keep;
};
using AttentionMask =
    std::variant<NoAttentionMask, EncoderAttentionMask, DecoderAttentionMask>;

struct AllHeads {};
struct EncoderHeadMask {
  std::span<const float> layer_head_mass;
  bool valid() const { return layer_head_mass.size() == 24; }
};
struct DecoderHeadMask {
  std::span<const float> layer_head_mass;
  bool valid() const { return layer_head_mass.size() == 24; }
};
struct CrossAttentionHeadMask {
  std::span<const float> layer_head_mass;
  bool valid() const { return layer_head_mass.size() == 24; }
};
using HeadSelection = std::variant<AllHeads, EncoderHeadMask, DecoderHeadMask,
                                   CrossAttentionHeadMask>;
struct HeadMasks {
  EncoderHeadMask encoder;
  DecoderHeadMask decoder;
  CrossAttentionHeadMask cross;
  bool valid() const {
    return encoder.valid() && decoder.valid() && cross.valid();
  }
};

struct EvalLogits {};
struct EvalWithAttentions {};
struct EvalWithHiddenStates {};
struct LabelledCrossEntropy {
  std::vector<std::int32_t> labels;
  bool valid() const {
    if (labels.empty() || labels.size() > 448)
      return false;
    for (auto value : labels)
      if (value != -100 && (value < 0 || value >= 51864))
        return false;
    return true;
  }
};
using Objective = std::variant<EvalLogits, EvalWithAttentions,
                               EvalWithHiddenStates, LabelledCrossEntropy>;

struct Greedy {};
struct CategoricalSample {
  std::uint64_t seed;
  float temperature;
};
struct HeuristicBeamStopping {
  bool valid() const { return true; }
};
struct StopWhenAllBeamsFinished {
  bool valid() const { return true; }
};
struct CanonicalBeamStopping {
  bool valid() const { return true; }
};
using BeamStoppingPolicy =
    std::variant<HeuristicBeamStopping, StopWhenAllBeamsFinished,
                 CanonicalBeamStopping>;
struct StandardBeamSearch {
  std::size_t beams;
  std::size_t return_sequences;
  float length_penalty;
  BeamStoppingPolicy stopping;
  bool valid() const {
    return beams > 1 && return_sequences > 0 && return_sequences <= beams &&
           length_penalty == length_penalty &&
           std::visit([](const auto &value) { return value.valid(); },
                      stopping);
  }
};
struct DiverseGroupBeamSearch {
  std::size_t beams;
  std::size_t groups;
  std::size_t return_sequences;
  float diversity_penalty;
  float length_penalty;
  BeamStoppingPolicy stopping;
  bool valid() const {
    return beams > 1 && groups > 1 && groups <= beams && beams % groups == 0 &&
           return_sequences > 0 && return_sequences <= beams &&
           diversity_penalty > 0.0f && length_penalty == length_penalty &&
           std::visit([](const auto &value) { return value.valid(); },
                      stopping);
  }
};
struct ForcedPhrase {
  std::vector<std::int32_t> tokens;
  bool valid() const {
    if (tokens.empty() || tokens.size() > 448)
      return false;
    for (auto token : tokens)
      if (token < 0 || token >= 51864)
        return false;
    return true;
  }
};
struct ForcedDisjunction {
  std::vector<ForcedPhrase> alternatives;
  bool valid() const {
    if (alternatives.empty())
      return false;
    for (const auto &alternative : alternatives)
      if (!alternative.valid())
        return false;
    for (std::size_t left = 0; left < alternatives.size(); ++left)
      for (std::size_t right = 0; right < alternatives.size(); ++right)
        if (left != right &&
            alternatives[left].tokens.size() <=
                alternatives[right].tokens.size() &&
            std::equal(alternatives[left].tokens.begin(),
                       alternatives[left].tokens.end(),
                       alternatives[right].tokens.begin()))
          return false;
    return true;
  }
};
using PositiveConstraint = std::variant<ForcedPhrase, ForcedDisjunction>;
struct ConstrainedBeamSearch {
  std::size_t beams;
  std::size_t return_sequences;
  float length_penalty;
  BeamStoppingPolicy stopping;
  std::vector<PositiveConstraint> constraints;
  bool valid() const {
    if (beams <= 1 || return_sequences == 0 || return_sequences > beams ||
        length_penalty != length_penalty || constraints.empty() ||
        !std::visit([](const auto &value) { return value.valid(); }, stopping))
      return false;
    for (const auto &constraint : constraints)
      if (!std::visit([](const auto &value) { return value.valid(); },
                      constraint))
        return false;
    return true;
  }
};
struct UnrestrictedVocabulary {};
struct PrefixAllowedTokensFn {
  std::function<bool(std::size_t, std::span<const std::int32_t>, std::int32_t)>
      allows;
  bool valid() const { return static_cast<bool>(allows); }
};
using VocabularyConstraint =
    std::variant<UnrestrictedVocabulary, PrefixAllowedTokensFn>;

struct NoRepetitionPenalty {
  bool valid() const { return true; }
};
struct RepetitionPenalty {
  float factor;
  bool valid() const { return factor > 0.0f; }
};
using RepetitionPolicy = std::variant<NoRepetitionPenalty, RepetitionPenalty>;
struct AllowRepeatedNGrams {
  bool valid() const { return true; }
};
struct NoRepeatNGram {
  std::size_t order;
  bool valid() const { return order > 0 && order <= 448; }
};
using NGramPolicy = std::variant<AllowRepeatedNGrams, NoRepeatNGram>;
struct AllowAllTokenSequences {
  bool valid() const { return true; }
};
struct ForbiddenTokenSequences {
  std::vector<std::vector<std::int32_t>> sequences;
  bool valid() const {
    if (sequences.empty())
      return false;
    for (const auto &sequence : sequences) {
      if (sequence.empty() || sequence.size() > 448)
        return false;
      for (auto token : sequence)
        if (token < 0 || token >= 51864)
          return false;
    }
    return true;
  }
};
using ForbiddenSequencePolicy =
    std::variant<AllowAllTokenSequences, ForbiddenTokenSequences>;
struct NoMinimumLength {
  bool valid() const { return true; }
};
struct MinimumLength {
  std::size_t positions;
  bool valid() const { return positions > 0 && positions <= 448; }
};
using MinimumLengthPolicy = std::variant<NoMinimumLength, MinimumLength>;
struct NoMinimumNewTokens {
  bool valid() const { return true; }
};
struct MinimumNewTokens {
  std::size_t count;
  bool valid() const { return count > 0 && count <= 448; }
};
using MinimumNewTokenPolicy =
    std::variant<NoMinimumNewTokens, MinimumNewTokens>;
struct NoTopK {
  bool valid() const { return true; }
};
struct TopK {
  std::size_t count;
  bool valid() const { return count > 0 && count <= 51864; }
};
using TopKPolicy = std::variant<NoTopK, TopK>;
struct NoTopP {
  bool valid() const { return true; }
};
struct TopP {
  float mass;
  bool valid() const { return mass >= 0.0f && mass <= 1.0f; }
};
using TopPPolicy = std::variant<NoTopP, TopP>;
struct NoMinP {
  bool valid() const { return true; }
};
struct MinP {
  float fraction_of_maximum;
  bool valid() const {
    return fraction_of_maximum >= 0.0f && fraction_of_maximum <= 1.0f;
  }
};
using MinPPolicy = std::variant<NoMinP, MinP>;
struct NoTypicalP {
  bool valid() const { return true; }
};
struct TypicalP {
  float mass;
  bool valid() const { return mass > 0.0f && mass < 1.0f; }
};
using TypicalPPolicy = std::variant<NoTypicalP, TypicalP>;
struct NoEpsilonCutoff {
  bool valid() const { return true; }
};
struct EpsilonCutoff {
  float probability;
  bool valid() const { return probability > 0.0f && probability < 1.0f; }
};
using EpsilonPolicy = std::variant<NoEpsilonCutoff, EpsilonCutoff>;
struct NoEtaCutoff {
  bool valid() const { return true; }
};
struct EtaCutoff {
  float epsilon;
  bool valid() const { return epsilon > 0.0f && epsilon < 1.0f; }
};
using EtaPolicy = std::variant<NoEtaCutoff, EtaCutoff>;
struct SamplingFilters {
  TopKPolicy top_k;
  TopPPolicy top_p;
  MinPPolicy min_p;
  TypicalPPolicy typical_p;
  EpsilonPolicy epsilon;
  EtaPolicy eta;
  bool valid() const {
    return std::visit([](const auto &value) { return value.valid(); }, top_k) &&
           std::visit([](const auto &value) { return value.valid(); }, top_p) &&
           std::visit([](const auto &value) { return value.valid(); }, min_p) &&
           std::visit([](const auto &value) { return value.valid(); },
                      typical_p) &&
           std::visit([](const auto &value) { return value.valid(); },
                      epsilon) &&
           std::visit([](const auto &value) { return value.valid(); }, eta);
  }
};
struct SampledBeamSearch {
  std::size_t beams;
  std::size_t return_sequences;
  float temperature;
  std::uint64_t seed;
  float length_penalty;
  BeamStoppingPolicy stopping;
  SamplingFilters sampling;
  bool valid() const {
    return beams > 1 && return_sequences > 0 && return_sequences <= beams &&
           temperature > 0.0f && length_penalty == length_penalty &&
           std::visit([](const auto &value) { return value.valid(); },
                      stopping) &&
           sampling.valid();
  }
};
struct SequentialCandidateEvaluation {
  bool valid() const { return true; }
};
struct ContrastiveSearch {
  std::size_t candidates;
  float degeneration_penalty;
  SequentialCandidateEvaluation execution;
  bool valid() const {
    return candidates > 1 && candidates <= 51864 &&
           degeneration_penalty > 0.0f &&
           std::isfinite(degeneration_penalty) && execution.valid();
  }
};
struct DolaLowLayers {
  bool valid() const { return true; }
};
struct DolaHighLayers {
  bool valid() const { return true; }
};
struct DolaExplicitLayers {
  std::vector<std::size_t> layers;
  bool valid() const { return !layers.empty(); }
};
using DolaLayerRequest =
    std::variant<DolaLowLayers, DolaHighLayers, DolaExplicitLayers>;
struct RejectDolaForEncoderDecoder {
  DolaLayerRequest requested;
  bool valid() const {
    return std::visit([](const auto &value) { return value.valid(); },
                      requested);
  }
};
struct RejectUnbatchedGuidanceForMelEncoder {
  float scale;
  bool valid() const {
    return std::isfinite(scale) && scale != 1.0f;
  }
};
struct IgnoreEncoderTokenPenaltyWithoutEncoderTokenIds {
  float repetition_factor;
  std::size_t no_repeat_ngram_size;
  bool valid() const {
    return std::isfinite(repetition_factor) && repetition_factor > 0.0f &&
           (repetition_factor != 1.0f || no_repeat_ngram_size > 0);
  }
};
struct IgnoreBosTokenInWhisperCustomInitialization {
  std::int32_t token;
  bool valid() const { return token >= 0 && token < 51864; }
};
struct StopStringSet {
  std::vector<std::string> values;
  bool valid() const {
    if (values.empty())
      return false;
    for (const auto &value : values)
      if (value.empty())
        return false;
    return true;
  }
};
struct PromptLookupSearch {
  std::size_t proposal_tokens;
  std::size_t maximum_matching_ngram;
  bool valid() const {
    return proposal_tokens > 0 && proposal_tokens <= 448 &&
           maximum_matching_ngram > 0 && maximum_matching_ngram < 448;
  }
};
using Selection =
    std::variant<Greedy, CategoricalSample, StandardBeamSearch,
                 DiverseGroupBeamSearch, ConstrainedBeamSearch,
                 SampledBeamSearch, ContrastiveSearch, PromptLookupSearch>;
struct NoSequenceBias {
  bool valid() const { return true; }
};
struct BiasedTokenSequence {
  std::vector<std::int32_t> tokens;
  float additive_logit_bias;
  bool valid() const {
    if (tokens.empty() || tokens.size() > 448 ||
        additive_logit_bias != additive_logit_bias)
      return false;
    for (auto token : tokens)
      if (token < 0 || token >= 51864)
        return false;
    return true;
  }
};
struct AdditiveSequenceBias {
  std::vector<BiasedTokenSequence> entries;
  bool valid() const {
    if (entries.empty())
      return false;
    for (const auto &entry : entries)
      if (!entry.valid())
        return false;
    return true;
  }
};
using SequenceBiasPolicy = std::variant<NoSequenceBias, AdditiveSequenceBias>;
struct NoForcedBeginningToken {
  bool valid() const { return true; }
};
struct ForcedBeginningToken {
  std::int32_t token;
  bool valid() const { return token >= 0 && token < 51864; }
};
using ForcedBeginningPolicy =
    std::variant<NoForcedBeginningToken, ForcedBeginningToken>;
struct NoForcedEndingTokens {
  bool valid() const { return true; }
};
struct ForcedEndingTokens {
  std::vector<std::int32_t> tokens;
  bool valid() const {
    if (tokens.empty())
      return false;
    for (auto token : tokens)
      if (token < 0 || token >= 51864)
        return false;
    return true;
  }
};
using ForcedEndingPolicy =
    std::variant<NoForcedEndingTokens, ForcedEndingTokens>;
struct PreserveInvalidLogits {
  bool valid() const { return true; }
};
struct RepairInvalidLogits {
  bool valid() const { return true; }
};
using InvalidLogitPolicy =
    std::variant<PreserveInvalidLogits, RepairInvalidLogits>;
struct NoExponentialEosDecay {
  bool valid() const { return true; }
};
struct ExponentialEosDecay {
  std::size_t start_after_new_tokens;
  float factor;
  bool valid() const { return factor > 0.0f; }
};
using ExponentialEosPolicy =
    std::variant<NoExponentialEosDecay, ExponentialEosDecay>;
struct PreserveLogitScale {
  bool valid() const { return true; }
};
struct NormalizeLogProbabilities {
  bool valid() const { return true; }
};
using LogitNormalizationPolicy =
    std::variant<PreserveLogitScale, NormalizeLogProbabilities>;
struct NoWatermark {
  bool valid() const { return true; }
};
struct LeftHashWatermark {
  double greenlist_ratio;
  float additive_bias;
  std::int64_t hashing_key;
  std::size_t context_width;
  bool valid() const {
    return greenlist_ratio > 0.0 && greenlist_ratio < 1.0 &&
           std::isfinite(greenlist_ratio) && std::isfinite(additive_bias) &&
           context_width > 0 && context_width <= 448;
  }
};
struct SelfHashWatermark {
  double greenlist_ratio;
  float additive_bias;
  std::int64_t hashing_key;
  std::size_t context_width;
  bool valid() const {
    return greenlist_ratio > 0.0 && greenlist_ratio < 1.0 &&
           std::isfinite(greenlist_ratio) && std::isfinite(additive_bias) &&
           context_width > 0 && context_width <= 448;
  }
};
struct SynthIDTextWatermark {
  std::size_t ngram_length;
  std::vector<std::int64_t> keys;
  std::size_t context_history_size;
  std::int64_t sampling_table_seed;
  std::size_t sampling_table_size;
  bool skip_first_ngram_calls;
  bool debug_uniform_scores;
  bool valid() const {
    return ngram_length > 0 && ngram_length <= 448 && !keys.empty() &&
           sampling_table_size > 0 && sampling_table_size <= (1u << 24);
  }
};
using WatermarkPolicy =
    std::variant<NoWatermark, LeftHashWatermark, SelfHashWatermark,
                 SynthIDTextWatermark>;
struct GenerationLogitPolicies {
  RepetitionPolicy repetition;
  NGramPolicy ngrams;
  ForbiddenSequencePolicy forbidden;
  MinimumLengthPolicy minimum_length;
  MinimumNewTokenPolicy minimum_new_tokens;
  SamplingFilters sampling{NoTopK{},     NoTopP{},          NoMinP{},
                           NoTypicalP{}, NoEpsilonCutoff{}, NoEtaCutoff{}};
  SequenceBiasPolicy sequence_bias{NoSequenceBias{}};
  ForcedBeginningPolicy forced_beginning{NoForcedBeginningToken{}};
  ForcedEndingPolicy forced_ending{NoForcedEndingTokens{}};
  InvalidLogitPolicy invalid_logits{PreserveInvalidLogits{}};
  ExponentialEosPolicy exponential_eos{NoExponentialEosDecay{}};
  WatermarkPolicy watermark{NoWatermark{}};
  LogitNormalizationPolicy normalization{PreserveLogitScale{}};
  bool valid() const {
    return std::visit([](const auto &value) { return value.valid(); },
                      repetition) &&
           std::visit([](const auto &value) { return value.valid(); },
                      ngrams) &&
           std::visit([](const auto &value) { return value.valid(); },
                      forbidden) &&
           std::visit([](const auto &value) { return value.valid(); },
                      minimum_length) &&
           std::visit([](const auto &value) { return value.valid(); },
                      minimum_new_tokens) &&
           sampling.valid() &&
           std::visit([](const auto &value) { return value.valid(); },
                      sequence_bias) &&
           std::visit([](const auto &value) { return value.valid(); },
                      forced_beginning) &&
           std::visit([](const auto &value) { return value.valid(); },
                      forced_ending) &&
           std::visit([](const auto &value) { return value.valid(); },
                      invalid_logits) &&
           std::visit([](const auto &value) { return value.valid(); },
                      exponential_eos) &&
           std::visit([](const auto &value) { return value.valid(); },
                      watermark) &&
           std::visit([](const auto &value) { return value.valid(); },
                      normalization);
  }
};

struct NoTimestamps {};
struct TimestampTokens {
  float seconds_per_token;
  bool valid() const { return seconds_per_token > 0.0f; }
};
struct TokenTimestamps {
  float seconds_per_token;
  bool valid() const { return seconds_per_token > 0.0f; }
};
struct Segments {
  float seconds_per_timestamp_token;
  float seconds_per_feature_frame;
  bool valid() const {
    return seconds_per_timestamp_token > 0.0f &&
           seconds_per_feature_frame > 0.0f;
  }
};
using TimeOutput =
    std::variant<NoTimestamps, TimestampTokens, TokenTimestamps, Segments>;

struct NoPrompt {};
struct PromptTokens {
  BoundedIndexSequence<448, 51864> tokens;
};
struct PreviousSegmentTokens {
  BoundedIndexSequence<448, 51864> tokens;
};
using PromptCondition =
    std::variant<NoPrompt, PromptTokens, PreviousSegmentTokens>;
struct FirstSegmentPrompt {};
struct AllSegmentsPrompt {};
using PromptConditionType = std::variant<FirstSegmentPrompt, AllSegmentsPrompt>;

struct ContiguousGenerationAttentionMask {
  std::size_t valid_frames;
  std::size_t total_frames;
  bool valid() const {
    return valid_frames > 0 && valid_frames <= total_frames;
  }
};
struct SegmentWindowFrames {
  std::size_t value;
  bool valid() const { return value == 3000; }
};
struct ShortFormWindow {};
struct LongFormWindowing {
  ContiguousGenerationAttentionMask attention;
  SegmentWindowFrames segment;
  bool condition_on_previous;
  bool valid() const {
    return attention.valid() && segment.valid() &&
           attention.total_frames > segment.value;
  }
};
using GenerationWindowing = std::variant<ShortFormWindow, LongFormWindowing>;
struct NoProgressMonitor {};
struct MonitorProgress {
  std::function<void(std::size_t, std::size_t)> notify;
  bool valid() const { return static_cast<bool>(notify); }
};
using ProgressMonitor = std::variant<NoProgressMonitor, MonitorProgress>;
struct NoFallback {};
struct FallbackThresholds {
  std::optional<float> compression_ratio;
  std::optional<float> average_logprob;
  std::optional<float> no_speech_probability;
  std::vector<float> temperatures;
  bool valid() const {
    if (temperatures.empty())
      return false;
    for (auto value : temperatures)
      if (value < 0.0f)
        return false;
    return !no_speech_probability || average_logprob.has_value();
  }
};
using FallbackPolicy = std::variant<NoFallback, FallbackThresholds>;
struct GenerationExtensionInventory {
  std::size_t generation_config_fields;
  std::size_t generic_extensions;
  std::size_t forward_kwargs;
  bool valid() const {
    return generation_config_fields == 74 && generic_extensions == 6 &&
           forward_kwargs == 17;
  }
};

struct MaximumModelPositions {
  bool valid() const { return true; }
};
struct MaximumTotalPositions {
  std::size_t count;
  bool valid() const { return count > 0 && count <= 448; }
};
struct MaximumNewTokens {
  std::size_t count;
  bool valid() const { return count > 0 && count <= 448; }
};
using GenerationLengthLimit =
    std::variant<MaximumModelPositions, MaximumTotalPositions,
                 MaximumNewTokens>;

struct ForwardRequest {
  EncoderInput encoder;
  DecoderInput decoder;
  PositionInput positions;
  CacheMode cache;
  AttentionMask attention;
  HeadMasks heads;
  Objective objective;
};
struct GenerationRequest {
  InputFeatures input;
  Selection selection;
  VocabularyConstraint vocabulary;
  GenerationLogitPolicies logit_policies;
  TimeOutput time_output;
  PromptCondition prompt;
  PromptConditionType prompt_scope;
  GenerationWindowing windowing;
  ProgressMonitor progress;
  FallbackPolicy fallback;
  GenerationExtensionInventory extensions;
  GenerationLengthLimit length;
};

} // namespace whisper_interface
