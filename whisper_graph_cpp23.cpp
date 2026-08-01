#include "generated_whisper_generation_config.hpp"
#include "generated_whisper_graph.hpp"
#include "whisper_interface_adt.hpp"
#if defined(WHISPER_PORTABLE_BACKEND)
#include "portable_backend.hpp"
#else
#include <Accelerate/Accelerate.h>
#endif
#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <type_traits>
#include <variant>
#include <vector>
#include <zlib.h>

namespace whisper_graph {
using F32 = std::vector<float>;
static std::uint64_t peak_rss_bytes() {
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0)
    throw std::runtime_error("getrusage failed");
  return static_cast<std::uint64_t>(usage.ru_maxrss);
}
enum class Stage { Frontend, Encoder, Decoder, Readout, Transition };
enum class Op {
  PcmAudioInput,
  LogMelStft,
  Conv1Gelu,
  Conv2Gelu,
  PositionalAdd,
  LayerNorm,
  SelfAttention,
  ResidualAdd,
  MlpGelu,
  TokenStackInput,
  TokenPositionEmbed,
  CachedSelfAttention,
  CrossAttention,
  TiedLmHead,
  GenerationPolicy,
  Softmax,
  SampleOrArgmax,
  TokenAndCacheAppend
};
struct FrontendState {
  std::span<const float> mel;
};
struct EncoderState {
  F32 hidden;
  std::size_t positions = 0;
};
struct DecoderState {
  std::vector<std::int32_t> tokens;
  std::array<F32, 8> key_value_cache;
};
struct CategoricalMass {
  F32 probability;
};
using Register =
    std::variant<FrontendState, EncoderState, DecoderState, CategoricalMass>;
struct TensorDesc {
  std::string name, dtype, shape;
  std::uint64_t begin = 0, end = 0, elements = 0;
  std::uint32_t crc = 0;
};

static std::vector<std::string> split(const std::string &s, char delim = '\t') {
  std::vector<std::string> v;
  std::stringstream q(s);
  std::string x;
  while (std::getline(q, x, delim))
    v.push_back(x);
  return v;
}
static std::uint32_t crc32(std::span<const unsigned char> d) {
  static const auto table = []() {
    std::array<std::uint32_t, 256> t{};
    for (std::uint32_t b = 0; b < 256; ++b) {
      auto c = b;
      for (int i = 0; i < 8; ++i)
        c = (c & 1) ? 0xedb88320U ^ (c >> 1) : c >> 1;
      t[b] = c;
    }
    return t;
  }();
  std::uint32_t c = 0xffffffffU;
  for (auto b : d)
    c = table[(c ^ b) & 255] ^ (c >> 8);
  return c ^ 0xffffffffU;
}

class TensorStore {
  std::ifstream checkpoint_;
  std::map<std::string, TensorDesc> desc_;
  std::map<std::string, F32> cache_;

public:
  TensorStore(const std::string &checkpoint, const std::string &manifest)
      : checkpoint_(checkpoint, std::ios::binary) {
    if (!checkpoint_)
      throw std::runtime_error("checkpoint open");
    std::ifstream in(manifest);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      auto f = split(line);
      if (f.size() != 7)
        throw std::runtime_error("bad tensor manifest row");
      std::stringstream hs(f[6]);
      std::uint32_t crc;
      hs >> std::hex >> crc;
      desc_.emplace(f[0],
                    TensorDesc{f[0], f[1], f[4], std::stoull(f[2]),
                               std::stoull(f[3]), std::stoull(f[5]), crc});
    }
    if (desc_.size() != 167)
      throw std::runtime_error("tensor count");
  }
  const F32 &get(const std::string &name) {
    if (auto i = cache_.find(name); i != cache_.end())
      return i->second;
    auto found = desc_.find(name);
    if (found == desc_.end())
      throw std::runtime_error("missing tensor " + name);
    auto d = found->second;
    if (d.dtype != "F32" || d.end - d.begin != 4 * d.elements)
      throw std::runtime_error("tensor metadata " + name);
    checkpoint_.seekg(d.begin);
    std::vector<unsigned char> raw(d.end - d.begin);
    checkpoint_.read(reinterpret_cast<char *>(raw.data()),
                     static_cast<std::streamsize>(raw.size()));
    if (!checkpoint_ || crc32(raw) != d.crc)
      throw std::runtime_error("tensor bytes " + name);
    F32 values(d.elements);
    for (std::size_t i = 0; i < values.size(); ++i) {
      std::uint32_t u = std::uint32_t(raw[4 * i]) |
                        (std::uint32_t(raw[4 * i + 1]) << 8) |
                        (std::uint32_t(raw[4 * i + 2]) << 16) |
                        (std::uint32_t(raw[4 * i + 3]) << 24);
      values[i] = std::bit_cast<float>(u);
      if (!std::isfinite(values[i]))
        throw std::runtime_error("nonfinite " + name);
    }
    return cache_.emplace(name, std::move(values)).first->second;
  }
  std::size_t count() const { return desc_.size(); }
  bool contains(std::string_view name) const {
    return desc_.contains(std::string(name));
  }
  std::size_t validate_all() {
    std::size_t count = 0;
    for (const auto &[name, d] : desc_) {
      checkpoint_.seekg(d.begin);
      std::vector<unsigned char> raw(d.end - d.begin);
      checkpoint_.read(reinterpret_cast<char *>(raw.data()),
                       static_cast<std::streamsize>(raw.size()));
      if (!checkpoint_ || crc32(raw) != d.crc)
        throw std::runtime_error("tensor validation " + name);
      ++count;
    }
    return count;
  }
};

static F32 read_f32(const std::string &path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in)
    throw std::runtime_error("open " + path);
  auto n = static_cast<std::size_t>(in.tellg());
  if (n % 4)
    throw std::runtime_error("unaligned " + path);
  in.seekg(0);
  F32 x(n / 4);
  in.read(reinterpret_cast<char *>(x.data()), static_cast<std::streamsize>(n));
  if (!in)
    throw std::runtime_error("read " + path);
  return x;
}
static std::vector<std::int32_t> read_i32(const std::string &path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in)
    throw std::runtime_error("open " + path);
  auto n = static_cast<std::size_t>(in.tellg());
  if (n % 4)
    throw std::runtime_error("unaligned " + path);
  in.seekg(0);
  std::vector<std::int32_t> x(n / 4);
  in.read(reinterpret_cast<char *>(x.data()), static_cast<std::streamsize>(n));
  if (!in)
    throw std::runtime_error("read " + path);
  return x;
}
static void write_f32(const std::string &path, std::span<const float> values) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size_bytes()));
  if (!out)
    throw std::runtime_error("write " + path);
}
static void write_u8(const std::string &path,
                     std::span<const std::uint8_t> values) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size_bytes()));
  if (!out)
    throw std::runtime_error("write " + path);
}
static void
write_named_tensors(const std::string &binary_path,
                    const std::string &manifest_path,
                    const std::vector<std::pair<std::string, F32>> &values,
                    const std::vector<std::string> &selected) {
  std::ofstream binary(binary_path, std::ios::binary);
  std::ofstream manifest(manifest_path);
  if (!binary || !manifest)
    throw std::runtime_error("open named tensor output");
  manifest << "name\tbegin_element\telements\n";
  std::size_t offset = 0;
  for (const auto &name : selected) {
    auto found =
        std::find_if(values.begin(), values.end(),
                     [&](const auto &entry) { return entry.first == name; });
    if (found == values.end())
      throw std::runtime_error("missing trace tensor " + name);
    manifest << name << '\t' << offset << '\t' << found->second.size() << '\n';
    binary.write(
        reinterpret_cast<const char *>(found->second.data()),
        static_cast<std::streamsize>(found->second.size() * sizeof(float)));
    offset += found->second.size();
  }
  if (!binary || !manifest)
    throw std::runtime_error("write named tensor output");
}
static std::vector<std::int32_t> parse_token_ids(const std::string &csv) {
  std::vector<std::int32_t> ids;
  for (const auto &field : split(csv, ',')) {
    if (field.empty())
      throw std::runtime_error("empty token id");
    std::size_t used = 0;
    long value = std::stol(field, &used);
    if (used != field.size() || value < 0 || value >= 51864)
      throw std::runtime_error("invalid token id");
    ids.push_back(static_cast<std::int32_t>(value));
  }
  if (ids.empty() || ids.size() > 448)
    throw std::runtime_error("token sequence length");
  return ids;
}
static F32 parse_f32_csv(const std::string &csv) {
  F32 values;
  for (const auto &field : split(csv, ',')) {
    if (field.empty())
      throw std::runtime_error("empty float");
    std::size_t used = 0;
    float value = std::stof(field, &used);
    if (used != field.size() || !std::isfinite(value))
      throw std::runtime_error("invalid float");
    values.push_back(value);
  }
  return values;
}
static std::optional<float> parse_optional_f32(const std::string &text) {
  if (text == "-")
    return std::nullopt;
  std::size_t used = 0;
  float value = std::stof(text, &used);
  if (used != text.size() || !std::isfinite(value))
    throw std::runtime_error("invalid optional float");
  return value;
}
static double parse_f64(const std::string &text) {
  std::size_t used = 0;
  const double value = std::stod(text, &used);
  if (used != text.size() || !std::isfinite(value))
    throw std::runtime_error("invalid double");
  return value;
}
static std::int64_t parse_i64(const std::string &text) {
  std::size_t used = 0;
  const auto value = std::stoll(text, &used);
  if (used != text.size())
    throw std::runtime_error("invalid signed integer");
  return value;
}
static std::vector<std::int64_t> parse_i64_csv(const std::string &text) {
  std::vector<std::int64_t> values;
  if (text == "-")
    return values;
  for (const auto &field : split(text, ','))
    values.push_back(parse_i64(field));
  return values;
}
static std::size_t parse_nonnegative_size(const std::string &text,
                                          std::size_t maximum) {
  std::size_t used = 0;
  const auto value = std::stoull(text, &used);
  if (used != text.size() || value > maximum)
    throw std::runtime_error("invalid nonnegative size");
  return static_cast<std::size_t>(value);
}
static bool parse_bool01(const std::string &text);
static std::optional<std::size_t>
parse_optional_size(const std::string &text, std::size_t maximum = 448) {
  if (text == "-")
    return std::nullopt;
  std::size_t used = 0;
  auto value = std::stoull(text, &used);
  if (used != text.size() || value == 0 || value > maximum)
    throw std::runtime_error("invalid optional size");
  return static_cast<std::size_t>(value);
}
static whisper_interface::WatermarkPolicy
parse_watermark_policy(const std::string &scheme, const std::string &ratio_text,
                       const std::string &bias_text,
                       const std::string &key_text,
                       const std::string &context_text) {
  const auto ratio = parse_f64(ratio_text);
  const auto bias = parse_optional_f32(bias_text);
  const auto key = parse_i64(key_text);
  const auto context = parse_optional_size(context_text);
  if (!bias || !context)
    throw std::runtime_error("watermark arguments");
  whisper_interface::WatermarkPolicy policy;
  if (scheme == "lefthash")
    policy = whisper_interface::LeftHashWatermark{ratio, *bias, key, *context};
  else if (scheme == "selfhash")
    policy = whisper_interface::SelfHashWatermark{ratio, *bias, key, *context};
  else
    throw std::runtime_error("watermark seeding scheme");
  if (!std::visit([](const auto &value) { return value.valid(); }, policy))
    throw std::runtime_error("watermark policy ADT invariant");
  return policy;
}
static whisper_interface::WatermarkPolicy parse_synthid_watermark_policy(
    const std::string &ngram_text, const std::string &keys_text,
    const std::string &history_text, const std::string &seed_text,
    const std::string &table_size_text, const std::string &skip_text,
    const std::string &debug_text) {
  const auto ngram = parse_optional_size(ngram_text);
  const auto table_size = parse_optional_size(table_size_text, 1u << 24);
  if (!ngram || !table_size)
    throw std::runtime_error("SynthID positive-size arguments");
  whisper_interface::SynthIDTextWatermark policy{
      *ngram,
      parse_i64_csv(keys_text),
      parse_nonnegative_size(history_text, 1u << 24),
      parse_i64(seed_text),
      *table_size,
      parse_bool01(skip_text),
      parse_bool01(debug_text)};
  if (!policy.valid())
    throw std::runtime_error("SynthID watermark ADT invariant");
  return policy;
}
static whisper_interface::ForbiddenSequencePolicy
parse_forbidden_sequences(const std::string &text) {
  if (text == "-")
    return whisper_interface::AllowAllTokenSequences{};
  whisper_interface::ForbiddenTokenSequences policy;
  for (const auto &sequence : split(text, ';'))
    policy.sequences.push_back(parse_token_ids(sequence));
  if (!policy.valid())
    throw std::runtime_error("forbidden sequence ADT invariant");
  return policy;
}
static std::vector<whisper_interface::PositiveConstraint>
parse_positive_constraints(const std::string &text) {
  std::vector<whisper_interface::PositiveConstraint> constraints;
  for (const auto &encoded : split(text, ';')) {
    if (encoded.starts_with("p:")) {
      whisper_interface::ForcedPhrase phrase{
          parse_token_ids(encoded.substr(2))};
      if (!phrase.valid())
        throw std::runtime_error("forced phrase ADT invariant");
      constraints.push_back(std::move(phrase));
    } else if (encoded.starts_with("d:")) {
      whisper_interface::ForcedDisjunction disjunction;
      for (const auto &alternative : split(encoded.substr(2), '|'))
        disjunction.alternatives.push_back(
            {parse_token_ids(alternative)});
      if (!disjunction.valid())
        throw std::runtime_error("forced disjunction ADT invariant");
      constraints.push_back(std::move(disjunction));
    } else
      throw std::runtime_error("positive constraint syntax");
  }
  if (constraints.empty())
    throw std::runtime_error("empty positive constraints");
  return constraints;
}
static whisper_interface::SamplingFilters parse_sampling_filters(
    const std::string &top_k_text, const std::string &top_p_text,
    const std::string &min_p_text, const std::string &typical_p_text,
    const std::string &epsilon_text, const std::string &eta_text) {
  const auto top_k = parse_optional_size(top_k_text, 51864);
  const auto top_p = parse_optional_f32(top_p_text);
  const auto min_p = parse_optional_f32(min_p_text);
  const auto typical_p = parse_optional_f32(typical_p_text);
  const auto epsilon = parse_optional_f32(epsilon_text);
  const auto eta = parse_optional_f32(eta_text);
  whisper_interface::SamplingFilters filters{
      top_k ? whisper_interface::TopKPolicy{whisper_interface::TopK{*top_k}}
            : whisper_interface::TopKPolicy{whisper_interface::NoTopK{}},
      top_p ? whisper_interface::TopPPolicy{whisper_interface::TopP{*top_p}}
            : whisper_interface::TopPPolicy{whisper_interface::NoTopP{}},
      min_p ? whisper_interface::MinPPolicy{whisper_interface::MinP{*min_p}}
            : whisper_interface::MinPPolicy{whisper_interface::NoMinP{}},
      typical_p
          ? whisper_interface::TypicalPPolicy{whisper_interface::TypicalP{
                *typical_p}}
          : whisper_interface::TypicalPPolicy{whisper_interface::NoTypicalP{}},
      epsilon
          ? whisper_interface::EpsilonPolicy{whisper_interface::EpsilonCutoff{
                *epsilon}}
          : whisper_interface::
                EpsilonPolicy{whisper_interface::NoEpsilonCutoff{}},
      eta ? whisper_interface::EtaPolicy{whisper_interface::EtaCutoff{*eta}}
          : whisper_interface::EtaPolicy{whisper_interface::NoEtaCutoff{}}};
  if (!filters.valid())
    throw std::runtime_error("sampling filter ADT invariant");
  return filters;
}
static bool parse_bool01(const std::string &text) {
  if (text == "0")
    return false;
  if (text == "1")
    return true;
  throw std::runtime_error("expected boolean 0 or 1");
}
static whisper_interface::DolaLayerRequest
parse_dola_layers(const std::string &text) {
  if (text == "low")
    return whisper_interface::DolaLowLayers{};
  if (text == "high")
    return whisper_interface::DolaHighLayers{};
  whisper_interface::DolaExplicitLayers explicit_layers;
  for (const auto &part : split(text, ',')) {
    std::size_t used = 0;
    const auto layer = std::stoull(part, &used);
    if (used != part.size())
      throw std::runtime_error("DoLa layer index");
    explicit_layers.layers.push_back(layer);
  }
  if (!explicit_layers.valid())
    throw std::runtime_error("DoLa layer request");
  return explicit_layers;
}
static whisper_interface::SequenceBiasPolicy
parse_sequence_bias(const std::string &text) {
  if (text == "-")
    return whisper_interface::NoSequenceBias{};
  whisper_interface::AdditiveSequenceBias policy;
  for (const auto &encoded : split(text, ';')) {
    const auto separator = encoded.rfind(':');
    if (separator == std::string::npos)
      throw std::runtime_error("sequence bias syntax");
    auto tokens = parse_token_ids(encoded.substr(0, separator));
    auto bias = parse_optional_f32(encoded.substr(separator + 1));
    if (!bias)
      throw std::runtime_error("sequence bias value");
    policy.entries.push_back({std::move(tokens), *bias});
  }
  if (!policy.valid())
    throw std::runtime_error("sequence bias ADT invariant");
  return policy;
}
static whisper_interface::ForcedBeginningPolicy
parse_forced_beginning(const std::string &text) {
  if (text == "-")
    return whisper_interface::NoForcedBeginningToken{};
  const auto tokens = parse_token_ids(text);
  if (tokens.size() != 1)
    throw std::runtime_error("forced beginning token count");
  return whisper_interface::ForcedBeginningToken{tokens.front()};
}
static whisper_interface::ForcedEndingPolicy
parse_forced_ending(const std::string &text) {
  if (text == "-")
    return whisper_interface::NoForcedEndingTokens{};
  whisper_interface::ForcedEndingTokens policy{parse_token_ids(text)};
  if (!policy.valid())
    throw std::runtime_error("forced ending ADT invariant");
  return policy;
}
static whisper_interface::ExponentialEosPolicy
parse_exponential_eos(const std::string &text) {
  if (text == "-")
    return whisper_interface::NoExponentialEosDecay{};
  const auto fields = split(text, ',');
  if (fields.size() != 2)
    throw std::runtime_error("exponential EOS syntax");
  std::size_t used = 0;
  const auto start = std::stoull(fields[0], &used);
  const auto factor = parse_optional_f32(fields[1]);
  if (used != fields[0].size() || start > 448 || !factor)
    throw std::runtime_error("exponential EOS values");
  whisper_interface::ExponentialEosDecay policy{static_cast<std::size_t>(start),
                                                *factor};
  if (!policy.valid())
    throw std::runtime_error("exponential EOS ADT invariant");
  return policy;
}
static whisper_interface::LabelledCrossEntropy
parse_labels(const std::string &csv) {
  whisper_interface::LabelledCrossEntropy objective;
  for (const auto &field : split(csv, ',')) {
    if (field.empty())
      throw std::runtime_error("empty label");
    std::size_t used = 0;
    long value = std::stol(field, &used);
    if (used != field.size())
      throw std::runtime_error("invalid label syntax");
    objective.labels.push_back(static_cast<std::int32_t>(value));
  }
  if (!objective.valid())
    throw std::runtime_error("label ADT invariant");
  return objective;
}
static std::vector<std::int32_t>
shift_labels_right(const whisper_interface::LabelledCrossEntropy &objective) {
  std::vector<std::int32_t> ids(objective.labels.size(), 50256);
  ids[0] = 50257;
  for (std::size_t i = 1; i < ids.size(); ++i)
    ids[i] = objective.labels[i - 1] == -100 ? 50256 : objective.labels[i - 1];
  return ids;
}
static double labelled_cross_entropy(
    std::span<const float> logits,
    const whisper_interface::LabelledCrossEntropy &objective) {
  constexpr std::size_t vocabulary = 51864;
  if (logits.size() != objective.labels.size() * vocabulary)
    throw std::runtime_error("labelled logits shape");
  double total = 0;
  std::size_t count = 0;
  for (std::size_t row = 0; row < objective.labels.size(); ++row) {
    auto label = objective.labels[row];
    if (label == -100)
      continue;
    auto begin = logits.begin() + static_cast<std::ptrdiff_t>(row * vocabulary);
    auto end = begin + vocabulary;
    float maximum = *std::max_element(begin, end);
    double sum = 0;
    for (auto i = begin; i != end; ++i)
      sum += std::exp(double(*i - maximum));
    total += double(maximum) + std::log(sum) - begin[label];
    ++count;
  }
  if (count == 0)
    throw std::runtime_error("all labels ignored");
  return total / count;
}
static std::uint32_t u32le(const unsigned char *p) {
  return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
         (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
}
static F32 read_wav_pcm16(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  std::array<unsigned char, 12> head{};
  in.read(reinterpret_cast<char *>(head.data()), head.size());
  if (!in || std::string(reinterpret_cast<char *>(head.data()), 4) != "RIFF" ||
      std::string(reinterpret_cast<char *>(head.data() + 8), 4) != "WAVE")
    throw std::runtime_error("wav header");
  std::uint16_t format = 0, channels = 0, bits = 0;
  std::uint32_t rate = 0;
  std::vector<unsigned char> data;
  while (in) {
    std::array<unsigned char, 8> chunk{};
    in.read(reinterpret_cast<char *>(chunk.data()), 8);
    if (!in)
      break;
    auto size = u32le(chunk.data() + 4);
    std::string tag(reinterpret_cast<char *>(chunk.data()), 4);
    std::vector<unsigned char> payload(size);
    in.read(reinterpret_cast<char *>(payload.data()), size);
    if (size & 1)
      in.get();
    if (tag == "fmt " && size >= 16) {
      format = std::uint16_t(payload[0] | payload[1] << 8);
      channels = std::uint16_t(payload[2] | payload[3] << 8);
      rate = u32le(payload.data() + 4);
      bits = std::uint16_t(payload[14] | payload[15] << 8);
    } else if (tag == "data")
      data = std::move(payload);
  }
  if (format != 1 || channels != 1 || rate != 16000 || bits != 16 ||
      data.empty() || data.size() % 2)
    throw std::runtime_error("wav PCM contract");
  F32 pcm(data.size() / 2);
  for (std::size_t i = 0; i < pcm.size(); ++i) {
    std::uint16_t u =
        std::uint16_t(data[2 * i]) | (std::uint16_t(data[2 * i + 1]) << 8);
    pcm[i] = std::int16_t(u) / 32768.0f;
  }
  return pcm;
}
static F32 log_mel_waveform(const F32 &waveform, const F32 &window,
                            const F32 &filters, std::size_t used_frames) {
  constexpr std::size_t nfft = 400, hop = 160, bins = 201, mels = 80, pad = 200;
  if (window.size() != nfft || filters.size() != bins * mels ||
      waveform.size() <= pad || used_frames == 0)
    throw std::runtime_error("frontend shape");
  const std::size_t samples = waveform.size(), frames = used_frames + 1;
  F32 padded(samples + 2 * pad);
  for (long i = -long(pad); i < long(samples + pad); ++i) {
    long source = i;
    if (source < 0)
      source = -source;
    if (source >= long(samples))
      source = 2 * long(samples) - 2 - source;
    padded[static_cast<std::size_t>(i + pad)] =
        waveform[static_cast<std::size_t>(source)];
  }
  F32 framed(frames * nfft);
  for (std::size_t t = 0; t < frames; ++t)
    for (std::size_t n = 0; n < nfft; ++n)
      framed[t * nfft + n] = padded[t * hop + n] * window[n];
  F32 cosine(bins * nfft), sine(bins * nfft);
  constexpr double tau = 6.283185307179586476925286766559;
  for (std::size_t k = 0; k < bins; ++k)
    for (std::size_t n = 0; n < nfft; ++n) {
      double phase = tau * k * n / nfft;
      cosine[k * nfft + n] = std::cos(phase);
      sine[k * nfft + n] = -std::sin(phase);
    }
  F32 real(frames * bins), imag(frames * bins);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(frames),
              bins, nfft, 1, framed.data(), nfft, cosine.data(), nfft, 0,
              real.data(), bins);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(frames),
              bins, nfft, 1, framed.data(), nfft, sine.data(), nfft, 0,
              imag.data(), bins);
  F32 power(used_frames * bins);
  for (std::size_t i = 0; i < power.size(); ++i)
    power[i] = real[i] * real[i] + imag[i] * imag[i];
  F32 frame_mel(used_frames * mels);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              static_cast<int>(used_frames), mels, bins, 1, power.data(), bins,
              filters.data(), mels, 0, frame_mel.data(), mels);
  float maximum = -std::numeric_limits<float>::infinity();
  for (auto &x : frame_mel) {
    x = std::log10(std::max(x, 1e-10f));
    maximum = std::max(maximum, x);
  }
  F32 mel(mels * used_frames);
  for (std::size_t t = 0; t < used_frames; ++t)
    for (std::size_t m = 0; m < mels; ++m)
      mel[m * used_frames + t] =
          (std::max(frame_mel[t * mels + m], maximum - 8) + 4) / 4;
  return mel;
}
static F32 log_mel(const F32 &input, const F32 &window, const F32 &filters) {
  constexpr std::size_t samples = 480000, used_frames = 3000;
  F32 waveform(samples);
  std::copy_n(input.begin(), std::min(input.size(), samples), waveform.begin());
  return log_mel_waveform(waveform, window, filters, used_frames);
}
static F32 log_mel_unpadded(const F32 &input, const F32 &window,
                            const F32 &filters) {
  constexpr std::size_t hop = 160;
  const std::size_t used_frames = input.size() / hop;
  return log_mel_waveform(input, window, filters, used_frames);
}
static F32 mel_window(std::span<const float> full, std::size_t total_frames,
                      std::size_t seek, std::size_t window_frames = 3000) {
  constexpr std::size_t mels = 80;
  if (full.size() != mels * total_frames || seek > total_frames ||
      window_frames != 3000)
    throw std::runtime_error("mel window shape");
  F32 window(mels * window_frames);
  const auto copied = std::min(window_frames, total_frames - seek);
  for (std::size_t m = 0; m < mels; ++m)
    std::copy_n(
        full.begin() + static_cast<std::ptrdiff_t>(m * total_frames + seek),
        copied,
        window.begin() + static_cast<std::ptrdiff_t>(m * window_frames));
  return window;
}
static F32 linear(std::span<const float> x, std::size_t rows, std::size_t in,
                  const F32 &w, std::size_t out, const F32 &b = F32{}) {
  if (x.size() != rows * in || w.size() != out * in ||
      (!b.empty() && b.size() != out))
    throw std::runtime_error("linear shape");
  F32 y(rows * out);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(rows),
              static_cast<int>(out), static_cast<int>(in), 1, x.data(),
              static_cast<int>(in), w.data(), static_cast<int>(in), 0, y.data(),
              static_cast<int>(out));
  if (!b.empty())
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t j = 0; j < out; ++j)
        y[r * out + j] += b[j];
  return y;
}
static void gelu(F32 &x) {
  constexpr float q = 0.7071067811865475244f;
  for (auto &v : x)
    v = 0.5f * v * (1.0f + std::erf(v * q));
}
static F32 conv1d(std::span<const float> x, std::size_t in_c, std::size_t in_t,
                  const F32 &w, const F32 &b, std::size_t out_c, int stride,
                  int pad, bool channel_major) {
  constexpr std::size_t kernel = 3;
  std::size_t out_t = (in_t + 2 * pad - kernel) / stride + 1;
  F32 patches(out_t * in_c * kernel);
  for (std::size_t t = 0; t < out_t; ++t)
    for (std::size_t c = 0; c < in_c; ++c)
      for (std::size_t k = 0; k < kernel; ++k) {
        auto source = static_cast<long>(t * stride + k) - pad;
        float v = 0;
        if (source >= 0 && source < static_cast<long>(in_t))
          v = channel_major ? x[c * in_t + source] : x[source * in_c + c];
        patches[t * (in_c * kernel) + c * kernel + k] = v;
      }
  return linear(patches, out_t, in_c * kernel, w, out_c, b);
}
static F32 layer_norm(std::span<const float> x, std::size_t rows, std::size_t d,
                      const F32 &w, const F32 &b, float eps) {
  F32 y(x.size());
  for (std::size_t r = 0; r < rows; ++r) {
    double sum = 0, sq = 0;
    for (std::size_t j = 0; j < d; ++j) {
      double v = x[r * d + j];
      sum += v;
      sq += v * v;
    }
    double mean = sum / d, var = std::max(0.0, sq / d - mean * mean),
           inv = 1 / std::sqrt(var + eps);
    for (std::size_t j = 0; j < d; ++j)
      y[r * d + j] =
          static_cast<float>((x[r * d + j] - mean) * inv) * w[j] + b[j];
  }
  return y;
}
static void add_inplace(F32 &x, const F32 &y) {
  if (x.size() != y.size())
    throw std::runtime_error("add shape");
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] += y[i];
}
static void softmax_rows(F32 &x, std::size_t rows, std::size_t cols) {
  for (std::size_t r = 0; r < rows; ++r) {
    auto first = x.begin() + r * cols, last = first + cols;
    float m = *std::max_element(first, last);
    double sum = 0;
    for (auto i = first; i != last; ++i) {
      *i = std::exp(*i - m);
      sum += *i;
    }
    for (auto i = first; i != last; ++i)
      *i = static_cast<float>(*i / sum);
  }
}

class GraphExecutionAudit {
  std::array<std::uint32_t, generated_whisper::nodes.size()> hits_{};

public:
  void hit(const generated_whisper::Node &node) {
    if (node.index >= hits_.size())
      throw std::runtime_error("graph execution node index");
    ++hits_[node.index];
  }
  std::size_t visited() const {
    return std::count_if(hits_.begin(), hits_.end(),
                         [](auto count) { return count != 0; });
  }
  void require_range(std::size_t begin, std::size_t end) const {
    if (begin > end || end > hits_.size())
      throw std::runtime_error("graph execution range");
    for (std::size_t i = begin; i < end; ++i)
      if (hits_[i] == 0)
        throw std::runtime_error("unexecuted graph node " + std::to_string(i));
  }
  void require_all() const {
    for (std::size_t i = 0; i < hits_.size(); ++i)
      if (hits_[i] == 0)
        throw std::runtime_error("unexecuted graph node " + std::to_string(i));
  }
};

class AttentionWriter {
  std::ofstream binary_, manifest_;
  std::size_t offset_ = 0;

public:
  AttentionWriter(const std::string &binary_path,
                  const std::string &manifest_path)
      : binary_(binary_path, std::ios::binary), manifest_(manifest_path) {
    if (!binary_ || !manifest_)
      throw std::runtime_error("open attention output");
    manifest_ << "name\tbegin_element\telements\tshape\n";
  }
  void write_head(const std::string &name, std::size_t head, std::size_t heads,
                  std::size_t rows, std::size_t columns,
                  std::span<const float> mass) {
    if (mass.size() != rows * columns)
      throw std::runtime_error("attention output shape");
    if (head == 0)
      manifest_ << name << '\t' << offset_ << '\t' << heads * rows * columns
                << '\t' << heads << 'x' << rows << 'x' << columns << '\n';
    binary_.write(reinterpret_cast<const char *>(mass.data()),
                  static_cast<std::streamsize>(mass.size_bytes()));
    offset_ += mass.size();
    if (!binary_ || !manifest_)
      throw std::runtime_error("write attention output");
  }
};

static std::string_view graph_weight_name(const generated_whisper::Node &node,
                                          std::string_view suffix) {
  using namespace generated_whisper;
  std::string_view found;
  for (std::size_t i = 0; i < node.weight_count; ++i) {
    auto name = weight_refs[node.weight_begin + i];
    if (name.ends_with(suffix)) {
      if (!found.empty())
        throw std::runtime_error("ambiguous graph weight suffix " +
                                 std::string(suffix));
      found = name;
    }
  }
  if (found.empty())
    throw std::runtime_error("missing graph weight suffix " +
                             std::string(suffix));
  return found;
}
static const F32 &graph_weight(TensorStore &store,
                               const generated_whisper::Node &node,
                               std::string_view suffix) {
  return store.get(std::string(graph_weight_name(node, suffix)));
}

static F32 execute_frontend(const std::string &wav, const F32 &window,
                            const F32 &filters, GraphExecutionAudit &audit) {
  using namespace generated_whisper;
  F32 pcm, mel;
  for (std::size_t i = 0; i < 2; ++i) {
    const auto &node = nodes[i];
    audit.hit(node);
    switch (node.opcode) {
    case Opcode::PcmAudioInput:
      pcm = read_wav_pcm16(wav);
      break;
    case Opcode::LogMelStft:
      mel = log_mel(pcm, window, filters);
      break;
    default:
      throw std::runtime_error("unexpected frontend opcode");
    }
  }
  return mel;
}

class Encoder {
  TensorStore &t_;
  GraphExecutionAudit &audit_;
  static constexpr std::size_t D = 384, H = 6, HD = 64, FF = 1536, T = 1500;
  F32 attention(const F32 &x, const generated_whisper::Node &node,
                std::span<const float> head_mask, AttentionWriter *writer) {
    if (!head_mask.empty() && head_mask.size() != 4 * H)
      throw std::runtime_error("encoder head mask shape");
    auto q = linear(x, T, D, graph_weight(t_, node, ".q_proj.weight"), D,
                    graph_weight(t_, node, ".q_proj.bias"));
    auto k = linear(x, T, D, graph_weight(t_, node, ".k_proj.weight"), D);
    auto v = linear(x, T, D, graph_weight(t_, node, ".v_proj.weight"), D,
                    graph_weight(t_, node, ".v_proj.bias"));
    float scale = 1 / std::sqrt(float(HD));
    for (auto &z : q)
      z *= scale;
    F32 context(T * D), score(T * T);
    auto layer = (node.index - 6) / 6;
    for (std::size_t h = 0; h < H; ++h) {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, T, HD, 1,
                  q.data() + h * HD, D, k.data() + h * HD, D, 0, score.data(),
                  T);
      softmax_rows(score, T, T);
      if (!head_mask.empty())
        for (auto &value : score)
          value *= head_mask[layer * H + h];
      if (writer)
        writer->write_head("encoder_attention_" + std::to_string(layer), h, H,
                           T, T, score);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, HD, T, 1,
                  score.data(), T, v.data() + h * HD, D, 0,
                  context.data() + h * HD, D);
    }
    return linear(context, T, D, graph_weight(t_, node, ".out_proj.weight"), D,
                  graph_weight(t_, node, ".out_proj.bias"));
  }

public:
  Encoder(TensorStore &t, GraphExecutionAudit &audit) : t_(t), audit_(audit) {}
  F32 run(std::span<const float> mel,
          std::vector<std::pair<std::string, F32>> &trace,
          std::span<const float> head_mask = {},
          AttentionWriter *writer = nullptr) {
    using namespace generated_whisper;
    F32 x, branch, normed;
    for (std::size_t i = 2; i <= 29; ++i) {
      const auto &node = nodes[i];
      audit_.hit(node);
      switch (node.opcode) {
      case Opcode::Conv1Gelu:
        x = conv1d(mel, 80, 3000, graph_weight(t_, node, ".weight"),
                   graph_weight(t_, node, ".bias"), 384, 1, 1, true);
        gelu(x);
        trace.emplace_back("conv1_gelu", x);
        break;
      case Opcode::Conv2Gelu:
        x = conv1d(x, 384, 3000, graph_weight(t_, node, ".weight"),
                   graph_weight(t_, node, ".bias"), 384, 2, 1, false);
        gelu(x);
        break;
      case Opcode::PositionalAdd:
        add_inplace(x, graph_weight(t_, node, ".weight"));
        trace.emplace_back("conv2_gelu_position", x);
        break;
      case Opcode::LayerNorm:
        normed = layer_norm(x, T, D, graph_weight(t_, node, ".weight"),
                            graph_weight(t_, node, ".bias"), 1e-5f);
        if (i == 29) {
          x = normed;
          trace.emplace_back("encoder_final", x);
        }
        break;
      case Opcode::SelfAttention:
        branch = attention(normed, node, head_mask, writer);
        break;
      case Opcode::MlpGelu:
        branch = linear(normed, T, D, graph_weight(t_, node, ".fc1.weight"), FF,
                        graph_weight(t_, node, ".fc1.bias"));
        gelu(branch);
        branch = linear(branch, T, FF, graph_weight(t_, node, ".fc2.weight"), D,
                        graph_weight(t_, node, ".fc2.bias"));
        break;
      case Opcode::ResidualAdd:
        add_inplace(x, branch);
        if (i >= 10 && (i - 10) % 6 == 0)
          trace.emplace_back("encoder_layer_" + std::to_string((i - 10) / 6),
                             x);
        break;
      default:
        throw std::runtime_error("unexpected encoder opcode at node " +
                                 std::to_string(i));
      }
    }
    return x;
  }
};
class Decoder {
  TensorStore &t_;
  GraphExecutionAudit &audit_;
  static constexpr std::size_t D = 384, H = 6, HD = 64, FF = 1536, V = 51864,
                               S = 1500;
  F32 attention(const F32 &query, const F32 &memory, std::size_t rows,
                std::size_t source, const generated_whisper::Node &node,
                bool causal, std::span<const std::int32_t> key_mask = {},
                std::span<const float> head_mask = {},
                AttentionWriter *writer = nullptr) {
    if (!key_mask.empty() && key_mask.size() != source)
      throw std::runtime_error("decoder attention mask shape");
    if (!head_mask.empty() && head_mask.size() != 4 * H)
      throw std::runtime_error("decoder head mask shape");
    auto q = linear(query, rows, D, graph_weight(t_, node, ".q_proj.weight"), D,
                    graph_weight(t_, node, ".q_proj.bias"));
    auto k =
        linear(memory, source, D, graph_weight(t_, node, ".k_proj.weight"), D);
    auto v = linear(memory, source, D, graph_weight(t_, node, ".v_proj.weight"),
                    D, graph_weight(t_, node, ".v_proj.bias"));
    float scale = 1 / std::sqrt(float(HD));
    for (auto &z : q)
      z *= scale;
    F32 context(rows * D), score(rows * source);
    auto layer = node.opcode == generated_whisper::Opcode::CachedSelfAttention
                     ? (node.index - 33) / 9
                     : (node.index - 36) / 9;
    for (std::size_t h = 0; h < H; ++h) {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, source, HD, 1,
                  q.data() + h * HD, D, k.data() + h * HD, D, 0, score.data(),
                  source);
      if (causal || !key_mask.empty())
        for (std::size_t r = 0; r < rows; ++r)
          for (std::size_t c = 0; c < source; ++c)
            if ((causal && c > r) || (!key_mask.empty() && key_mask[c] == 0))
              score[r * source + c] = -std::numeric_limits<float>::infinity();
      softmax_rows(score, rows, source);
      if (!head_mask.empty())
        for (auto &value : score)
          value *= head_mask[layer * H + h];
      if (writer)
        writer->write_head((causal ? std::string("decoder_attention_")
                                   : std::string("cross_attention_")) +
                               std::to_string(layer),
                           h, H, rows, source, score);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, HD, source,
                  1, score.data(), source, v.data() + h * HD, D, 0,
                  context.data() + h * HD, D);
    }
    return linear(context, rows, D, graph_weight(t_, node, ".out_proj.weight"),
                  D, graph_weight(t_, node, ".out_proj.bias"));
  }
  F32 run_graph(const std::vector<std::int32_t> *ids,
                std::span<const float> supplied_embeddings, std::size_t rows,
                const F32 &memory, std::span<const std::int32_t> decoder_mask,
                std::span<const std::int32_t> position_ids,
                std::span<const float> decoder_head_mask,
                std::span<const float> cross_head_mask, AttentionWriter *writer,
                std::vector<std::pair<std::string, F32>> &trace) {
    using namespace generated_whisper;
    if (rows == 0 || rows > 448 ||
        (!ids && supplied_embeddings.size() != rows * D) ||
        (!decoder_mask.empty() && decoder_mask.size() != rows) ||
        (!position_ids.empty() && position_ids.size() != rows))
      throw std::runtime_error("decoder input shape");
    F32 x, branch, normed, logits;
    for (std::size_t i = 30; i <= 69; ++i) {
      const auto &node = nodes[i];
      audit_.hit(node);
      switch (node.opcode) {
      case Opcode::TokenStackInput:
        break;
      case Opcode::TokenPositionEmbed: {
        const auto &position = graph_weight(t_, node, "embed_positions.weight");
        x.resize(rows * D);
        if (ids) {
          const auto &embedding = graph_weight(t_, node, "embed_tokens.weight");
          for (std::size_t r = 0; r < rows; ++r) {
            if ((*ids)[r] < 0 || (*ids)[r] >= static_cast<std::int32_t>(V))
              throw std::runtime_error("token id");
            auto position_id = position_ids.empty()
                                   ? r
                                   : static_cast<std::size_t>(position_ids[r]);
            if (position_id >= 448)
              throw std::runtime_error("decoder position id");
            for (std::size_t j = 0; j < D; ++j)
              x[r * D + j] = embedding[std::size_t((*ids)[r]) * D + j] +
                             position[position_id * D + j];
          }
        } else
          for (std::size_t r = 0; r < rows; ++r) {
            auto position_id = position_ids.empty()
                                   ? r
                                   : static_cast<std::size_t>(position_ids[r]);
            if (position_id >= 448)
              throw std::runtime_error("decoder position id");
            for (std::size_t j = 0; j < D; ++j)
              x[r * D + j] = supplied_embeddings[r * D + j] +
                             position[position_id * D + j];
          }
        trace.emplace_back("decoder_embed_position", x);
        break;
      }
      case Opcode::LayerNorm:
        normed = layer_norm(x, rows, D, graph_weight(t_, node, ".weight"),
                            graph_weight(t_, node, ".bias"), 1e-5f);
        if (i == 68) {
          x = normed;
          trace.emplace_back("decoder_final", x);
        }
        break;
      case Opcode::CachedSelfAttention:
        branch = attention(normed, normed, rows, rows, node, true, decoder_mask,
                           decoder_head_mask, writer);
        break;
      case Opcode::CrossAttention:
        branch = attention(normed, memory, rows, S, node, false, {},
                           cross_head_mask, writer);
        break;
      case Opcode::MlpGelu:
        branch = linear(normed, rows, D, graph_weight(t_, node, ".fc1.weight"),
                        FF, graph_weight(t_, node, ".fc1.bias"));
        gelu(branch);
        branch = linear(branch, rows, FF, graph_weight(t_, node, ".fc2.weight"),
                        D, graph_weight(t_, node, ".fc2.bias"));
        break;
      case Opcode::ResidualAdd:
        add_inplace(x, branch);
        if (i >= 40 && (i - 40) % 9 == 0)
          trace.emplace_back("decoder_layer_" + std::to_string((i - 40) / 9),
                             x);
        break;
      case Opcode::TiedLmHead:
        logits = linear(x, rows, D, graph_weight(t_, node, ".weight"), V);
        trace.emplace_back("logits", logits);
        break;
      default:
        throw std::runtime_error("unexpected decoder opcode at node " +
                                 std::to_string(i));
      }
    }
    return logits;
  }

public:
  Decoder(TensorStore &t, GraphExecutionAudit &audit) : t_(t), audit_(audit) {}
  F32 run(const std::vector<std::int32_t> &ids, const F32 &memory,
          std::vector<std::pair<std::string, F32>> &trace) {
    return run_graph(&ids, {}, ids.size(), memory, {}, {}, {}, {}, nullptr,
                     trace);
  }
  F32 run_masked(const std::vector<std::int32_t> &ids, const F32 &memory,
                 std::span<const std::int32_t> decoder_mask,
                 std::span<const std::int32_t> position_ids,
                 std::vector<std::pair<std::string, F32>> &trace) {
    return run_graph(&ids, {}, ids.size(), memory, decoder_mask, position_ids,
                     {}, {}, nullptr, trace);
  }
  F32 run_head_masked(const std::vector<std::int32_t> &ids, const F32 &memory,
                      std::span<const float> decoder_head_mask,
                      std::span<const float> cross_head_mask,
                      std::vector<std::pair<std::string, F32>> &trace) {
    return run_graph(&ids, {}, ids.size(), memory, {}, {}, decoder_head_mask,
                     cross_head_mask, nullptr, trace);
  }
  F32 run_attentions(const std::vector<std::int32_t> &ids, const F32 &memory,
                     AttentionWriter &writer,
                     std::vector<std::pair<std::string, F32>> &trace) {
    return run_graph(&ids, {}, ids.size(), memory, {}, {}, {}, {}, &writer,
                     trace);
  }
  F32 run_embeddings(std::span<const float> embeddings, std::size_t rows,
                     const F32 &memory,
                     std::vector<std::pair<std::string, F32>> &trace) {
    return run_graph(nullptr, embeddings, rows, memory, {}, {}, {}, {}, nullptr,
                     trace);
  }
};
struct LayerKV {
  F32 self_key, self_value, cross_key, cross_value;
};
struct DecoderKVState {
  std::array<LayerKV, 4> layers;
  std::size_t position = 0;
};
struct AlignmentTrace {
  static constexpr std::size_t heads = 8, source_positions = 1500;
  F32 weights;
  std::size_t positions = 0, rows_in_position = 0;
  static bool selected(std::size_t layer, std::size_t head) {
    return (layer == 1 && head == 0) ||
           (layer == 2 && (head == 0 || head == 5)) ||
           (layer == 3 && head <= 4);
  }
  void append(std::size_t layer, std::size_t head,
              std::span<const float> mass) {
    if (!selected(layer, head))
      return;
    if (mass.size() != source_positions)
      throw std::runtime_error("alignment source shape");
    weights.insert(weights.end(), mass.begin(), mass.end());
    ++rows_in_position;
  }
  void finish_position() {
    if (rows_in_position != heads)
      throw std::runtime_error("alignment head coverage");
    rows_in_position = 0;
    ++positions;
  }
};
static DecoderKVState deserialize_cache(std::span<const float> serialized,
                                        std::size_t position) {
  whisper_interface::SuppliedKeyValueCache input{serialized, position};
  if (!input.valid())
    throw std::runtime_error("serialized cache ADT invariant");
  DecoderKVState state;
  state.position = position;
  std::size_t offset = 0;
  auto take = [&](F32 &target, std::size_t count) {
    target.assign(serialized.begin() + static_cast<std::ptrdiff_t>(offset),
                  serialized.begin() +
                      static_cast<std::ptrdiff_t>(offset + count));
    offset += count;
  };
  for (auto &layer : state.layers) {
    take(layer.self_key, position * 384);
    take(layer.self_value, position * 384);
    take(layer.cross_key, 1500 * 384);
    take(layer.cross_value, 1500 * 384);
  }
  if (offset != serialized.size())
    throw std::runtime_error("serialized cache trailing data");
  return state;
}
static F32 serialize_cache(const DecoderKVState &state) {
  F32 serialized;
  serialized.reserve(4 * (2 * state.position * 384 + 2 * 1500 * 384));
  for (const auto &layer : state.layers) {
    if (layer.self_key.size() != state.position * 384 ||
        layer.self_value.size() != state.position * 384 ||
        layer.cross_key.size() != 1500 * 384 ||
        layer.cross_value.size() != 1500 * 384)
      throw std::runtime_error("cache serialization shape");
    serialized.insert(serialized.end(), layer.self_key.begin(),
                      layer.self_key.end());
    serialized.insert(serialized.end(), layer.self_value.begin(),
                      layer.self_value.end());
    serialized.insert(serialized.end(), layer.cross_key.begin(),
                      layer.cross_key.end());
    serialized.insert(serialized.end(), layer.cross_value.begin(),
                      layer.cross_value.end());
  }
  return serialized;
}
class CachedDecoder {
  TensorStore &t_;
  GraphExecutionAudit &audit_;
  static constexpr std::size_t D = 384, H = 6, HD = 64, FF = 1536, V = 51864,
                               S = 1500;
  F32 attend_one(const F32 &query, const F32 &memory, LayerKV &cache,
                 const generated_whisper::Node &node, bool cross,
                 std::size_t layer, AlignmentTrace *alignment) {
    auto q = linear(query, 1, D, graph_weight(t_, node, ".q_proj.weight"), D,
                    graph_weight(t_, node, ".q_proj.bias"));
    F32 *keys;
    F32 *values;
    std::size_t source;
    if (cross) {
      if (cache.cross_key.empty()) {
        cache.cross_key =
            linear(memory, S, D, graph_weight(t_, node, ".k_proj.weight"), D);
        cache.cross_value =
            linear(memory, S, D, graph_weight(t_, node, ".v_proj.weight"), D,
                   graph_weight(t_, node, ".v_proj.bias"));
      }
      keys = &cache.cross_key;
      values = &cache.cross_value;
      source = S;
    } else {
      auto k = linear(query, 1, D, graph_weight(t_, node, ".k_proj.weight"), D);
      auto v = linear(query, 1, D, graph_weight(t_, node, ".v_proj.weight"), D,
                      graph_weight(t_, node, ".v_proj.bias"));
      cache.self_key.insert(cache.self_key.end(), k.begin(), k.end());
      cache.self_value.insert(cache.self_value.end(), v.begin(), v.end());
      keys = &cache.self_key;
      values = &cache.self_value;
      source = keys->size() / D;
    }
    float scale = 1 / std::sqrt(float(HD));
    for (auto &z : q)
      z *= scale;
    F32 context(D), score(source);
    for (std::size_t h = 0; h < H; ++h) {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, source, HD, 1,
                  q.data() + h * HD, D, keys->data() + h * HD, D, 0,
                  score.data(), source);
      softmax_rows(score, 1, source);
      if (cross && alignment)
        alignment->append(layer, h, score);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, HD, source, 1,
                  score.data(), source, values->data() + h * HD, D, 0,
                  context.data() + h * HD, D);
    }
    return linear(context, 1, D, graph_weight(t_, node, ".out_proj.weight"), D,
                  graph_weight(t_, node, ".out_proj.bias"));
  }

public:
  CachedDecoder(TensorStore &t, GraphExecutionAudit &audit)
      : t_(t), audit_(audit) {}
  F32 step(std::int32_t token, const F32 &memory, DecoderKVState &state,
           AlignmentTrace *alignment = nullptr, F32 *final_hidden = nullptr) {
    using namespace generated_whisper;
    if (token < 0 || token >= static_cast<std::int32_t>(V) ||
        state.position >= 448)
      throw std::runtime_error("cached decoder position");
    F32 x, branch, normed, logits;
    for (std::size_t i = 30; i <= 69; ++i) {
      const auto &node = nodes[i];
      audit_.hit(node);
      switch (node.opcode) {
      case Opcode::TokenStackInput:
        break;
      case Opcode::TokenPositionEmbed: {
        const auto &embedding = graph_weight(t_, node, "embed_tokens.weight");
        const auto &position = graph_weight(t_, node, "embed_positions.weight");
        x.resize(D);
        for (std::size_t j = 0; j < D; ++j)
          x[j] = embedding[std::size_t(token) * D + j] +
                 position[state.position * D + j];
        break;
      }
      case Opcode::LayerNorm:
        normed = layer_norm(x, 1, D, graph_weight(t_, node, ".weight"),
                            graph_weight(t_, node, ".bias"), 1e-5f);
        if (i == 68)
          x = normed;
        break;
      case Opcode::CachedSelfAttention:
        branch = attend_one(normed, normed, state.layers[(i - 33) / 9], node,
                            false, (i - 33) / 9, alignment);
        break;
      case Opcode::CrossAttention:
        branch = attend_one(normed, memory, state.layers[(i - 36) / 9], node,
                            true, (i - 36) / 9, alignment);
        break;
      case Opcode::MlpGelu:
        branch = linear(normed, 1, D, graph_weight(t_, node, ".fc1.weight"), FF,
                        graph_weight(t_, node, ".fc1.bias"));
        gelu(branch);
        branch = linear(branch, 1, FF, graph_weight(t_, node, ".fc2.weight"), D,
                        graph_weight(t_, node, ".fc2.bias"));
        break;
      case Opcode::ResidualAdd:
        add_inplace(x, branch);
        break;
      case Opcode::TiedLmHead:
        logits = linear(x, 1, D, graph_weight(t_, node, ".weight"), V);
        break;
      default:
        throw std::runtime_error("unexpected cached decoder opcode at node " +
                                 std::to_string(i));
      }
    }
    if (alignment)
      alignment->finish_position();
    if (final_hidden)
      *final_hidden = x;
    ++state.position;
    return logits;
  }
};
static F32 policy_logits(std::span<const float> logits, std::size_t step) {
  using namespace generated_whisper;
  F32 policy(logits.begin(), logits.end());
  for (std::int32_t i = 0; i < 51864; ++i) {
    bool blocked = std::find(suppress_tokens.begin(), suppress_tokens.end(),
                             i) != suppress_tokens.end() ||
                   (step == 0 && std::find(begin_suppress_tokens.begin(),
                                           begin_suppress_tokens.end(),
                                           i) != begin_suppress_tokens.end());
    if (blocked)
      policy[i] = -std::numeric_limits<float>::infinity();
  }
  return policy;
}
static void apply_generation_logit_policies(
    F32 &logits, std::span<const std::int32_t> stack, std::size_t begin_index,
    std::size_t maximum_positions,
    const whisper_interface::GenerationLogitPolicies &policies) {
  if (!policies.valid() || begin_index > stack.size() ||
      maximum_positions == 0 || maximum_positions > 448)
    throw std::runtime_error("generation logit policy ADT invariant");
  if (const auto *bias = std::get_if<whisper_interface::AdditiveSequenceBias>(
          &policies.sequence_bias))
    for (const auto &entry : bias->entries) {
      const auto prefix = entry.tokens.size() - 1;
      if (entry.tokens.size() > stack.size())
        continue;
      bool matches = true;
      for (std::size_t i = 0; i < prefix; ++i)
        if (stack[stack.size() - prefix + i] != entry.tokens[i]) {
          matches = false;
          break;
        }
      if (matches)
        logits[static_cast<std::size_t>(entry.tokens.back())] +=
            entry.additive_logit_bias;
    }
  if (const auto *repetition =
          std::get_if<whisper_interface::RepetitionPenalty>(
              &policies.repetition)) {
    std::array<bool, 51864> seen{};
    for (auto token : stack)
      if (!seen[static_cast<std::size_t>(token)]) {
        seen[static_cast<std::size_t>(token)] = true;
        auto &score = logits[static_cast<std::size_t>(token)];
        score =
            score < 0 ? score * repetition->factor : score / repetition->factor;
      }
  }
  if (const auto *ngram =
          std::get_if<whisper_interface::NoRepeatNGram>(&policies.ngrams);
      ngram && stack.size() + 1 >= ngram->order) {
    const auto prefix = ngram->order - 1;
    for (std::size_t start = 0; start + ngram->order <= stack.size(); ++start) {
      bool matches = true;
      for (std::size_t i = 0; i < prefix; ++i)
        if (stack[start + i] != stack[stack.size() - prefix + i]) {
          matches = false;
          break;
        }
      if (matches)
        logits[static_cast<std::size_t>(stack[start + prefix])] =
            -std::numeric_limits<float>::infinity();
    }
  }
  if (const auto *forbidden =
          std::get_if<whisper_interface::ForbiddenTokenSequences>(
              &policies.forbidden))
    for (const auto &sequence : forbidden->sequences) {
      if (sequence.size() == 1 &&
          sequence.front() == generated_whisper::eos_token)
        continue;
      const auto prefix = sequence.size() - 1;
      if (stack.size() < prefix)
        continue;
      bool matches = true;
      for (std::size_t i = 0; i < prefix; ++i)
        if (stack[stack.size() - prefix + i] != sequence[i]) {
          matches = false;
          break;
        }
      if (matches)
        logits[static_cast<std::size_t>(sequence.back())] =
            -std::numeric_limits<float>::infinity();
    }
  if (const auto *minimum = std::get_if<whisper_interface::MinimumLength>(
          &policies.minimum_length);
      minimum && stack.size() < minimum->positions)
    logits[generated_whisper::eos_token] =
        -std::numeric_limits<float>::infinity();
  if (const auto *minimum = std::get_if<whisper_interface::MinimumNewTokens>(
          &policies.minimum_new_tokens);
      minimum && stack.size() - begin_index < minimum->count)
    logits[generated_whisper::eos_token] =
        -std::numeric_limits<float>::infinity();
  if (const auto *forced = std::get_if<whisper_interface::ForcedBeginningToken>(
          &policies.forced_beginning);
      forced && stack.size() == 1) {
    std::fill(logits.begin(), logits.end(),
              -std::numeric_limits<float>::infinity());
    logits[static_cast<std::size_t>(forced->token)] = 0.0f;
  }
  if (const auto *forced = std::get_if<whisper_interface::ForcedEndingTokens>(
          &policies.forced_ending);
      forced && stack.size() == maximum_positions - 1) {
    std::fill(logits.begin(), logits.end(),
              -std::numeric_limits<float>::infinity());
    for (auto token : forced->tokens)
      logits[static_cast<std::size_t>(token)] = 0.0f;
  }
  if (std::holds_alternative<whisper_interface::RepairInvalidLogits>(
          policies.invalid_logits))
    for (auto &value : logits) {
      if (std::isnan(value))
        value = 0.0f;
      else if (value == std::numeric_limits<float>::infinity())
        value = std::numeric_limits<float>::max();
      else if (value == -std::numeric_limits<float>::infinity())
        value = std::numeric_limits<float>::lowest();
    }
  if (const auto *decay = std::get_if<whisper_interface::ExponentialEosDecay>(
          &policies.exponential_eos)) {
    const auto regulation_start = begin_index + decay->start_after_new_tokens;
    if (stack.size() > regulation_start) {
      const auto exponent = stack.size() - regulation_start;
      auto &eos = logits[generated_whisper::eos_token];
      eos += std::abs(eos) * (std::pow(decay->factor, exponent) - 1.0f);
    }
  }
}
static std::vector<std::int32_t> torch_cpu_randperm(std::size_t count,
                                                    std::uint64_t seed) {
  // PyTorch 2.10 CPU randperm for count < UINT32_MAX / 20: MT19937 seeded
  // from the low 32 bits, followed by forward Fisher-Yates swaps.
  std::mt19937 generator(static_cast<std::uint32_t>(seed));
  std::vector<std::int32_t> permutation(count);
  std::iota(permutation.begin(), permutation.end(), 0);
  for (std::size_t index = 0; index + 1 < count; ++index) {
    const auto offset = static_cast<std::size_t>(generator()) % (count - index);
    std::swap(permutation[index], permutation[index + offset]);
  }
  return permutation;
}
static const std::vector<std::int32_t> &
watermark_fixed_table(std::int64_t hashing_key) {
  static std::map<std::int64_t, std::vector<std::int32_t>> tables;
  auto found = tables.find(hashing_key);
  if (found == tables.end())
    found = tables
                .emplace(hashing_key,
                         torch_cpu_randperm(
                             1'000'003,
                             static_cast<std::uint64_t>(hashing_key)))
                .first;
  return found->second;
}
static std::uint64_t left_hash_seed(std::int64_t hashing_key,
                                    std::int32_t final_token) {
  constexpr auto modulus = std::numeric_limits<std::uint64_t>::max();
  const __int128 product = static_cast<__int128>(hashing_key) * final_token;
  if (product >= 0)
    return static_cast<std::uint64_t>(
        static_cast<unsigned __int128>(product) % modulus);
  const auto magnitude = static_cast<unsigned __int128>(-product);
  const auto remainder = static_cast<std::uint64_t>(magnitude % modulus);
  return remainder == 0 ? 0 : modulus - remainder;
}
static std::uint64_t python_signed_mod_u64_max(std::int64_t value) {
  if (value >= 0)
    return static_cast<std::uint64_t>(value);
  // For M = 2^64-1 and negative x in int64, Python's x % M is x+M.
  return std::bit_cast<std::uint64_t>(value) - 1;
}
static std::uint64_t self_hash_seed(
    std::span<const std::int32_t> sequence, std::int64_t hashing_key,
    std::size_t context_width) {
  constexpr std::size_t table_size = 1'000'003;
  if (sequence.empty() || context_width == 0)
    throw std::runtime_error("self-hash seed context");
  const auto &table = watermark_fixed_table(hashing_key);
  const auto begin = sequence.size() > context_width
                         ? sequence.size() - context_width
                         : 0;
  const std::uint64_t b =
      static_cast<std::uint64_t>(table[static_cast<std::size_t>(
          sequence.back()) % table_size]) +
      1;
  auto minimum = std::numeric_limits<std::int64_t>::max();
  for (std::size_t index = begin; index < sequence.size(); ++index) {
    const std::uint64_t a =
        static_cast<std::uint64_t>(table[static_cast<std::size_t>(
            sequence[index]) % table_size]) +
        1;
    const std::uint64_t wrapped =
        static_cast<std::uint64_t>(hashing_key) * a * b;
    minimum = std::min(minimum, std::bit_cast<std::int64_t>(wrapped));
  }
  return python_signed_mod_u64_max(minimum);
}
static F32 normalized_mass(std::span<const float> logits);
struct SynthIDWatermarkTrace {
  std::size_t call = 0;
  std::int64_t context_hash = 0;
  bool repeated_context = false;
  bool skipped_initial_ngram = false;
  bool capture_g_values = false;
  std::vector<std::uint8_t> g_values;
};
class SynthIDWatermarkRuntime {
  whisper_interface::SynthIDTextWatermark configuration_;
  std::vector<std::uint8_t> sampling_table_;
  std::vector<std::int64_t> context_;
  std::vector<std::int64_t> context_history_;
  std::size_t calls_ = 0;
  bool initialized_ = false;

  static std::int64_t accumulate(std::int64_t current, std::int64_t data) {
    constexpr std::uint64_t multiplier = 6364136223846793005ULL;
    auto bits = std::bit_cast<std::uint64_t>(current);
    bits += std::bit_cast<std::uint64_t>(data);
    bits *= multiplier;
    bits += 1;
    return std::bit_cast<std::int64_t>(bits);
  }
  static std::size_t positive_mod(std::int64_t value, std::size_t modulus) {
    if (modulus == 0)
      throw std::runtime_error("SynthID zero sampling table");
    if (value >= 0)
      return static_cast<std::uint64_t>(value) % modulus;
    const auto bits = std::bit_cast<std::uint64_t>(value);
    const auto magnitude = (~bits) + 1;
    const auto remainder = static_cast<std::size_t>(magnitude % modulus);
    return remainder == 0 ? 0 : modulus - remainder;
  }
  std::int64_t context_hash() const {
    std::int64_t hash = 1;
    for (auto token : context_)
      hash = accumulate(hash, token);
    return hash;
  }
  std::uint8_t g_value(std::int64_t candidate_hash,
                       std::int64_t key) const {
    const auto keyed = accumulate(candidate_hash, key);
    return sampling_table_[positive_mod(keyed, sampling_table_.size())];
  }

public:
  explicit SynthIDWatermarkRuntime(
      const whisper_interface::SynthIDTextWatermark &configuration)
      : configuration_(configuration),
        sampling_table_(configuration.sampling_table_size),
        context_(configuration.ngram_length - 1, 0),
        context_history_(configuration.context_history_size, 0) {
    if (!configuration_.valid())
      throw std::runtime_error("SynthID runtime ADT invariant");
    std::mt19937 generator(
        static_cast<std::uint32_t>(configuration_.sampling_table_seed));
    for (auto &value : sampling_table_)
      value = static_cast<std::uint8_t>(generator() % 2);
  }

  std::size_t apply(F32 &scores, std::span<const std::int32_t> stack,
                    SynthIDWatermarkTrace *trace = nullptr) {
    if (scores.size() != 51864)
      throw std::runtime_error("SynthID vocabulary shape");
    if (configuration_.debug_uniform_scores)
      std::fill(scores.begin(), scores.end(), 1.0f);
    if (!initialized_) {
      initialized_ = true;
    } else {
      if (stack.empty())
        throw std::runtime_error("SynthID empty generation stack");
      if (!context_.empty()) {
        std::move(context_.begin() + 1, context_.end(), context_.begin());
        context_.back() = stack.back();
      }
    }
    ++calls_;
    if (trace) {
      const bool capture_g_values = trace->capture_g_values;
      trace->call = calls_;
      trace->g_values.clear();
      trace->repeated_context = false;
      trace->skipped_initial_ngram = false;
      trace->capture_g_values = capture_g_values;
    }
    if (configuration_.skip_first_ngram_calls &&
        calls_ < configuration_.ngram_length) {
      if (trace)
        trace->skipped_initial_ngram = true;
      return 0;
    }

    const auto only_context_hash = context_hash();
    const bool repeated =
        std::find(context_history_.begin(), context_history_.end(),
                  only_context_hash) != context_history_.end();
    if (!context_history_.empty()) {
      std::move_backward(context_history_.begin(), context_history_.end() - 1,
                         context_history_.end());
      context_history_.front() = only_context_hash;
    }
    if (trace) {
      trace->context_hash = only_context_hash;
      trace->repeated_context = repeated;
      if (trace->capture_g_values)
        trace->g_values.assign(51864 * configuration_.keys.size(), 0);
    }

    const F32 unwatermarked = scores;
    auto probabilities = normalized_mass(scores);
    std::vector<std::int64_t> candidate_hashes(51864);
    for (std::int32_t token = 0; token < 51864; ++token)
      candidate_hashes[static_cast<std::size_t>(token)] =
          accumulate(only_context_hash, token);
    std::size_t one_count = 0;
    for (std::size_t depth = 0; depth < configuration_.keys.size(); ++depth) {
      std::vector<std::uint8_t> layer(51864);
      double mass = 0.0;
      for (std::size_t token = 0; token < 51864; ++token) {
        layer[token] =
            g_value(candidate_hashes[token], configuration_.keys[depth]);
        mass += static_cast<double>(layer[token]) * probabilities[token];
        one_count += layer[token];
        if (trace && trace->capture_g_values)
          trace->g_values[token * configuration_.keys.size() + depth] =
              layer[token];
      }
      const float g_mass = static_cast<float>(mass);
      for (std::size_t token = 0; token < 51864; ++token)
        probabilities[token] *=
            1.0f + static_cast<float>(layer[token]) - g_mass;
    }
    for (std::size_t token = 0; token < 51864; ++token) {
      scores[token] = std::log(probabilities[token]);
      if (!std::isfinite(scores[token]))
        scores[token] = std::numeric_limits<float>::lowest();
    }
    if (repeated)
      scores = unwatermarked;
    return one_count;
  }
};
template <typename Configuration>
static std::size_t watermark_greenlist_size(const Configuration &configuration) {
  return static_cast<std::size_t>(51864.0 * configuration.greenlist_ratio);
}
template <typename Configuration>
static std::vector<std::int32_t>
watermark_greenlist(std::span<const std::int32_t> sequence,
                    const Configuration &configuration) {
  const auto seed = [&]() {
    if constexpr (std::is_same_v<Configuration,
                                 whisper_interface::LeftHashWatermark>)
      return left_hash_seed(configuration.hashing_key, sequence.back());
    else if constexpr (std::is_same_v<
                           Configuration,
                           whisper_interface::SelfHashWatermark>)
      return self_hash_seed(sequence, configuration.hashing_key,
                            configuration.context_width);
    else
      throw std::runtime_error("non-greenlist watermark constructor");
  }();
  auto permutation = torch_cpu_randperm(51864, seed);
  permutation.resize(watermark_greenlist_size(configuration));
  return permutation;
}
static std::size_t apply_watermark_logit_policy(
    F32 &scores, std::span<const std::int32_t> stack,
    const whisper_interface::WatermarkPolicy &policy,
    std::vector<std::uint8_t> *green_mask = nullptr,
    SynthIDWatermarkRuntime *synthid_runtime = nullptr,
    SynthIDWatermarkTrace *synthid_trace = nullptr) {
  if (scores.size() != 51864)
    throw std::runtime_error("watermark vocabulary shape");
  if (green_mask)
    green_mask->assign(51864, 0);
  return std::visit(
      [&](const auto &configuration) -> std::size_t {
        using Configuration = std::decay_t<decltype(configuration)>;
        if constexpr (std::is_same_v<Configuration,
                                     whisper_interface::NoWatermark>) {
          return 0;
        } else if constexpr (std::is_same_v<
                                 Configuration,
                                 whisper_interface::SynthIDTextWatermark>) {
          if (!synthid_runtime)
            throw std::runtime_error("missing SynthID state graph");
          return synthid_runtime->apply(scores, stack, synthid_trace);
        } else {
          if (!configuration.valid())
            throw std::runtime_error("watermark ADT invariant");
          if (stack.size() < configuration.context_width)
            return 0;
          std::vector<std::int32_t> green;
          if constexpr (std::is_same_v<
                            Configuration,
                            whisper_interface::LeftHashWatermark>) {
            green = watermark_greenlist(stack, configuration);
          } else {
            std::vector<std::size_t> candidates(scores.size());
            std::iota(candidates.begin(), candidates.end(), 0);
            constexpr std::size_t rejection_candidates = 40;
            std::partial_sort(
                candidates.begin(), candidates.begin() + rejection_candidates,
                candidates.end(), [&](auto left, auto right) {
                  if (scores[left] != scores[right])
                    return scores[left] > scores[right];
                  return left < right;
                });
            std::vector<std::int32_t> candidate_sequence(stack.begin(),
                                                         stack.end());
            candidate_sequence.push_back(0);
            for (std::size_t rank = 0; rank < rejection_candidates; ++rank) {
              const auto candidate =
                  static_cast<std::int32_t>(candidates[rank]);
              candidate_sequence.back() = candidate;
              auto candidate_green =
                  watermark_greenlist(candidate_sequence, configuration);
              if (std::find(candidate_green.begin(), candidate_green.end(),
                            candidate) != candidate_green.end())
                green.push_back(candidate);
            }
          }
          for (auto token : green) {
            scores[static_cast<std::size_t>(token)] +=
                configuration.additive_bias;
            if (green_mask)
              (*green_mask)[static_cast<std::size_t>(token)] = 1;
          }
          return green.size();
        }
      },
      policy);
}
static F32 normalized_mass(std::span<const float> logits) {
  F32 mass(logits.begin(), logits.end());
  softmax_rows(mass, 1, mass.size());
  return mass;
}
static void
apply_sampling_filters(F32 &logits,
                       const whisper_interface::SamplingFilters &filters,
                       std::size_t minimum_tokens_to_keep = 1) {
  if (!filters.valid() || logits.empty())
    throw std::runtime_error("sampling filter ADT invariant");
  const float blocked = -std::numeric_limits<float>::infinity();
  minimum_tokens_to_keep =
      std::clamp<std::size_t>(minimum_tokens_to_keep, 1, logits.size());
  const auto preserve_minimum = [&](const F32 &before) {
    const auto retained = static_cast<std::size_t>(std::count_if(
        logits.begin(), logits.end(), [](float value) {
          return value != -std::numeric_limits<float>::infinity();
        }));
    if (retained >= minimum_tokens_to_keep)
      return;
    std::vector<std::size_t> order(before.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(
        order.begin(), order.begin() + minimum_tokens_to_keep, order.end(),
        [&](auto left, auto right) { return before[left] > before[right]; });
    for (std::size_t rank = 0; rank < minimum_tokens_to_keep; ++rank)
      logits[order[rank]] = before[order[rank]];
  };
  if (const auto *top_k =
          std::get_if<whisper_interface::TopK>(&filters.top_k)) {
    F32 ordered = logits;
    const auto count = std::min(
        std::max(top_k->count, minimum_tokens_to_keep), ordered.size());
    std::nth_element(ordered.begin(), ordered.begin() + count - 1,
                     ordered.end(), std::greater<float>{});
    const float threshold = ordered[count - 1];
    for (auto &value : logits)
      if (value < threshold)
        value = blocked;
  }
  if (const auto *top_p =
          std::get_if<whisper_interface::TopP>(&filters.top_p)) {
    const F32 before = logits;
    const auto probability = normalized_mass(logits);
    std::vector<std::size_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](auto left, auto right) {
      return logits[left] < logits[right];
    });
    double cumulative = 0.0;
    for (std::size_t rank = 0; rank + 1 < order.size(); ++rank) {
      cumulative += probability[order[rank]];
      if (cumulative <= 1.0 - top_p->mass)
        logits[order[rank]] = blocked;
    }
    preserve_minimum(before);
  }
  if (const auto *min_p =
          std::get_if<whisper_interface::MinP>(&filters.min_p)) {
    const F32 before = logits;
    const auto probability = normalized_mass(logits);
    const float threshold =
        min_p->fraction_of_maximum *
        *std::max_element(probability.begin(), probability.end());
    const auto best =
        std::max_element(logits.begin(), logits.end()) - logits.begin();
    for (std::size_t token = 0; token < logits.size(); ++token)
      if (token != static_cast<std::size_t>(best) &&
          probability[token] < threshold)
        logits[token] = blocked;
    preserve_minimum(before);
  }
  if (const auto *typical =
          std::get_if<whisper_interface::TypicalP>(&filters.typical_p)) {
    const F32 before = logits;
    const auto probability = normalized_mass(logits);
    std::vector<double> information(logits.size()), shifted(logits.size());
    double entropy = 0.0;
    for (std::size_t token = 0; token < logits.size(); ++token) {
      information[token] = probability[token] > 0.0f
                               ? -std::log(double(probability[token]))
                               : std::numeric_limits<double>::infinity();
      if (probability[token] > 0.0f)
        entropy += probability[token] * information[token];
    }
    for (std::size_t token = 0; token < logits.size(); ++token)
      shifted[token] = std::abs(information[token] - entropy);
    std::vector<std::size_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](auto left, auto right) {
      return shifted[left] < shifted[right];
    });
    double cumulative = 0.0;
    std::size_t last = 0;
    for (auto token : order) {
      cumulative += probability[token];
      if (cumulative < typical->mass)
        ++last;
    }
    last = std::min(last, order.size() - 1);
    const double threshold = shifted[order[last]];
    for (std::size_t token = 0; token < logits.size(); ++token)
      if (shifted[token] > threshold)
        logits[token] = blocked;
    logits[order.front()] =
        std::isfinite(logits[order.front()]) ? logits[order.front()] : blocked;
    preserve_minimum(before);
  }
  if (const auto *epsilon =
          std::get_if<whisper_interface::EpsilonCutoff>(&filters.epsilon)) {
    const F32 before = logits;
    const auto probability = normalized_mass(logits);
    const float best = *std::max_element(logits.begin(), logits.end());
    for (std::size_t token = 0; token < logits.size(); ++token)
      if (probability[token] < epsilon->probability && logits[token] < best)
        logits[token] = blocked;
    preserve_minimum(before);
  }
  if (const auto *eta =
          std::get_if<whisper_interface::EtaCutoff>(&filters.eta)) {
    const F32 before = logits;
    const auto probability = normalized_mass(logits);
    double entropy = 0.0;
    for (auto value : probability)
      if (value > 0.0f)
        entropy -= value * std::log(double(value));
    const double threshold =
        std::min(double(eta->epsilon),
                 std::sqrt(double(eta->epsilon)) * std::exp(-entropy));
    const float best = *std::max_element(logits.begin(), logits.end());
    for (std::size_t token = 0; token < logits.size(); ++token)
      if (probability[token] < threshold && logits[token] < best)
        logits[token] = blocked;
    preserve_minimum(before);
  }
}
static F32 timestamp_policy_logits(std::span<const float> logits,
                                   std::size_t step,
                                   std::span<const std::int32_t> stack,
                                   std::size_t begin_index) {
  constexpr std::int32_t no_timestamps = 50362, timestamp_begin = 50363,
                         eos = 50256, max_initial_index = 50;
  auto policy = policy_logits(logits, step);
  policy[no_timestamps] = -std::numeric_limits<float>::infinity();
  auto generated = stack.subspan(std::min(begin_index, stack.size()));
  bool last_timestamp =
      !generated.empty() && generated.back() >= timestamp_begin;
  bool penultimate_timestamp =
      generated.size() < 2 ||
      generated[generated.size() - 2] >= timestamp_begin;
  if (last_timestamp) {
    if (penultimate_timestamp)
      std::fill(policy.begin() + timestamp_begin, policy.end(),
                -std::numeric_limits<float>::infinity());
    else
      std::fill(policy.begin(), policy.begin() + eos,
                -std::numeric_limits<float>::infinity());
  }
  std::int32_t last = -1;
  for (auto token : generated)
    if (token >= timestamp_begin)
      last = token;
  if (last >= timestamp_begin) {
    auto first_forbidden_end =
        last + (last_timestamp && !penultimate_timestamp ? 0 : 1);
    std::fill(policy.begin() + timestamp_begin,
              policy.begin() +
                  std::min<std::int32_t>(first_forbidden_end, 51864),
              -std::numeric_limits<float>::infinity());
  }
  if (stack.size() == begin_index) {
    std::fill(policy.begin(), policy.begin() + timestamp_begin,
              -std::numeric_limits<float>::infinity());
    std::fill(policy.begin() + timestamp_begin + max_initial_index + 1,
              policy.end(), -std::numeric_limits<float>::infinity());
  }
  double timestamp_max = -std::numeric_limits<double>::infinity();
  for (auto i = timestamp_begin; i < 51864; ++i)
    timestamp_max = std::max(timestamp_max, double(policy[i]));
  double timestamp_sum = 0;
  if (std::isfinite(timestamp_max))
    for (auto i = timestamp_begin; i < 51864; ++i)
      timestamp_sum += std::exp(double(policy[i]) - timestamp_max);
  double timestamp_logsum = timestamp_max + std::log(timestamp_sum);
  double text_max =
      *std::max_element(policy.begin(), policy.begin() + timestamp_begin);
  if (timestamp_logsum > text_max)
    std::fill(policy.begin(), policy.begin() + timestamp_begin,
              -std::numeric_limits<float>::infinity());
  return policy;
}
enum class Selection { Greedy, Sample };
static std::int32_t select_token(const F32 &mass, Selection mode,
                                 std::mt19937_64 &rng) {
  if (mode == Selection::Greedy)
    return static_cast<std::int32_t>(
        std::max_element(mass.begin(), mass.end()) - mass.begin());
  std::uniform_real_distribution<double> u(0, 1);
  double draw = u(rng), sum = 0;
  for (std::size_t i = 0; i < mass.size(); ++i) {
    sum += mass[i];
    if (draw <= sum)
      return static_cast<std::int32_t>(i);
  }
  return static_cast<std::int32_t>(mass.size() - 1);
}
struct GenerationResult {
  std::vector<std::int32_t> tokens;
  double max_mass_sum_error = 0, max_selected_mass_error = 0,
         max_cache_logit_error = 0, selected_logprob_sum = 0,
         no_speech_probability = 0;
  std::size_t selected_logprob_count = 0, cache_positions = 0;
  bool terminated_by_eos = false, terminated_by_stop_string = false;
  double average_logprob() const {
    return selected_logprob_count
               ? selected_logprob_sum / selected_logprob_count
               : -std::numeric_limits<double>::infinity();
  }
};
using StopStringPredicate =
    std::function<bool(std::span<const std::int32_t>, std::int32_t)>;
static double categorical_probability(std::span<const float> logits,
                                      std::size_t index) {
  if (index >= logits.size())
    throw std::runtime_error("probability index");
  const float maximum = *std::max_element(logits.begin(), logits.end());
  double sum = 0;
  for (auto value : logits)
    sum += std::exp(double(value - maximum));
  return std::exp(double(logits[index] - maximum)) / sum;
}
static double compression_ratio(std::span<const std::int32_t> tokens,
                                bool append_eos) {
  std::vector<unsigned char> raw;
  raw.reserve(2 * (tokens.size() + append_eos));
  for (auto token : tokens) {
    raw.push_back(static_cast<unsigned char>(token & 255));
    raw.push_back(static_cast<unsigned char>((token >> 8) & 255));
  }
  if (append_eos) {
    raw.push_back(
        static_cast<unsigned char>(generated_whisper::eos_token & 255));
    raw.push_back(
        static_cast<unsigned char>((generated_whisper::eos_token >> 8) & 255));
  }
  if (raw.empty())
    return 0;
  uLongf compressed_size = compressBound(raw.size());
  std::vector<unsigned char> compressed(compressed_size);
  if (compress2(compressed.data(), &compressed_size, raw.data(), raw.size(),
                Z_DEFAULT_COMPRESSION) != Z_OK)
    throw std::runtime_error("zlib compression");
  return double(raw.size()) / compressed_size;
}
struct TimestampSegment {
  double start_seconds = 0, end_seconds = 0;
  std::size_t token_begin = 0, token_end = 0;
};
static std::vector<TimestampSegment>
timestamp_segments(std::span<const std::int32_t> tokens,
                   double seconds_per_token, double fallback_seconds = 30.0) {
  constexpr std::int32_t timestamp_begin = 50363;
  if (seconds_per_token <= 0)
    throw std::runtime_error("timestamp precision");
  std::vector<std::size_t> boundaries;
  for (std::size_t i = 0; i + 1 < tokens.size(); ++i)
    if (tokens[i] >= timestamp_begin && tokens[i + 1] >= timestamp_begin)
      boundaries.push_back(i + 1);
  const bool single_timestamp_ending =
      tokens.size() >= 2 && tokens[tokens.size() - 2] < timestamp_begin &&
      tokens.back() >= timestamp_begin;
  std::vector<TimestampSegment> segments;
  if (boundaries.empty()) {
    auto last = std::find_if(tokens.rbegin(), tokens.rend(), [](auto token) {
      return token >= timestamp_begin;
    });
    const double end =
        last == tokens.rend() || *last == timestamp_begin
            ? fallback_seconds
            : double(*last - timestamp_begin) * seconds_per_token;
    segments.push_back({0.0, end, 0, tokens.size()});
    return segments;
  }
  if (single_timestamp_ending)
    boundaries.push_back(tokens.size());
  else
    ++boundaries.back();
  std::size_t begin = 0;
  for (std::size_t i = 0; i < boundaries.size(); ++i) {
    const auto end = boundaries[i];
    if (end <= begin || tokens[begin] < timestamp_begin)
      throw std::runtime_error("timestamp segment structure");
    const auto end_token =
        (i + 1 < boundaries.size() || single_timestamp_ending)
            ? tokens[end - 1]
            : tokens[end - 2];
    if (end_token < timestamp_begin)
      throw std::runtime_error("timestamp segment end");
    segments.push_back(
        {double(tokens[begin] - timestamp_begin) * seconds_per_token,
         double(end_token - timestamp_begin) * seconds_per_token, begin, end});
    begin = end;
  }
  return segments;
}
struct LongFormSegment {
  double start_seconds = 0, end_seconds = 0;
  std::vector<std::int32_t> tokens;
};
static std::size_t timestamp_seek_offset(std::span<const std::int32_t> tokens,
                                         std::size_t seek_frames) {
  constexpr std::int32_t timestamp_begin = 50363;
  const bool single_timestamp_ending =
      tokens.size() >= 2 && tokens[tokens.size() - 2] < timestamp_begin &&
      tokens.back() >= timestamp_begin;
  std::size_t last_pair = tokens.size();
  for (std::size_t i = 0; i + 1 < tokens.size(); ++i)
    if (tokens[i] >= timestamp_begin && tokens[i + 1] >= timestamp_begin)
      last_pair = i;
  if (last_pair == tokens.size() || single_timestamp_ending)
    return seek_frames;
  return static_cast<std::size_t>(tokens[last_pair] - timestamp_begin) * 2;
}
static std::vector<std::int32_t> previous_segment_prefix(
    const std::vector<LongFormSegment> &segments,
    const std::vector<std::int32_t> *all_segment_prompt = nullptr) {
  constexpr std::int32_t timestamp_begin = 50363, previous_start_token = 50360;
  constexpr std::size_t cutoff = 223;
  std::vector<std::int32_t> history;
  for (const auto &segment : segments) {
    auto end = segment.tokens.size();
    if (end > 2 && segment.tokens[end - 2] >= timestamp_begin)
      --end;
    history.insert(history.end(), segment.tokens.begin(),
                   segment.tokens.begin() + static_cast<std::ptrdiff_t>(end));
  }
  if (history.size() > cutoff)
    history.erase(history.begin(),
                  history.end() - static_cast<std::ptrdiff_t>(cutoff));
  if (all_segment_prompt)
    history.insert(history.begin(), all_segment_prompt->begin(),
                   all_segment_prompt->end());
  else
    history.insert(history.begin(), previous_start_token);
  return history;
}
static F32 aligned_token_timestamps(const AlignmentTrace &trace,
                                    std::size_t prefix_positions,
                                    std::size_t feature_frames,
                                    float seconds_per_encoder_position) {
  constexpr std::size_t heads = AlignmentTrace::heads,
                        source_stride = AlignmentTrace::source_positions,
                        filter_width = 7, pad = filter_width / 2;
  if (prefix_positions > trace.positions ||
      trace.weights.size() != trace.positions * heads * source_stride)
    throw std::runtime_error("alignment trace shape");
  const std::size_t output_positions = trace.positions - prefix_positions;
  const std::size_t source_positions =
      std::min(source_stride, feature_frames / 2);
  if (output_positions == 0 || source_positions == 0 ||
      seconds_per_encoder_position <= 0)
    throw std::runtime_error("alignment timestamp domain");
  F32 normalized(heads * output_positions * source_positions);
  for (std::size_t h = 0; h < heads; ++h)
    for (std::size_t s = 0; s < source_positions; ++s) {
      double sum = 0, square_sum = 0;
      for (std::size_t t = 0; t < output_positions; ++t) {
        const float value =
            trace.weights[((prefix_positions + t) * heads + h) * source_stride +
                          s];
        sum += value;
        square_sum += double(value) * value;
      }
      const double mean = sum / output_positions;
      const double variance =
          std::max(0.0, square_sum / output_positions - mean * mean);
      const float inverse_std = 1.0f / static_cast<float>(std::sqrt(variance));
      for (std::size_t t = 0; t < output_positions; ++t)
        normalized[(h * output_positions + t) * source_positions + s] =
            (trace
                 .weights[((prefix_positions + t) * heads + h) * source_stride +
                          s] -
             static_cast<float>(mean)) *
            inverse_std;
    }
  F32 matrix(output_positions * source_positions);
  for (std::size_t h = 0; h < heads; ++h)
    for (std::size_t t = 0; t < output_positions; ++t)
      for (std::size_t s = 0; s < source_positions; ++s) {
        std::array<float, filter_width> window{};
        for (std::size_t k = 0; k < filter_width; ++k) {
          long source = static_cast<long>(s) + static_cast<long>(k) -
                        static_cast<long>(pad);
          if (source < 0)
            source = -source;
          if (source >= static_cast<long>(source_positions))
            source = 2 * static_cast<long>(source_positions) - 2 - source;
          window[k] = normalized[(h * output_positions + t) * source_positions +
                                 static_cast<std::size_t>(source)];
        }
        std::sort(window.begin(), window.end());
        matrix[t * source_positions + s] += window[pad] / heads;
      }
  const std::size_t cost_columns = source_positions + 1;
  F32 cost((output_positions + 1) * cost_columns,
           std::numeric_limits<float>::infinity());
  std::vector<std::int8_t> direction(cost.size(), -1);
  cost[0] = 0;
  for (std::size_t j = 1; j <= source_positions; ++j)
    for (std::size_t i = 1; i <= output_positions; ++i) {
      const float diagonal = cost[(i - 1) * cost_columns + j - 1],
                  vertical = cost[(i - 1) * cost_columns + j],
                  horizontal = cost[i * cost_columns + j - 1];
      float previous;
      std::int8_t step;
      if (diagonal < vertical && diagonal < horizontal) {
        previous = diagonal;
        step = 0;
      } else if (vertical < diagonal && vertical < horizontal) {
        previous = vertical;
        step = 1;
      } else {
        previous = horizontal;
        step = 2;
      }
      cost[i * cost_columns + j] =
          -matrix[(i - 1) * source_positions + j - 1] + previous;
      direction[i * cost_columns + j] = step;
    }
  for (std::size_t j = 0; j <= source_positions; ++j)
    direction[j] = 2;
  for (std::size_t i = 0; i <= output_positions; ++i)
    direction[i * cost_columns] = 1;
  std::vector<std::size_t> text_indices, time_indices;
  std::size_t i = output_positions, j = source_positions;
  while (i > 0 || j > 0) {
    text_indices.push_back(i - 1);
    time_indices.push_back(j - 1);
    switch (direction[i * cost_columns + j]) {
    case 0:
      --i;
      --j;
      break;
    case 1:
      --i;
      break;
    case 2:
      --j;
      break;
    default:
      throw std::runtime_error("alignment DTW trace");
    }
  }
  std::reverse(text_indices.begin(), text_indices.end());
  std::reverse(time_indices.begin(), time_indices.end());
  F32 timestamps(prefix_positions, 0.0f);
  for (std::size_t k = 0; k < text_indices.size(); ++k)
    if (k == 0 || text_indices[k] != text_indices[k - 1])
      timestamps.push_back(static_cast<float>(time_indices[k]) *
                           seconds_per_encoder_position);
  if (timestamps.size() != prefix_positions + output_positions)
    throw std::runtime_error("alignment DTW token coverage");
  timestamps.push_back(timestamps.back());
  return timestamps;
}
static GenerationResult generate(
    Decoder &decoder, CachedDecoder &cached, const F32 &memory,
    Selection selection, float temperature, std::uint64_t seed,
    bool verify_recompute, std::span<const std::int32_t> prompt,
    std::span<const std::int32_t> prefix,
    const whisper_interface::TimeOutput &time_output,
    const whisper_interface::PrefixAllowedTokensFn *prefix_allowed,
    AlignmentTrace *alignment, GraphExecutionAudit &audit,
    const whisper_interface::GenerationLogitPolicies *logit_policies = nullptr,
    std::size_t maximum_positions = 448,
    const StopStringPredicate *stop_string = nullptr,
    std::vector<SynthIDWatermarkTrace> *synthid_traces = nullptr) {
  using namespace generated_whisper;
  const bool timestamp_mode =
      std::holds_alternative<whisper_interface::TimestampTokens>(time_output) ||
      std::holds_alternative<whisper_interface::Segments>(time_output);
  std::vector<std::int32_t> stack;
  DecoderKVState state;
  GenerationResult result;
  F32 cached_logits;
  auto verify_step = [&](std::int32_t token) {
    audit.hit(nodes[73]);
    stack.push_back(token);
    cached_logits = cached.step(token, memory, state, alignment);
    if (verify_recompute) {
      std::vector<std::pair<std::string, F32>> trace;
      auto full = decoder.run(stack, memory, trace);
      auto offset = (stack.size() - 1) * 51864;
      for (std::size_t i = 0; i < 51864; ++i)
        result.max_cache_logit_error =
            std::max(result.max_cache_logit_error,
                     std::abs(double(cached_logits[i]) - full[offset + i]));
    }
  };
  if (maximum_positions == 0 || maximum_positions > 448 ||
      prompt.size() + prefix.size() >= maximum_positions)
    throw std::runtime_error("generation prefix length");
  for (auto token : prompt)
    verify_step(token);
  for (auto token : prefix)
    verify_step(token);
  const auto begin_index = stack.size();
  std::optional<SynthIDWatermarkRuntime> synthid_runtime;
  if (logit_policies)
    if (const auto *configuration =
            std::get_if<whisper_interface::SynthIDTextWatermark>(
                &logit_policies->watermark))
      synthid_runtime.emplace(*configuration);
  std::mt19937_64 rng(seed);
  for (std::size_t step = 0; state.position < maximum_positions; ++step) {
    if (step == 0)
      result.no_speech_probability =
          categorical_probability(cached_logits, 50361);
    F32 mass, quality_mass;
    std::int32_t token = -1;
    for (std::size_t i = 70; i <= 72; ++i) {
      const auto &node = nodes[i];
      audit.hit(node);
      switch (node.opcode) {
      case Opcode::GenerationPolicy: {
        F32 configured_logits = cached_logits;
        if (logit_policies)
          apply_generation_logit_policies(configured_logits, stack, begin_index,
                                          maximum_positions, *logit_policies);
        mass = timestamp_mode ? timestamp_policy_logits(configured_logits, step,
                                                        stack, begin_index)
                              : policy_logits(configured_logits, step);
        if (prefix_allowed) {
          if (!prefix_allowed->valid())
            throw std::runtime_error("prefix vocabulary callback");
          bool any = false;
          for (std::int32_t token = 0; token < 51864; ++token)
            if (prefix_allowed->allows(step, stack, token))
              any = true;
            else
              mass[token] = -std::numeric_limits<float>::infinity();
          if (!any)
            throw std::runtime_error("empty allowed vocabulary");
        }
        quality_mass = mass;
        if (selection == Selection::Sample) {
          if (temperature <= 0)
            throw std::runtime_error("sampling temperature");
          for (auto &value : mass)
            value /= temperature;
          if (logit_policies)
            apply_sampling_filters(mass, logit_policies->sampling);
        }
        SynthIDWatermarkTrace synthid_trace;
        if (logit_policies)
          apply_watermark_logit_policy(mass, stack,
                                       logit_policies->watermark, nullptr,
                                       synthid_runtime
                                           ? &*synthid_runtime
                                           : nullptr,
                                       synthid_traces ? &synthid_trace
                                                      : nullptr);
        if (synthid_traces && synthid_runtime)
          synthid_traces->push_back(std::move(synthid_trace));
        if (logit_policies && std::holds_alternative<
                                  whisper_interface::NormalizeLogProbabilities>(
                                  logit_policies->normalization)) {
          const auto probability = normalized_mass(mass);
          for (std::size_t token_index = 0; token_index < mass.size();
               ++token_index)
            mass[token_index] = probability[token_index] > 0.0f
                                    ? std::log(probability[token_index])
                                    : -std::numeric_limits<float>::infinity();
        }
        break;
      }
      case Opcode::Softmax:
        softmax_rows(mass, 1, mass.size());
        break;
      case Opcode::SampleOrArgmax:
        token = select_token(mass, selection, rng);
        break;
      default:
        throw std::runtime_error("unexpected generation opcode at node " +
                                 std::to_string(i));
      }
    }
    double sum = std::accumulate(mass.begin(), mass.end(), 0.0);
    result.max_mass_sum_error =
        std::max(result.max_mass_sum_error, std::abs(sum - 1));
    softmax_rows(quality_mass, 1, quality_mass.size());
    result.selected_logprob_sum += std::log(std::max(
        double(quality_mass[token]), std::numeric_limits<double>::min()));
    ++result.selected_logprob_count;
    if (!timestamp_mode && step < expected_selected_mass.size() &&
        token == expected_sample_tokens[step])
      result.max_selected_mass_error = std::max(
          result.max_selected_mass_error,
          std::abs(double(mass[token]) - expected_selected_mass[step]));
    if (token == eos_token) {
      result.terminated_by_eos = true;
      result.cache_positions = state.position;
      for (const auto &layer : state.layers)
        if (layer.self_key.size() != state.position * 384 ||
            layer.self_value.size() != state.position * 384 ||
            layer.cross_key.size() != 1500 * 384 ||
            layer.cross_value.size() != 1500 * 384)
          throw std::runtime_error("cache shape invariant");
      return result;
    }
    result.tokens.push_back(token);
    if (stop_string && (*stop_string)(stack, token)) {
      result.terminated_by_stop_string = true;
      result.cache_positions = state.position;
      for (const auto &layer : state.layers)
        if (layer.self_key.size() != state.position * 384 ||
            layer.self_value.size() != state.position * 384 ||
            layer.cross_key.size() != 1500 * 384 ||
            layer.cross_value.size() != 1500 * 384)
          throw std::runtime_error("cache shape invariant");
      return result;
    }
    verify_step(token);
  }
  result.cache_positions = state.position;
  for (const auto &layer : state.layers)
    if (layer.self_key.size() != state.position * 384 ||
        layer.self_value.size() != state.position * 384 ||
        layer.cross_key.size() != 1500 * 384 ||
        layer.cross_value.size() != 1500 * 384)
      throw std::runtime_error("cache shape invariant");
  return result;
}
struct BeamSequence {
  std::vector<std::int32_t> tokens;
  float score = -std::numeric_limits<float>::infinity();
  std::vector<std::size_t> parent_beams;
};
struct BeamSearchResult {
  std::vector<BeamSequence> sequences;
  std::size_t expanded_candidates = 0;
  std::size_t cache_branches = 0;
  std::vector<std::int64_t> synthid_context_hashes;
  std::vector<std::uint8_t> synthid_repeated;
  std::vector<std::uint8_t> synthid_skipped;
};
struct LiveBeam {
  std::vector<std::int32_t> stack;
  DecoderKVState state;
  F32 next_logits;
  float cumulative_log_probability = 0.0f;
  std::vector<std::size_t> parent_beams;
};
struct BeamCandidate {
  std::size_t parent = 0;
  std::int32_t token = 0;
  float cumulative_log_probability = 0.0f;
  bool finished = false;
};
static F32 log_softmax_logits(std::span<const float> logits) {
  const float maximum = *std::max_element(logits.begin(), logits.end());
  double sum = 0.0;
  for (auto value : logits)
    sum += std::exp(double(value - maximum));
  const float normalizer = maximum + static_cast<float>(std::log(sum));
  F32 output(logits.size());
  for (std::size_t token = 0; token < logits.size(); ++token)
    output[token] = logits[token] - normalizer;
  return output;
}
static float length_normalized_beam_score(float cumulative,
                                          std::size_t generated_positions,
                                          float length_penalty) {
  if (generated_positions == 0)
    throw std::runtime_error("beam score zero length");
  return cumulative / static_cast<float>(std::pow(double(generated_positions),
                                                  double(length_penalty)));
}
static BeamSearchResult beam_search(
    CachedDecoder &cached, const F32 &memory,
    std::span<const std::int32_t> prefix,
    const whisper_interface::StandardBeamSearch &configuration,
    std::size_t maximum_positions, GraphExecutionAudit &audit,
    const whisper_interface::GenerationLogitPolicies *policies = nullptr) {
  using namespace generated_whisper;
  if (!configuration.valid() || prefix.empty() ||
      prefix.size() >= maximum_positions || maximum_positions > 448)
    throw std::runtime_error("beam search ADT invariant");
  LiveBeam initial;
  for (auto token : prefix) {
    audit.hit(nodes[73]);
    initial.stack.push_back(token);
    initial.next_logits = cached.step(token, memory, initial.state);
  }
  const auto decoder_prompt_length = initial.stack.size();
  std::vector<LiveBeam> live;
  live.push_back(std::move(initial));
  std::vector<BeamSequence> finished;
  BeamSearchResult result;
  std::vector<SynthIDWatermarkRuntime> synthid_rows;
  if (policies)
    if (const auto *synthid =
            std::get_if<whisper_interface::SynthIDTextWatermark>(
                &policies->watermark)) {
      synthid_rows.reserve(configuration.beams);
      for (std::size_t row = 0; row < configuration.beams; ++row)
        synthid_rows.emplace_back(*synthid);
    }
  bool first_beam_step = true;
  while (!live.empty()) {
    if (!synthid_rows.empty() && !first_beam_step &&
        live.size() != configuration.beams)
      throw std::runtime_error("SynthID beam-row cardinality");
    std::vector<BeamCandidate> candidates;
    candidates.reserve(live.size() * 51864);
    for (std::size_t beam_index = 0; beam_index < live.size(); ++beam_index) {
      audit.hit(nodes[70]);
      auto log_probability = log_softmax_logits(live[beam_index].next_logits);
      if (policies)
        apply_generation_logit_policies(log_probability, live[beam_index].stack,
                                        decoder_prompt_length,
                                        maximum_positions, *policies);
      log_probability =
          policy_logits(log_probability,
                        live[beam_index].stack.size() - decoder_prompt_length);
      SynthIDWatermarkTrace synthid_trace;
      if (policies)
        apply_watermark_logit_policy(log_probability,
                                     live[beam_index].stack,
                                     policies->watermark, nullptr,
                                     synthid_rows.empty()
                                         ? nullptr
                                         : &synthid_rows[beam_index],
                                     synthid_rows.empty() ? nullptr
                                                          : &synthid_trace);
      if (!synthid_rows.empty()) {
        result.synthid_context_hashes.push_back(synthid_trace.context_hash);
        result.synthid_repeated.push_back(synthid_trace.repeated_context);
        result.synthid_skipped.push_back(
            synthid_trace.skipped_initial_ngram);
        if (first_beam_step) {
          if (beam_index != 0 || live.size() != 1)
            throw std::runtime_error("SynthID initial beam shape");
          for (std::size_t row = 1; row < configuration.beams; ++row) {
            synthid_rows[row] = synthid_rows[0];
            result.synthid_context_hashes.push_back(
                synthid_trace.context_hash);
            result.synthid_repeated.push_back(
                synthid_trace.repeated_context);
            result.synthid_skipped.push_back(
                synthid_trace.skipped_initial_ngram);
          }
        }
      }
      audit.hit(nodes[71]);
      for (std::int32_t token = 0; token < 51864; ++token) {
        const float score = live[beam_index].cumulative_log_probability +
                            log_probability[static_cast<std::size_t>(token)];
        candidates.push_back(
            {beam_index, token, score,
             token == eos_token ||
                 live[beam_index].stack.size() + 1 >= maximum_positions});
      }
    }
    result.expanded_candidates += candidates.size();
    const auto keep =
        std::min<std::size_t>(2 * configuration.beams, candidates.size());
    std::partial_sort(candidates.begin(),
                      candidates.begin() + static_cast<std::ptrdiff_t>(keep),
                      candidates.end(),
                      [](const auto &left, const auto &right) {
                        if (left.cumulative_log_probability !=
                            right.cumulative_log_probability)
                          return left.cumulative_log_probability >
                                 right.cumulative_log_probability;
                        if (left.parent != right.parent)
                          return left.parent < right.parent;
                        return left.token < right.token;
                      });
    candidates.resize(keep);
    audit.hit(nodes[72]);

    for (std::size_t rank = 0;
         rank < std::min(configuration.beams, candidates.size()); ++rank) {
      const auto &candidate = candidates[rank];
      if (!candidate.finished)
        continue;
      BeamSequence completed;
      completed.tokens = live[candidate.parent].stack;
      completed.tokens.push_back(candidate.token);
      completed.parent_beams = live[candidate.parent].parent_beams;
      completed.parent_beams.push_back(candidate.parent);
      completed.score = length_normalized_beam_score(
          candidate.cumulative_log_probability,
          completed.tokens.size() - decoder_prompt_length,
          configuration.length_penalty);
      finished.push_back(std::move(completed));
    }
    std::sort(finished.begin(), finished.end(),
              [](const auto &left, const auto &right) {
                return left.score > right.score;
              });
    if (finished.size() > configuration.beams)
      finished.resize(configuration.beams);

    std::vector<LiveBeam> next;
    next.reserve(configuration.beams);
    for (const auto &candidate : candidates) {
      if (candidate.finished || next.size() == configuration.beams)
        continue;
      LiveBeam branch = live[candidate.parent];
      branch.cumulative_log_probability = candidate.cumulative_log_probability;
      branch.parent_beams.push_back(candidate.parent);
      branch.stack.push_back(candidate.token);
      audit.hit(nodes[73]);
      branch.next_logits =
          cached.step(candidate.token, memory, branch.state, nullptr);
      ++result.cache_branches;
      next.push_back(std::move(branch));
    }
    live = std::move(next);
    first_beam_step = false;
    if (live.empty())
      break;
    if (std::holds_alternative<whisper_interface::StopWhenAllBeamsFinished>(
            configuration.stopping) &&
        finished.size() == configuration.beams)
      break;
    if (finished.size() == configuration.beams) {
      const auto generated = live.front().stack.size() - decoder_prompt_length;
      const auto best_running = length_normalized_beam_score(
          live.front().cumulative_log_probability,
          std::holds_alternative<whisper_interface::CanonicalBeamStopping>(
              configuration.stopping) &&
                  configuration.length_penalty > 0.0f
              ? maximum_positions - decoder_prompt_length
              : generated,
          configuration.length_penalty);
      if (best_running <= finished.back().score)
        break;
    }
  }
  if (finished.size() < configuration.return_sequences)
    throw std::runtime_error("beam search finalization coverage");
  finished.resize(configuration.return_sequences);
  result.sequences = std::move(finished);
  return result;
}

static bool better_beam_candidate(const BeamCandidate &left,
                                  const BeamCandidate &right) {
  if (left.cumulative_log_probability != right.cumulative_log_probability)
    return left.cumulative_log_probability > right.cumulative_log_probability;
  if (left.parent != right.parent)
    return left.parent < right.parent;
  return left.token < right.token;
}

struct DiverseBeamGroup {
  std::vector<LiveBeam> live;
  std::vector<BeamSequence> finished;
  bool done = false;
};

static BeamSearchResult group_beam_search(
    CachedDecoder &cached, const F32 &memory,
    std::span<const std::int32_t> prefix,
    const whisper_interface::DiverseGroupBeamSearch &configuration,
    std::size_t maximum_positions, GraphExecutionAudit &audit,
    const whisper_interface::GenerationLogitPolicies *policies = nullptr) {
  using namespace generated_whisper;
  if (!configuration.valid() || prefix.empty() ||
      prefix.size() >= maximum_positions || maximum_positions > 448)
    throw std::runtime_error("group beam search ADT invariant");
  LiveBeam initial;
  for (auto token : prefix) {
    audit.hit(nodes[73]);
    initial.stack.push_back(token);
    initial.next_logits = cached.step(token, memory, initial.state);
  }
  const auto decoder_prompt_length = initial.stack.size();
  const auto group_size = configuration.beams / configuration.groups;
  std::vector<DiverseBeamGroup> groups(configuration.groups);
  for (auto &group : groups)
    group.live.push_back(initial);
  BeamSearchResult result;
  std::vector<SynthIDWatermarkRuntime> synthid_rows;
  if (policies)
    if (const auto *synthid =
            std::get_if<whisper_interface::SynthIDTextWatermark>(
                &policies->watermark)) {
      synthid_rows.reserve(group_size);
      for (std::size_t row = 0; row < group_size; ++row)
        synthid_rows.emplace_back(*synthid);
    }
  bool first_group_position = true;
  bool first_synthid_call = true;

  while (std::any_of(groups.begin(), groups.end(),
                     [](const auto &group) { return !group.done; })) {
    std::vector<std::int32_t> earlier_group_tokens;
    earlier_group_tokens.reserve(configuration.beams);
    for (auto &group : groups) {
      if (group.done) {
        earlier_group_tokens.insert(earlier_group_tokens.end(), group_size,
                                    generated_whisper::eos_token);
        continue;
      }
      std::vector<BeamCandidate> candidates;
      candidates.reserve(group.live.size() * 51864);
      if (!synthid_rows.empty() && !first_group_position &&
          group.live.size() != group_size)
        throw std::runtime_error("SynthID group beam-row cardinality");
      for (std::size_t beam_index = 0; beam_index < group.live.size();
           ++beam_index) {
        audit.hit(nodes[70]);
        auto log_probability =
            log_softmax_logits(group.live[beam_index].next_logits);
        if (policies)
          apply_generation_logit_policies(
              log_probability, group.live[beam_index].stack,
              decoder_prompt_length, maximum_positions, *policies);
        log_probability = policy_logits(
            log_probability,
            group.live[beam_index].stack.size() - decoder_prompt_length);
        SynthIDWatermarkTrace synthid_trace;
        if (policies)
          apply_watermark_logit_policy(
              log_probability, group.live[beam_index].stack,
              policies->watermark, nullptr,
              synthid_rows.empty() ? nullptr : &synthid_rows[beam_index],
              synthid_rows.empty() ? nullptr : &synthid_trace);
        if (!synthid_rows.empty()) {
          result.synthid_context_hashes.push_back(synthid_trace.context_hash);
          result.synthid_repeated.push_back(synthid_trace.repeated_context);
          result.synthid_skipped.push_back(synthid_trace.skipped_initial_ngram);
          if (first_group_position && group.live.size() == 1) {
            if (beam_index != 0)
              throw std::runtime_error("SynthID group initial beam shape");
            for (std::size_t row = 1; row < group_size; ++row) {
              if (first_synthid_call) {
                synthid_rows[row] = synthid_rows[0];
                result.synthid_context_hashes.push_back(
                    synthid_trace.context_hash);
                result.synthid_repeated.push_back(
                    synthid_trace.repeated_context);
                result.synthid_skipped.push_back(
                    synthid_trace.skipped_initial_ngram);
              } else {
                auto virtual_scores = log_probability;
                SynthIDWatermarkTrace virtual_trace;
                apply_watermark_logit_policy(
                    virtual_scores, group.live[0].stack, policies->watermark,
                    nullptr, &synthid_rows[row], &virtual_trace);
                result.synthid_context_hashes.push_back(
                    virtual_trace.context_hash);
                result.synthid_repeated.push_back(
                    virtual_trace.repeated_context);
                result.synthid_skipped.push_back(
                    virtual_trace.skipped_initial_ngram);
              }
            }
            first_synthid_call = false;
          }
        }
        // Hamming diversity is appended after configured processors by the
        // pinned grouped-beam implementation. The ordering is observable for
        // SynthID because its probability tournament is nonlinear.
        for (auto token : earlier_group_tokens)
          log_probability[static_cast<std::size_t>(token)] -=
              configuration.diversity_penalty;
        audit.hit(nodes[71]);
        for (std::int32_t token = 0; token < 51864; ++token) {
          const float score = group.live[beam_index].cumulative_log_probability +
                              log_probability[static_cast<std::size_t>(token)];
          candidates.push_back({beam_index, token, score, token == eos_token});
        }
      }
      result.expanded_candidates += candidates.size();
      const auto keep = std::min<std::size_t>(2 * group_size,
                                              candidates.size());
      std::partial_sort(candidates.begin(),
                        candidates.begin() + static_cast<std::ptrdiff_t>(keep),
                        candidates.end(), better_beam_candidate);
      candidates.resize(keep);
      audit.hit(nodes[72]);

      for (std::size_t rank = 0; rank < std::min(group_size, candidates.size());
           ++rank) {
        const auto &candidate = candidates[rank];
        if (!candidate.finished)
          continue;
        BeamSequence completed;
        completed.tokens = group.live[candidate.parent].stack;
        completed.tokens.push_back(candidate.token);
        completed.parent_beams = group.live[candidate.parent].parent_beams;
        completed.parent_beams.push_back(candidate.parent);
        completed.score = length_normalized_beam_score(
            candidate.cumulative_log_probability,
            completed.tokens.size() - decoder_prompt_length,
            configuration.length_penalty);
        group.finished.push_back(std::move(completed));
      }
      std::sort(group.finished.begin(), group.finished.end(),
                [](const auto &left, const auto &right) {
                  return left.score > right.score;
                });
      if (group.finished.size() > group_size)
        group.finished.resize(group_size);

      std::vector<LiveBeam> next;
      next.reserve(group_size);
      for (const auto &candidate : candidates) {
        if (candidate.finished || next.size() == group_size)
          continue;
        LiveBeam branch = group.live[candidate.parent];
        branch.cumulative_log_probability =
            candidate.cumulative_log_probability;
        branch.parent_beams.push_back(candidate.parent);
        branch.stack.push_back(candidate.token);
        audit.hit(nodes[73]);
        branch.next_logits =
            cached.step(candidate.token, memory, branch.state, nullptr);
        earlier_group_tokens.push_back(candidate.token);
        ++result.cache_branches;
        next.push_back(std::move(branch));
      }
      group.live = std::move(next);
      if (!group.live.empty() &&
          group.live.front().stack.size() >= maximum_positions) {
        for (const auto &beam : group.live) {
          BeamSequence completed;
          completed.tokens = beam.stack;
          completed.parent_beams = beam.parent_beams;
          completed.score = length_normalized_beam_score(
              beam.cumulative_log_probability,
              completed.tokens.size() - decoder_prompt_length,
              configuration.length_penalty);
          group.finished.push_back(std::move(completed));
        }
        std::sort(group.finished.begin(), group.finished.end(),
                  [](const auto &left, const auto &right) {
                    return left.score > right.score;
                  });
        if (group.finished.size() > group_size)
          group.finished.resize(group_size);
        group.done = true;
      }
      if (group.live.empty()) {
        group.done = true;
        earlier_group_tokens.insert(earlier_group_tokens.end(), group_size,
                                    generated_whisper::eos_token);
        continue;
      }
      if (std::holds_alternative<whisper_interface::StopWhenAllBeamsFinished>(
              configuration.stopping) &&
          group.finished.size() == group_size)
        group.done = true;
      if (!group.done && group.finished.size() == group_size) {
        const auto generated =
            group.live.front().stack.size() - decoder_prompt_length;
        const auto best_running = length_normalized_beam_score(
            group.live.front().cumulative_log_probability,
            std::holds_alternative<whisper_interface::CanonicalBeamStopping>(
                configuration.stopping) &&
                    configuration.length_penalty > 0.0f
                ? maximum_positions - decoder_prompt_length
                : generated,
            configuration.length_penalty);
        if (best_running <= group.finished.back().score)
          group.done = true;
      }
      if (group.done && earlier_group_tokens.size() % group_size != 0)
        throw std::runtime_error("group beam token transport invariant");
    }
    first_group_position = false;
  }

  for (auto &group : groups)
    for (auto &sequence : group.finished)
      result.sequences.push_back(std::move(sequence));
  std::sort(result.sequences.begin(), result.sequences.end(),
            [](const auto &left, const auto &right) {
              return left.score > right.score;
            });
  if (result.sequences.size() < configuration.return_sequences)
    throw std::runtime_error("group beam finalization coverage");
  result.sequences.resize(configuration.return_sequences);
  const auto tensor_columns = std::max_element(
                                  result.sequences.begin(),
                                  result.sequences.end(),
                                  [](const auto &left, const auto &right) {
                                    return left.tokens.size() <
                                           right.tokens.size();
                                  })
                                  ->tokens.size();
  for (auto &sequence : result.sequences)
    sequence.tokens.resize(tensor_columns, generated_whisper::eos_token);
  return result;
}

struct ConstraintMachine {
  std::size_t constraint = 0;
  std::vector<std::int32_t> progress;
};
struct ConstraintReplayState {
  std::vector<std::size_t> pending;
  std::optional<ConstraintMachine> active;
  std::size_t completed = 0;
  std::size_t bank = 0;
  bool all_completed = false;
  std::vector<std::int32_t> advances;
};

static std::size_t constraint_maximum_length(
    const whisper_interface::PositiveConstraint &constraint) {
  return std::visit(
      [](const auto &value) {
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value, whisper_interface::ForcedPhrase>)
          return value.tokens.size();
        else {
          std::size_t maximum = 0;
          for (const auto &alternative : value.alternatives)
            maximum = std::max(maximum, alternative.tokens.size());
          return maximum;
        }
      },
      constraint);
}

static std::vector<std::int32_t> constraint_advances(
    const whisper_interface::PositiveConstraint &constraint,
    std::span<const std::int32_t> progress) {
  return std::visit(
      [&](const auto &value) {
        using Value = std::decay_t<decltype(value)>;
        std::vector<std::int32_t> tokens;
        if constexpr (std::is_same_v<Value, whisper_interface::ForcedPhrase>) {
          if (progress.size() < value.tokens.size() &&
              std::equal(progress.begin(), progress.end(),
                         value.tokens.begin()))
            tokens.push_back(value.tokens[progress.size()]);
        } else {
          for (const auto &alternative : value.alternatives)
            if (progress.size() < alternative.tokens.size() &&
                std::equal(progress.begin(), progress.end(),
                           alternative.tokens.begin()) &&
                std::find(tokens.begin(), tokens.end(),
                          alternative.tokens[progress.size()]) == tokens.end())
              tokens.push_back(alternative.tokens[progress.size()]);
        }
        return tokens;
      },
      constraint);
}

static bool constraint_progress_complete(
    const whisper_interface::PositiveConstraint &constraint,
    std::span<const std::int32_t> progress) {
  return std::visit(
      [&](const auto &value) {
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value, whisper_interface::ForcedPhrase>)
          return progress.size() == value.tokens.size() &&
                 std::equal(progress.begin(), progress.end(),
                            value.tokens.begin());
        else {
          for (const auto &alternative : value.alternatives)
            if (progress.size() == alternative.tokens.size() &&
                std::equal(progress.begin(), progress.end(),
                           alternative.tokens.begin()))
              return true;
          return false;
        }
      },
      constraint);
}

static ConstraintReplayState replay_constraints(
    std::span<const std::int32_t> sequence,
    const std::vector<whisper_interface::PositiveConstraint> &constraints) {
  if (constraints.empty())
    throw std::runtime_error("constraint replay domain");
  ConstraintReplayState state;
  state.pending.resize(constraints.size());
  std::iota(state.pending.begin(), state.pending.end(), 0);
  const auto maximum = std::transform_reduce(
      constraints.begin(), constraints.end(), std::size_t{0},
      [](auto left, auto right) { return std::max(left, right); },
      constraint_maximum_length);
  for (auto token : sequence) {
    if (state.completed == constraints.size())
      break;
    if (state.active) {
      auto possible = constraint_advances(
          constraints[state.active->constraint], state.active->progress);
      if (std::find(possible.begin(), possible.end(), token) ==
          possible.end()) {
        state.pending.push_back(state.active->constraint);
        state.active.reset();
        continue;
      }
      state.active->progress.push_back(token);
      if (constraint_progress_complete(
              constraints[state.active->constraint], state.active->progress)) {
        ++state.completed;
        state.active.reset();
      }
      continue;
    }
    for (std::size_t pending = 0; pending < state.pending.size(); ++pending) {
      const auto constraint_index = state.pending[pending];
      auto possible = constraint_advances(constraints[constraint_index], {});
      if (std::find(possible.begin(), possible.end(), token) == possible.end())
        continue;
      ConstraintMachine machine{constraint_index, {token}};
      state.pending.erase(state.pending.begin() +
                          static_cast<std::ptrdiff_t>(pending));
      if (constraint_progress_complete(constraints[constraint_index],
                                       machine.progress))
        ++state.completed;
      else
        state.active = std::move(machine);
      break;
    }
  }
  state.all_completed = state.completed == constraints.size();
  state.bank = state.completed * maximum;
  if (state.active) {
    const auto remaining =
        constraint_maximum_length(constraints[state.active->constraint]) -
        state.active->progress.size();
    state.bank += maximum - remaining;
    state.advances = constraint_advances(
        constraints[state.active->constraint], state.active->progress);
  } else if (!state.all_completed) {
    for (auto pending : state.pending) {
      auto possible = constraint_advances(constraints[pending], {});
      state.advances.insert(state.advances.end(), possible.begin(),
                            possible.end());
    }
  }
  return state;
}

struct ConstrainedCandidate {
  std::size_t parent = 0;
  std::int32_t token = 0;
  float cumulative_log_probability = 0.0f;
  std::size_t bank = 0;
  std::size_t insertion_order = 0;
};

static BeamSearchResult constrained_beam_search(
    CachedDecoder &cached, const F32 &memory,
    std::span<const std::int32_t> prefix,
    const whisper_interface::ConstrainedBeamSearch &configuration,
    std::size_t maximum_positions, GraphExecutionAudit &audit,
    const whisper_interface::GenerationLogitPolicies *policies = nullptr) {
  using namespace generated_whisper;
  if (!configuration.valid() || prefix.empty() ||
      prefix.size() >= maximum_positions || maximum_positions > 448)
    throw std::runtime_error("constrained beam search ADT invariant");
  LiveBeam initial;
  for (auto token : prefix) {
    audit.hit(nodes[73]);
    initial.stack.push_back(token);
    initial.next_logits = cached.step(token, memory, initial.state);
  }
  const auto decoder_prompt_length = initial.stack.size();
  std::vector<LiveBeam> live{std::move(initial)};
  std::vector<BeamSequence> finished;
  BeamSearchResult result;
  std::vector<SynthIDWatermarkRuntime> synthid_rows;
  if (policies)
    if (const auto *synthid =
            std::get_if<whisper_interface::SynthIDTextWatermark>(
                &policies->watermark)) {
      synthid_rows.reserve(configuration.beams);
      for (std::size_t row = 0; row < configuration.beams; ++row)
        synthid_rows.emplace_back(*synthid);
    }
  bool first_beam_step = true;
  bool stopped = false;
  while (!live.empty() && !stopped) {
    if (!synthid_rows.empty() && !first_beam_step &&
        live.size() != configuration.beams)
      throw std::runtime_error("SynthID constrained beam-row cardinality");
    std::vector<BeamCandidate> frontier;
    frontier.reserve(live.size() * 51864);
    std::vector<F32> all_scores;
    all_scores.reserve(live.size());
    for (std::size_t beam = 0; beam < live.size(); ++beam) {
      audit.hit(nodes[70]);
      auto scores = log_softmax_logits(live[beam].next_logits);
      if (policies)
        apply_generation_logit_policies(scores, live[beam].stack,
                                        decoder_prompt_length,
                                        maximum_positions, *policies);
      scores = policy_logits(
          scores, live[beam].stack.size() - decoder_prompt_length);
      SynthIDWatermarkTrace synthid_trace;
      if (policies)
        apply_watermark_logit_policy(
            scores, live[beam].stack, policies->watermark, nullptr,
            synthid_rows.empty() ? nullptr : &synthid_rows[beam],
            synthid_rows.empty() ? nullptr : &synthid_trace);
      if (!synthid_rows.empty()) {
        result.synthid_context_hashes.push_back(synthid_trace.context_hash);
        result.synthid_repeated.push_back(synthid_trace.repeated_context);
        result.synthid_skipped.push_back(synthid_trace.skipped_initial_ngram);
        if (first_beam_step) {
          if (beam != 0 || live.size() != 1)
            throw std::runtime_error("SynthID constrained initial beam shape");
          for (std::size_t row = 1; row < configuration.beams; ++row) {
            synthid_rows[row] = synthid_rows[0];
            result.synthid_context_hashes.push_back(
                synthid_trace.context_hash);
            result.synthid_repeated.push_back(
                synthid_trace.repeated_context);
            result.synthid_skipped.push_back(
                synthid_trace.skipped_initial_ngram);
          }
        }
      }
      audit.hit(nodes[71]);
      for (auto &score : scores)
        score += live[beam].cumulative_log_probability;
      for (std::int32_t token = 0; token < 51864; ++token)
        frontier.push_back(
            {beam, token, scores[static_cast<std::size_t>(token)],
             token == eos_token});
      all_scores.push_back(std::move(scores));
    }
    result.expanded_candidates += frontier.size();
    const auto frontier_keep =
        std::min<std::size_t>(2 * configuration.beams, frontier.size());
    std::partial_sort(frontier.begin(),
                      frontier.begin() +
                          static_cast<std::ptrdiff_t>(frontier_keep),
                      frontier.end(), better_beam_candidate);
    frontier.resize(frontier_keep);

    std::vector<ConstrainedCandidate> base;
    base.reserve(configuration.beams);
    for (std::size_t rank = 0; rank < frontier.size(); ++rank) {
      const auto &candidate = frontier[rank];
      if (candidate.token == eos_token) {
        if (rank < configuration.beams &&
            replay_constraints(live[candidate.parent].stack,
                               configuration.constraints)
                .all_completed) {
          BeamSequence completed;
          completed.tokens = live[candidate.parent].stack;
          completed.parent_beams = live[candidate.parent].parent_beams;
          completed.parent_beams.push_back(candidate.parent);
          completed.score = length_normalized_beam_score(
              candidate.cumulative_log_probability,
              completed.tokens.size() + 1 - decoder_prompt_length,
              configuration.length_penalty);
          finished.push_back(std::move(completed));
        }
        continue;
      }
      if (base.size() == configuration.beams)
        continue;
      auto sequence = live[candidate.parent].stack;
      sequence.push_back(candidate.token);
      auto constraint_state =
          replay_constraints(sequence, configuration.constraints);
      base.push_back({candidate.parent, candidate.token,
                      candidate.cumulative_log_probability,
                      constraint_state.bank, base.size()});
    }
    if (base.size() != configuration.beams)
      throw std::runtime_error("constrained beam base frontier");
    std::sort(finished.begin(), finished.end(),
              [](const auto &left, const auto &right) {
                return left.score > right.score;
              });
    if (finished.size() > configuration.beams)
      finished.resize(configuration.beams);

    std::vector<ConstrainedCandidate> merged = base;
    std::vector<std::vector<std::int32_t>> known_sequences;
    known_sequences.reserve(configuration.beams * 2);
    for (const auto &candidate : base) {
      auto sequence = live[candidate.parent].stack;
      sequence.push_back(candidate.token);
      known_sequences.push_back(std::move(sequence));
    }
    std::size_t insertion = base.size();
    for (std::size_t beam = 0; beam < live.size(); ++beam) {
      auto state = replay_constraints(live[beam].stack,
                                      configuration.constraints);
      if (state.all_completed)
        continue;
      for (auto token : state.advances) {
        auto sequence = live[beam].stack;
        sequence.push_back(token);
        if (std::find(known_sequences.begin(), known_sequences.end(),
                      sequence) != known_sequences.end())
          continue;
        auto next_state =
            replay_constraints(sequence, configuration.constraints);
        merged.push_back(
            {beam, token, all_scores[beam][static_cast<std::size_t>(token)],
             next_state.bank, insertion++});
        known_sequences.push_back(std::move(sequence));
      }
    }

    std::vector<std::size_t> order(merged.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](auto left, auto right) {
      const double left_key = double(merged[left].bank) * 100.0 +
                              merged[left].cumulative_log_probability;
      const double right_key = double(merged[right].bank) * 100.0 +
                               merged[right].cumulative_log_probability;
      return left_key > right_key;
    });
    std::map<std::size_t, std::size_t> bank_counts;
    std::vector<std::size_t> within_bank(merged.size());
    for (auto index : order)
      within_bank[index] = bank_counts[merged[index].bank]++;
    std::stable_sort(order.begin(), order.end(), [&](auto left, auto right) {
      return within_bank[left] < within_bank[right];
    });
    order.resize(configuration.beams);
    audit.hit(nodes[72]);

    std::vector<LiveBeam> next;
    next.reserve(configuration.beams);
    for (auto index : order) {
      const auto &candidate = merged[index];
      LiveBeam branch = live[candidate.parent];
      branch.cumulative_log_probability =
          candidate.cumulative_log_probability;
      branch.parent_beams.push_back(candidate.parent);
      branch.stack.push_back(candidate.token);
      audit.hit(nodes[73]);
      branch.next_logits =
          cached.step(candidate.token, memory, branch.state, nullptr);
      ++result.cache_branches;
      next.push_back(std::move(branch));
    }
    live = std::move(next);
    first_beam_step = false;
    if (live.front().stack.size() >= maximum_positions)
      break;
    if (std::holds_alternative<whisper_interface::StopWhenAllBeamsFinished>(
            configuration.stopping) &&
        finished.size() == configuration.beams)
      stopped = true;
    if (!stopped && finished.size() == configuration.beams) {
      const auto generated = live.front().stack.size() - decoder_prompt_length;
      const auto best_running = length_normalized_beam_score(
          frontier.front().cumulative_log_probability,
          std::holds_alternative<whisper_interface::CanonicalBeamStopping>(
              configuration.stopping) &&
                  configuration.length_penalty > 0.0f
              ? maximum_positions - decoder_prompt_length
              : generated,
          configuration.length_penalty);
      if (best_running <= finished.back().score)
        stopped = true;
    }
  }

  std::size_t satisfying_live = 0;
  for (const auto &beam : live)
    if (replay_constraints(beam.stack, configuration.constraints)
            .all_completed) {
      BeamSequence completed{beam.stack,
                             length_normalized_beam_score(
                                 beam.cumulative_log_probability,
                                 beam.stack.size() - decoder_prompt_length,
                                 configuration.length_penalty),
                             beam.parent_beams};
      finished.push_back(std::move(completed));
      ++satisfying_live;
    }
  if (satisfying_live < configuration.return_sequences)
    for (const auto &beam : live)
      if (!replay_constraints(beam.stack, configuration.constraints)
               .all_completed)
        finished.push_back(
            {beam.stack,
             length_normalized_beam_score(
                 beam.cumulative_log_probability,
                 beam.stack.size() - decoder_prompt_length,
                 configuration.length_penalty),
             beam.parent_beams});
  std::sort(finished.begin(), finished.end(),
            [](const auto &left, const auto &right) {
              return left.score > right.score;
            });
  if (finished.size() > configuration.beams)
    finished.resize(configuration.beams);
  if (finished.size() < configuration.return_sequences)
    throw std::runtime_error("constrained beam finalization coverage");
  finished.resize(configuration.return_sequences);

  std::size_t tensor_columns = 0;
  for (const auto &sequence : finished)
    tensor_columns = std::max(
        tensor_columns,
        std::min(maximum_positions, sequence.tokens.size() + std::size_t{1}));
  for (auto &sequence : finished)
    sequence.tokens.resize(tensor_columns, generated_whisper::eos_token);
  result.sequences = std::move(finished);
  return result;
}

struct SampledBeamSearchResult : BeamSearchResult {
  std::size_t sampled_candidates = 0;
  double maximum_conditional_mass_error = 0.0;
  bool every_draw_set_unique = true;
};

static std::vector<BeamCandidate> sample_candidates_without_replacement(
    std::span<const BeamCandidate> candidates, std::size_t count,
    std::mt19937_64 &generator, double &maximum_mass_error) {
  if (count == 0 || count > candidates.size())
    throw std::runtime_error("beam sample count");
  float maximum = -std::numeric_limits<float>::infinity();
  for (const auto &candidate : candidates)
    maximum = std::max(maximum, candidate.cumulative_log_probability);
  std::vector<double> weights(candidates.size());
  for (std::size_t index = 0; index < candidates.size(); ++index)
    if (std::isfinite(candidates[index].cumulative_log_probability))
      weights[index] = std::exp(
          double(candidates[index].cumulative_log_probability - maximum));
  std::vector<BeamCandidate> selected;
  selected.reserve(count);
  for (std::size_t draw_index = 0; draw_index < count; ++draw_index) {
    const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (!(total > 0.0) || !std::isfinite(total))
      throw std::runtime_error("beam sample empty categorical support");
    double normalized_sum = 0.0;
    for (auto weight : weights)
      normalized_sum += weight / total;
    maximum_mass_error =
        std::max(maximum_mass_error, std::abs(normalized_sum - 1.0));
    const double draw = std::generate_canonical<double, 53>(generator) * total;
    double cumulative = 0.0;
    std::size_t chosen = weights.size();
    for (std::size_t index = 0; index < weights.size(); ++index) {
      cumulative += weights[index];
      if (weights[index] > 0.0 && draw < cumulative) {
        chosen = index;
        break;
      }
    }
    if (chosen == weights.size())
      for (std::size_t index = weights.size(); index-- > 0;)
        if (weights[index] > 0.0) {
          chosen = index;
          break;
        }
    if (chosen == weights.size())
      throw std::runtime_error("beam sample selection");
    selected.push_back(candidates[chosen]);
    weights[chosen] = 0.0;
  }
  return selected;
}

static SampledBeamSearchResult sampled_beam_search(
    CachedDecoder &cached, const F32 &memory,
    std::span<const std::int32_t> prefix,
    const whisper_interface::SampledBeamSearch &configuration,
    std::size_t maximum_positions, GraphExecutionAudit &audit,
    const whisper_interface::GenerationLogitPolicies *policies = nullptr) {
  using namespace generated_whisper;
  if (!configuration.valid() || prefix.empty() ||
      prefix.size() >= maximum_positions || maximum_positions > 448)
    throw std::runtime_error("sampled beam search ADT invariant");
  LiveBeam initial;
  for (auto token : prefix) {
    audit.hit(nodes[73]);
    initial.stack.push_back(token);
    initial.next_logits = cached.step(token, memory, initial.state);
  }
  const auto decoder_prompt_length = initial.stack.size();
  std::vector<LiveBeam> live{std::move(initial)};
  std::vector<BeamSequence> finished;
  SampledBeamSearchResult result;
  std::vector<SynthIDWatermarkRuntime> synthid_rows;
  if (policies)
    if (const auto *synthid =
            std::get_if<whisper_interface::SynthIDTextWatermark>(
                &policies->watermark)) {
      synthid_rows.reserve(configuration.beams);
      for (std::size_t row = 0; row < configuration.beams; ++row)
        synthid_rows.emplace_back(*synthid);
    }
  bool first_beam_step = true;
  std::mt19937_64 generator(configuration.seed);
  while (!live.empty()) {
    if (!synthid_rows.empty() && !first_beam_step &&
        live.size() != configuration.beams)
      throw std::runtime_error("SynthID sampled beam-row cardinality");
    std::vector<BeamCandidate> categorical;
    categorical.reserve(live.size() * 51864);
    for (std::size_t beam = 0; beam < live.size(); ++beam) {
      audit.hit(nodes[70]);
      auto scores = log_softmax_logits(live[beam].next_logits);
      if (policies)
        apply_generation_logit_policies(scores, live[beam].stack,
                                        decoder_prompt_length,
                                        maximum_positions, *policies);
      scores = policy_logits(
          scores, live[beam].stack.size() - decoder_prompt_length);
      for (auto &score : scores)
        score /= configuration.temperature;
      apply_sampling_filters(scores, configuration.sampling, 2);
      SynthIDWatermarkTrace synthid_trace;
      if (policies)
        apply_watermark_logit_policy(
            scores, live[beam].stack, policies->watermark, nullptr,
            synthid_rows.empty() ? nullptr : &synthid_rows[beam],
            synthid_rows.empty() ? nullptr : &synthid_trace);
      if (!synthid_rows.empty()) {
        result.synthid_context_hashes.push_back(synthid_trace.context_hash);
        result.synthid_repeated.push_back(synthid_trace.repeated_context);
        result.synthid_skipped.push_back(synthid_trace.skipped_initial_ngram);
        if (first_beam_step) {
          if (beam != 0 || live.size() != 1)
            throw std::runtime_error("SynthID sampled initial beam shape");
          for (std::size_t row = 1; row < configuration.beams; ++row) {
            synthid_rows[row] = synthid_rows[0];
            result.synthid_context_hashes.push_back(
                synthid_trace.context_hash);
            result.synthid_repeated.push_back(
                synthid_trace.repeated_context);
            result.synthid_skipped.push_back(
                synthid_trace.skipped_initial_ngram);
          }
        }
      }
      audit.hit(nodes[71]);
      for (std::int32_t token = 0; token < 51864; ++token)
        categorical.push_back(
            {beam, token,
             live[beam].cumulative_log_probability +
                 scores[static_cast<std::size_t>(token)],
             token == eos_token ||
                 live[beam].stack.size() + 1 >= maximum_positions});
    }
    result.expanded_candidates += categorical.size();
    const auto draw_count =
        std::min<std::size_t>(2 * configuration.beams, categorical.size());
    auto sampled = sample_candidates_without_replacement(
        categorical, draw_count, generator,
        result.maximum_conditional_mass_error);
    result.sampled_candidates += sampled.size();
    {
      std::vector<std::pair<std::size_t, std::int32_t>> identities;
      for (const auto &candidate : sampled)
        identities.emplace_back(candidate.parent, candidate.token);
      std::sort(identities.begin(), identities.end());
      if (std::adjacent_find(identities.begin(), identities.end()) !=
          identities.end())
        result.every_draw_set_unique = false;
    }

    for (std::size_t draw = 0;
         draw < std::min(configuration.beams, sampled.size()); ++draw) {
      const auto &candidate = sampled[draw];
      if (!candidate.finished)
        continue;
      BeamSequence completed;
      completed.tokens = live[candidate.parent].stack;
      completed.tokens.push_back(candidate.token);
      completed.parent_beams = live[candidate.parent].parent_beams;
      completed.parent_beams.push_back(candidate.parent);
      completed.score = length_normalized_beam_score(
          candidate.cumulative_log_probability,
          completed.tokens.size() - decoder_prompt_length,
          configuration.length_penalty);
      finished.push_back(std::move(completed));
    }
    std::sort(finished.begin(), finished.end(),
              [](const auto &left, const auto &right) {
                return left.score > right.score;
              });
    if (finished.size() > configuration.beams)
      finished.resize(configuration.beams);

    std::vector<BeamCandidate> continuations;
    for (const auto &candidate : sampled)
      if (!candidate.finished)
        continuations.push_back(candidate);
    std::sort(continuations.begin(), continuations.end(),
              better_beam_candidate);
    if (continuations.size() > configuration.beams)
      continuations.resize(configuration.beams);
    audit.hit(nodes[72]);
    std::vector<LiveBeam> next;
    next.reserve(continuations.size());
    for (const auto &candidate : continuations) {
      LiveBeam branch = live[candidate.parent];
      branch.cumulative_log_probability =
          candidate.cumulative_log_probability;
      branch.parent_beams.push_back(candidate.parent);
      branch.stack.push_back(candidate.token);
      audit.hit(nodes[73]);
      branch.next_logits =
          cached.step(candidate.token, memory, branch.state, nullptr);
      ++result.cache_branches;
      next.push_back(std::move(branch));
    }
    live = std::move(next);
    first_beam_step = false;
    if (live.empty())
      break;
    if (std::holds_alternative<whisper_interface::StopWhenAllBeamsFinished>(
            configuration.stopping) &&
        finished.size() == configuration.beams)
      break;
    if (finished.size() == configuration.beams) {
      const auto generated = live.front().stack.size() - decoder_prompt_length;
      const auto best_running = length_normalized_beam_score(
          live.front().cumulative_log_probability,
          std::holds_alternative<whisper_interface::CanonicalBeamStopping>(
              configuration.stopping) &&
                  configuration.length_penalty > 0.0f
              ? maximum_positions - decoder_prompt_length
              : generated,
          configuration.length_penalty);
      if (best_running <= finished.back().score)
        break;
    }
  }
  if (finished.size() < configuration.return_sequences)
    throw std::runtime_error("sampled beam finalization coverage");
  finished.resize(configuration.return_sequences);
  std::size_t tensor_columns = 0;
  for (const auto &sequence : finished)
    tensor_columns = std::max(tensor_columns, sequence.tokens.size());
  for (auto &sequence : finished)
    sequence.tokens.resize(tensor_columns, generated_whisper::eos_token);
  result.sequences = std::move(finished);
  return result;
}

struct ContrastiveSearchResult {
  std::vector<std::int32_t> tokens;
  std::size_t candidate_branches = 0;
  std::size_t cosine_edges = 0;
  float maximum_cosine = -1.0f;
  std::vector<std::int32_t> first_candidate_tokens;
  F32 first_probabilities, first_degenerations, first_scores;
  std::size_t first_selected_rank = 0;
};

static ContrastiveSearchResult contrastive_search(
    CachedDecoder &cached, const F32 &memory,
    std::span<const std::int32_t> prefix,
    const whisper_interface::ContrastiveSearch &configuration,
    std::size_t maximum_positions, GraphExecutionAudit &audit,
    const whisper_interface::GenerationLogitPolicies *policies = nullptr) {
  using namespace generated_whisper;
  if (!configuration.valid() || prefix.empty() ||
      prefix.size() >= maximum_positions || maximum_positions > 448)
    throw std::runtime_error("contrastive search ADT invariant");

  ContrastiveSearchResult result;
  result.tokens.assign(prefix.begin(), prefix.end());
  DecoderKVState state;
  F32 context_hidden;
  F32 next_logits;
  for (auto token : prefix) {
    F32 hidden;
    audit.hit(nodes[73]);
    next_logits = cached.step(token, memory, state, nullptr, &hidden);
    if (hidden.size() != 384)
      throw std::runtime_error("contrastive prefix hidden shape");
    context_hidden.insert(context_hidden.end(), hidden.begin(), hidden.end());
  }
  const auto decoder_prompt_length = result.tokens.size();

  while (result.tokens.size() < maximum_positions) {
    audit.hit(nodes[70]);
    F32 processed = next_logits;
    if (policies)
      apply_generation_logit_policies(
          processed, result.tokens, decoder_prompt_length, maximum_positions,
          *policies);
    processed = policy_logits(
        processed, result.tokens.size() - decoder_prompt_length);
    if (policies)
      apply_watermark_logit_policy(processed, result.tokens,
                                   policies->watermark);
    audit.hit(nodes[71]);
    softmax_rows(processed, 1, processed.size());

    std::vector<std::size_t> candidates(processed.size());
    std::iota(candidates.begin(), candidates.end(), 0);
    std::partial_sort(
        candidates.begin(), candidates.begin() + configuration.candidates,
        candidates.end(), [&](auto left, auto right) {
          return processed[left] > processed[right];
        });
    candidates.resize(configuration.candidates);

    float best_score = -std::numeric_limits<float>::infinity();
    std::int32_t best_token = -1;
    DecoderKVState best_state;
    F32 best_hidden, best_logits;
    std::size_t best_rank = 0;
    const auto context_positions = context_hidden.size() / 384;
    const bool capture_first_decision = result.candidate_branches == 0;
    for (std::size_t rank = 0; rank < candidates.size(); ++rank) {
      const auto token_index = candidates[rank];
      DecoderKVState candidate_state = state;
      F32 candidate_hidden;
      audit.hit(nodes[73]);
      auto candidate_logits = cached.step(
          static_cast<std::int32_t>(token_index), memory, candidate_state,
          nullptr, &candidate_hidden);
      if (candidate_hidden.size() != 384 || context_positions == 0)
        throw std::runtime_error("contrastive candidate hidden shape");
      const float candidate_norm =
          std::sqrt(cblas_sdot(384, candidate_hidden.data(), 1,
                              candidate_hidden.data(), 1));
      if (!(candidate_norm > 0.0f))
        throw std::runtime_error("contrastive candidate zero norm");
      float degeneration = -std::numeric_limits<float>::infinity();
      for (std::size_t position = 0; position < context_positions;
           ++position) {
        const auto *context = context_hidden.data() + position * 384;
        const float context_norm =
            std::sqrt(cblas_sdot(384, context, 1, context, 1));
        if (!(context_norm > 0.0f))
          throw std::runtime_error("contrastive context zero norm");
        const float cosine =
            cblas_sdot(384, context, 1, candidate_hidden.data(), 1) /
            (context_norm * candidate_norm);
        degeneration = std::max(degeneration, cosine);
        result.maximum_cosine = std::max(result.maximum_cosine, cosine);
        ++result.cosine_edges;
      }
      const float score =
          (1.0f - configuration.degeneration_penalty) *
              processed[token_index] -
          configuration.degeneration_penalty * degeneration;
      if (capture_first_decision) {
        result.first_candidate_tokens.push_back(
            static_cast<std::int32_t>(token_index));
        result.first_probabilities.push_back(processed[token_index]);
        result.first_degenerations.push_back(degeneration);
        result.first_scores.push_back(score);
      }
      ++result.candidate_branches;
      if (score > best_score) {
        best_score = score;
        best_token = static_cast<std::int32_t>(token_index);
        best_state = std::move(candidate_state);
        best_hidden = std::move(candidate_hidden);
        best_logits = std::move(candidate_logits);
        best_rank = rank;
      }
    }
    if (best_token < 0)
      throw std::runtime_error("contrastive candidate selection");
    audit.hit(nodes[72]);
    if (capture_first_decision)
      result.first_selected_rank = best_rank;
    result.tokens.push_back(best_token);
    state = std::move(best_state);
    next_logits = std::move(best_logits);
    context_hidden.insert(context_hidden.end(), best_hidden.begin(),
                          best_hidden.end());
    if (best_token == eos_token)
      break;
  }
  return result;
}

struct PromptLookupResult {
  std::vector<std::int32_t> tokens;
  std::size_t proposal_rounds = 0, proposed_tokens = 0,
              accepted_candidate_tokens = 0, correction_tokens = 0,
              target_evaluations = 0;
  std::vector<std::int32_t> first_proposal;
  std::size_t first_accepted = 0;
  bool terminated_by_eos = false;
};

static std::vector<std::int32_t> prompt_lookup_proposal(
    std::span<const std::int32_t> stack,
    const whisper_interface::PromptLookupSearch &configuration,
    std::size_t maximum_positions) {
  if (!configuration.valid() || stack.size() < 2 ||
      stack.size() >= maximum_positions)
    return {};
  const auto maximum_ngram =
      std::min(configuration.maximum_matching_ngram, stack.size() - 1);
  for (std::size_t ngram = maximum_ngram; ngram > 0; --ngram) {
    const auto suffix_begin = stack.size() - ngram;
    for (std::size_t match = 0; match + ngram <= stack.size(); ++match) {
      if (!std::equal(stack.begin() + static_cast<std::ptrdiff_t>(match),
                      stack.begin() +
                          static_cast<std::ptrdiff_t>(match + ngram),
                      stack.begin() +
                          static_cast<std::ptrdiff_t>(suffix_begin)))
        continue;
      const auto begin = match + ngram;
      const auto end = std::min(
          {begin + configuration.proposal_tokens, stack.size(),
           maximum_positions});
      if (begin >= end)
        continue;
      std::vector<std::int32_t> proposal(
          stack.begin() + static_cast<std::ptrdiff_t>(begin),
          stack.begin() + static_cast<std::ptrdiff_t>(end));
      const auto eos = std::find(proposal.begin(), proposal.end(),
                                 generated_whisper::eos_token);
      proposal.erase(eos, proposal.end());
      if (!proposal.empty())
        return proposal;
    }
  }
  return {};
}

static PromptLookupResult prompt_lookup_search(
    CachedDecoder &cached, const F32 &memory,
    std::span<const std::int32_t> initial_stack,
    const whisper_interface::PromptLookupSearch &configuration,
    std::size_t maximum_positions, GraphExecutionAudit &audit,
    const whisper_interface::GenerationLogitPolicies *policies = nullptr) {
  using namespace generated_whisper;
  if (!configuration.valid() || initial_stack.empty() ||
      initial_stack.size() >= maximum_positions || maximum_positions > 448)
    throw std::runtime_error("prompt lookup ADT invariant");
  PromptLookupResult result;
  std::vector<std::int32_t> stack(initial_stack.begin(), initial_stack.end());
  DecoderKVState state;
  F32 next_logits;
  for (auto token : stack) {
    audit.hit(nodes[73]);
    next_logits = cached.step(token, memory, state);
  }
  const auto begin_index = stack.size();
  auto target_token = [&]() {
    audit.hit(nodes[70]);
    F32 scores = next_logits;
    if (policies)
      apply_generation_logit_policies(scores, stack, begin_index,
                                      maximum_positions, *policies);
    scores = policy_logits(scores, stack.size() - begin_index);
    if (policies)
      apply_watermark_logit_policy(scores, stack, policies->watermark);
    audit.hit(nodes[71]);
    softmax_rows(scores, 1, scores.size());
    audit.hit(nodes[72]);
    ++result.target_evaluations;
    return static_cast<std::int32_t>(
        std::max_element(scores.begin(), scores.end()) - scores.begin());
  };
  auto commit = [&](std::int32_t token) {
    result.tokens.push_back(token);
    stack.push_back(token);
    audit.hit(nodes[73]);
    next_logits = cached.step(token, memory, state);
  };

  while (state.position < maximum_positions) {
    auto proposal =
        prompt_lookup_proposal(stack, configuration, maximum_positions);
    const bool capture_first_proposal =
        result.first_proposal.empty() && !proposal.empty();
    ++result.proposal_rounds;
    result.proposed_tokens += proposal.size();
    if (capture_first_proposal)
      result.first_proposal = proposal;
    std::size_t accepted_this_round = 0;
    bool corrected = false;
    for (auto candidate : proposal) {
      const auto target = target_token();
      if (target != candidate) {
        ++result.correction_tokens;
        corrected = true;
        if (target == eos_token) {
          result.terminated_by_eos = true;
          break;
        }
        commit(target);
        break;
      }
      ++result.accepted_candidate_tokens;
      ++accepted_this_round;
      if (candidate == eos_token) {
        result.terminated_by_eos = true;
        break;
      }
      commit(candidate);
      if (state.position >= maximum_positions)
        break;
    }
    if (capture_first_proposal)
      result.first_accepted = accepted_this_round;
    if (result.terminated_by_eos || state.position >= maximum_positions)
      break;
    if (!proposal.empty() && !corrected &&
        accepted_this_round == proposal.size()) {
      const auto extra = target_token();
      ++result.correction_tokens;
      if (extra == eos_token) {
        result.terminated_by_eos = true;
        break;
      }
      commit(extra);
    } else if (proposal.empty()) {
      const auto target = target_token();
      ++result.correction_tokens;
      if (target == eos_token) {
        result.terminated_by_eos = true;
        break;
      }
      commit(target);
    }
  }
  return result;
}
struct FallbackObservation {
  std::size_t seek = 0, attempt = 0;
  float temperature = 0;
  double compression = 0, average_logprob = 0, no_speech_probability = 0;
  bool needs_fallback = false, should_skip = false;
};
struct LongFormExecution {
  std::vector<LongFormSegment> segments;
  std::vector<std::size_t> seeks;
  std::vector<FallbackObservation> fallback;
  std::size_t total_frames = 0, generation_calls = 0, skipped_windows = 0;
};
static LongFormExecution execute_long_form(
    TensorStore &tensors, const std::string &wav, const F32 &window,
    const F32 &filters, bool condition_on_previous,
    const std::vector<std::int32_t> &prompt,
    const whisper_interface::PromptConditionType &prompt_condition,
    const whisper_interface::MonitorProgress *monitor,
    const whisper_interface::FallbackThresholds *fallback,
    GraphExecutionAudit &execution) {
  const bool all_segments =
      std::holds_alternative<whisper_interface::AllSegmentsPrompt>(
          prompt_condition);
  if (all_segments && !condition_on_previous)
    throw std::runtime_error(
        "all-segments prompt requires previous conditioning");
  execution.hit(generated_whisper::nodes[0]);
  auto pcm = read_wav_pcm16(wav);
  execution.hit(generated_whisper::nodes[1]);
  auto full_mel = log_mel_unpadded(pcm, window, filters);
  LongFormExecution result;
  result.total_frames = full_mel.size() / 80;
  whisper_interface::LongFormWindowing windowing{
      {result.total_frames, result.total_frames},
      {3000},
      condition_on_previous};
  if (!windowing.valid())
    throw std::runtime_error("long-form window ADT invariant");
  result.seeks.push_back(0);
  Encoder encoder(tensors, execution);
  Decoder decoder(tensors, execution);
  whisper_interface::Segments time_output{0.02f, 0.01f};
  auto timestamp_prefix =
      std::span<const std::int32_t>(generated_whisper::forced_prefix).first(1);
  std::size_t seek = 0;
  bool do_condition_on_previous = condition_on_previous;
  while (seek < result.total_frames) {
    if (monitor) {
      if (!monitor->valid())
        throw std::runtime_error("progress callback");
      monitor->notify(seek, result.total_frames);
    }
    const auto seek_frames =
        std::min<std::size_t>(3000, result.total_frames - seek);
    auto mel = mel_window(full_mel, result.total_frames, seek);
    std::vector<std::pair<std::string, F32>> trace;
    auto memory = encoder.run(mel, trace);
    trace.clear();
    std::vector<std::int32_t> decoder_history;
    if (!prompt.empty() && !all_segments && do_condition_on_previous) {
      auto conditioning_segments = result.segments;
      conditioning_segments.insert(conditioning_segments.begin(),
                                   LongFormSegment{0, 0, prompt});
      decoder_history = previous_segment_prefix(conditioning_segments);
    } else if (result.generation_calls == 0)
      decoder_history = prompt;
    else if (do_condition_on_previous)
      decoder_history = previous_segment_prefix(
          result.segments, all_segments ? &prompt : nullptr);
    else if (!prompt.empty())
      decoder_history = prompt;
    if (fallback && !fallback->valid())
      throw std::runtime_error("fallback ADT invariant");
    const std::vector<float> default_temperatures{0.0f};
    const auto &temperatures =
        fallback ? fallback->temperatures : default_temperatures;
    GenerationResult generated;
    bool should_skip = false;
    float accepted_temperature = temperatures.back();
    for (std::size_t attempt = 0; attempt < temperatures.size(); ++attempt) {
      const float temperature = temperatures[attempt];
      CachedDecoder cached(tensors, execution);
      generated =
          generate(decoder, cached, memory,
                   temperature > 0 ? Selection::Sample : Selection::Greedy,
                   temperature, 0x5eed + attempt, false, decoder_history,
                   timestamp_prefix, time_output, nullptr, nullptr, execution);
      const double ratio =
          compression_ratio(generated.tokens, generated.terminated_by_eos);
      const double average = generated.average_logprob();
      bool needs =
          fallback &&
          ((fallback->compression_ratio &&
            ratio > *fallback->compression_ratio) ||
           (fallback->average_logprob && average < *fallback->average_logprob));
      should_skip =
          fallback && fallback->no_speech_probability &&
          average < *fallback->average_logprob &&
          generated.no_speech_probability > *fallback->no_speech_probability;
      if (should_skip)
        needs = false;
      result.fallback.push_back({seek, attempt, temperature, ratio, average,
                                 generated.no_speech_probability, needs,
                                 should_skip});
      accepted_temperature = temperature;
      if (!needs || attempt + 1 == temperatures.size())
        break;
    }
    do_condition_on_previous =
        condition_on_previous && accepted_temperature < 0.5f;
    if (should_skip) {
      seek += seek_frames;
      result.seeks.push_back(seek);
      ++result.generation_calls;
      ++result.skipped_windows;
      continue;
    }
    auto local_segments = timestamp_segments(
        generated.tokens, time_output.seconds_per_timestamp_token,
        (seek_frames / 2) * time_output.seconds_per_timestamp_token);
    const double time_offset = seek * time_output.seconds_per_feature_frame;
    for (const auto &local : local_segments) {
      LongFormSegment segment;
      segment.start_seconds = time_offset + local.start_seconds;
      segment.end_seconds = time_offset + local.end_seconds;
      segment.tokens.assign(generated.tokens.begin() +
                                static_cast<std::ptrdiff_t>(local.token_begin),
                            generated.tokens.begin() +
                                static_cast<std::ptrdiff_t>(local.token_end));
      result.segments.push_back(std::move(segment));
    }
    auto offset = timestamp_seek_offset(generated.tokens, seek_frames);
    if (offset == 0 || offset > seek_frames)
      throw std::runtime_error("long-form seek transition");
    seek += offset;
    result.seeks.push_back(seek);
    ++result.generation_calls;
  }
  return result;
}
struct Error {
  double max_abs = 0, rmse = 0, cosine = 0;
};
class TokenDecoder {
  struct Entry {
    std::uint32_t begin = 0, end = 0;
    bool special = false;
  };
  std::vector<Entry> entries_;
  std::vector<char> bytes_;

public:
  TokenDecoder(const std::string &manifest, const std::string &blob) {
    std::ifstream in(manifest);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      auto f = split(line);
      if (f.size() != 4)
        throw std::runtime_error("token manifest row");
      auto id = std::stoul(f[0]);
      if (id != entries_.size())
        throw std::runtime_error("token id order");
      entries_.push_back({std::uint32_t(std::stoul(f[1])),
                          std::uint32_t(std::stoul(f[2])),
                          std::stoi(f[3]) != 0});
    }
    std::ifstream bin(blob, std::ios::binary | std::ios::ate);
    auto n = static_cast<std::size_t>(bin.tellg());
    bin.seekg(0);
    bytes_.resize(n);
    bin.read(bytes_.data(), static_cast<std::streamsize>(n));
    if (entries_.size() != 51864 || !bin)
      throw std::runtime_error("token decoder load");
  }
  std::string decode(std::span<const std::int32_t> ids) const {
    std::string out;
    for (auto id : ids) {
      if (id < 0 || id >= static_cast<std::int32_t>(entries_.size()))
        throw std::runtime_error("decode token id");
      auto e = entries_[id];
      if (!e.special)
        out.append(bytes_.data() + e.begin, e.end - e.begin);
    }
    return out;
  }
};

static bool stop_string_overlaps_last_token(
    const TokenDecoder &decoder,
    const whisper_interface::StopStringSet &stop_strings,
    std::span<const std::int32_t> prior_tokens, std::int32_t final_token) {
  if (!stop_strings.valid())
    throw std::runtime_error("stop string ADT invariant");
  const std::string before = decoder.decode(prior_tokens);
  const std::array<std::int32_t, 1> singleton{final_token};
  std::string text = before;
  text += decoder.decode(singleton);
  const auto boundary = before.size();
  for (const auto &stop : stop_strings.values) {
    for (std::size_t position = text.find(stop); position != std::string::npos;
         position = text.find(stop, position + 1))
      if (position + stop.size() > boundary)
        return true;
  }
  return false;
}

struct ShortFormTranscription {
  std::vector<std::int32_t> tokens;
  std::string text;
  double maximum_cache_logit_error = 0;
  std::size_t graph_nodes_visited = 0;
};

static std::string trim_ascii_space(std::string text) {
  while (!text.empty() &&
         std::isspace(static_cast<unsigned char>(text.front())))
    text.erase(text.begin());
  while (!text.empty() &&
         std::isspace(static_cast<unsigned char>(text.back())))
    text.pop_back();
  return text;
}

static ShortFormTranscription transcribe_short_form_item(
    TensorStore &tensors, const std::string &wav, const F32 &window,
    const F32 &filters, const TokenDecoder &token_decoder,
    bool verify_recompute) {
  GraphExecutionAudit execution;
  auto mel = execute_frontend(wav, window, filters, execution);
  std::vector<std::pair<std::string, F32>> trace;
  Encoder encoder(tensors, execution);
  auto memory = encoder.run(mel, trace);
  Decoder decoder(tensors, execution);
  CachedDecoder cached(tensors, execution);
  auto generated = generate(
      decoder, cached, memory, Selection::Greedy, 0.0f, 0, verify_recompute,
      {}, generated_whisper::forced_prefix, whisper_interface::NoTimestamps{},
      nullptr, nullptr, execution);
  execution.require_all();
  auto text = trim_ascii_space(token_decoder.decode(generated.tokens));
  return {std::move(generated.tokens), std::move(text),
          generated.max_cache_logit_error, execution.visited()};
}
static Error compare(std::span<const float> a, std::span<const float> b) {
  if (a.size() != b.size())
    throw std::runtime_error("reference shape");
  double sq = 0, dot = 0, aa = 0, bb = 0, m = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    double d = double(a[i]) - b[i];
    m = std::max(m, std::abs(d));
    sq += d * d;
    dot += double(a[i]) * b[i];
    aa += double(a[i]) * a[i];
    bb += double(b[i]) * b[i];
  }
  return {m, std::sqrt(sq / a.size()), dot / std::sqrt(aa * bb)};
}
static void audit_graph(TensorStore &t) {
  using namespace generated_whisper;
  if (nodes.size() != 74 || tensor_names.size() != 167)
    throw std::runtime_error("generated graph cardinality");
  if (whisper_generation_config::pinned_generation_fields.size() != 74 ||
      whisper_generation_config::transformers_version != "4.57.3")
    throw std::runtime_error("generation configuration inventory");
  std::map<std::string_view, int> seen;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const auto &n = nodes[i];
    if (n.index != i || n.input_begin + n.input_count > ports.size() ||
        n.output_begin + n.output_count > ports.size() ||
        n.weight_begin + n.weight_count > weight_refs.size())
      throw std::runtime_error("generated node bounds");
    for (std::size_t j = 0; j < n.weight_count; ++j)
      ++seen[weight_refs[n.weight_begin + j]];
  }
  for (auto name : tensor_names)
    if (!t.contains(name) || !seen.contains(name))
      throw std::runtime_error("unbound graph tensor " + std::string(name));
  if (seen.size() != tensor_names.size())
    throw std::runtime_error("extra graph tensor");
  whisper_interface::GenerationLogitPolicies repair_policy{};
  repair_policy.invalid_logits = whisper_interface::RepairInvalidLogits{};
  F32 repair_probe(51864, 0.0f);
  repair_probe[0] = std::numeric_limits<float>::quiet_NaN();
  repair_probe[1] = std::numeric_limits<float>::infinity();
  repair_probe[2] = -std::numeric_limits<float>::infinity();
  apply_generation_logit_policies(repair_probe, forced_prefix,
                                  forced_prefix.size(), 448, repair_policy);
  if (repair_probe[0] != 0.0f ||
      repair_probe[1] != std::numeric_limits<float>::max() ||
      repair_probe[2] != std::numeric_limits<float>::lowest())
    throw std::runtime_error("invalid-logit repair semantics");
}
} // namespace whisper_graph

int main(int argc, char **argv) try {
  using namespace whisper_graph;
  if (argc == 16 && std::string(argv[1]) == "--watermark-mass") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    const auto watermark =
        parse_watermark_policy(argv[9], argv[10], argv[11], argv[12], argv[13]);
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    CachedDecoder cached(tensors, execution);
    DecoderKVState state;
    F32 logits;
    for (auto token : generated_whisper::forced_prefix) {
      execution.hit(generated_whisper::nodes[73]);
      logits = cached.step(token, memory, state);
    }
    execution.hit(generated_whisper::nodes[70]);
    auto mass = policy_logits(logits, 0);
    std::vector<std::uint8_t> green_mask;
    const auto green = apply_watermark_logit_policy(
        mass, generated_whisper::forced_prefix, watermark, &green_mask);
    execution.hit(generated_whisper::nodes[71]);
    softmax_rows(mass, 1, mass.size());
    execution.require_range(0, 72);
    if (execution.visited() != 73)
      throw std::runtime_error("watermark mass graph coverage");
    write_f32(argv[14], mass);
    write_u8(argv[15], green_mask);
    const auto selected =
        std::max_element(mass.begin(), mass.end()) - mass.begin();
    std::cout << "WHISPER_CPP23_WATERMARK_MASS scheme=" << argv[9]
              << " green=" << green << " selected=" << selected
              << " sum=" << std::setprecision(17)
              << std::accumulate(mass.begin(), mass.end(), 0.0)
              << " graph_nodes_visited=" << execution.visited() << "\n";
    return 0;
  }
  if (argc == 15 && std::string(argv[1]) == "--transcribe-watermark") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    whisper_interface::GenerationLogitPolicies policies{};
    policies.watermark =
        parse_watermark_policy(argv[9], argv[10], argv[11], argv[12], argv[13]);
    const auto maximum_positions = parse_optional_size(argv[14]);
    if (!maximum_positions || !policies.valid() ||
        *maximum_positions <= generated_whisper::forced_prefix.size())
      throw std::runtime_error("watermark generation request ADT invariant");
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    Decoder decoder(tensors, execution);
    CachedDecoder cached(tensors, execution);
    auto generated = generate(
        decoder, cached, memory, Selection::Greedy, 0.0f, 0, false, {},
        generated_whisper::forced_prefix, whisper_interface::NoTimestamps{},
        nullptr, nullptr, execution, &policies, *maximum_positions);
    execution.require_all();
    std::cout << "WHISPER_CPP23_WATERMARK_TRANSCRIPT scheme=" << argv[9]
              << " tokens=";
    for (std::size_t index = 0; index < generated.tokens.size(); ++index) {
      if (index)
        std::cout << ',';
      std::cout << generated.tokens[index];
    }
    std::cout << " terminated_by_eos=" << generated.terminated_by_eos
              << " graph_nodes_visited=" << execution.visited() << "\n";
    return 0;
  }
  if (argc == 18 && std::string(argv[1]) == "--synthid-mass") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    const auto policy = parse_synthid_watermark_policy(
        argv[9], argv[10], argv[11], argv[12], argv[13], argv[14], argv[15]);
    const auto *configuration =
        std::get_if<whisper_interface::SynthIDTextWatermark>(&policy);
    if (!configuration)
      throw std::runtime_error("SynthID policy constructor");
    SynthIDWatermarkRuntime runtime(*configuration);
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    CachedDecoder cached(tensors, execution);
    DecoderKVState state;
    F32 logits;
    for (auto token : generated_whisper::forced_prefix) {
      execution.hit(generated_whisper::nodes[73]);
      logits = cached.step(token, memory, state);
    }
    execution.hit(generated_whisper::nodes[70]);
    auto mass = policy_logits(logits, 0);
    SynthIDWatermarkTrace synthid_trace;
    synthid_trace.capture_g_values = true;
    const auto ones = apply_watermark_logit_policy(
        mass, generated_whisper::forced_prefix, policy, nullptr, &runtime,
        &synthid_trace);
    execution.hit(generated_whisper::nodes[71]);
    softmax_rows(mass, 1, mass.size());
    execution.require_range(0, 72);
    if (execution.visited() != 73)
      throw std::runtime_error("SynthID mass graph coverage");
    write_f32(argv[16], mass);
    write_u8(argv[17], synthid_trace.g_values);
    const auto selected =
        std::max_element(mass.begin(), mass.end()) - mass.begin();
    std::cout << "WHISPER_CPP23_SYNTHID_MASS call=" << synthid_trace.call
              << " context_hash=" << synthid_trace.context_hash
              << " repeated=" << synthid_trace.repeated_context
              << " skipped=" << synthid_trace.skipped_initial_ngram
              << " g_ones=" << ones << " selected=" << selected
              << " sum=" << std::setprecision(17)
              << std::accumulate(mass.begin(), mass.end(), 0.0)
              << " graph_nodes_visited=" << execution.visited() << "\n";
    return 0;
  }
  if (argc == 17 && std::string(argv[1]) == "--transcribe-synthid") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    whisper_interface::GenerationLogitPolicies policies{};
    policies.watermark = parse_synthid_watermark_policy(
        argv[9], argv[10], argv[11], argv[12], argv[13], argv[14], argv[15]);
    const auto maximum_positions = parse_optional_size(argv[16]);
    if (!maximum_positions || !policies.valid() ||
        *maximum_positions <= generated_whisper::forced_prefix.size())
      throw std::runtime_error("SynthID generation request ADT invariant");
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    Decoder decoder(tensors, execution);
    CachedDecoder cached(tensors, execution);
    std::vector<SynthIDWatermarkTrace> synthid_traces;
    auto generated = generate(
        decoder, cached, memory, Selection::Greedy, 0.0f, 0, false, {},
        generated_whisper::forced_prefix, whisper_interface::NoTimestamps{},
        nullptr, nullptr, execution, &policies, *maximum_positions, nullptr,
        &synthid_traces);
    execution.require_all();
    std::cout << "WHISPER_CPP23_SYNTHID_TRANSCRIPT tokens=";
    for (std::size_t index = 0; index < generated.tokens.size(); ++index) {
      if (index)
        std::cout << ',';
      std::cout << generated.tokens[index];
    }
    std::cout << " calls=" << synthid_traces.size() << " context_hashes=";
    for (std::size_t index = 0; index < synthid_traces.size(); ++index) {
      if (index)
        std::cout << ',';
      std::cout << synthid_traces[index].context_hash;
    }
    std::cout << " repeated=";
    for (const auto &item : synthid_traces)
      std::cout << (item.repeated_context ? '1' : '0');
    std::cout << " skipped=";
    for (const auto &item : synthid_traces)
      std::cout << (item.skipped_initial_ngram ? '1' : '0');
    std::cout << " terminated_by_eos=" << generated.terminated_by_eos
              << " graph_nodes_visited=" << execution.visited() << "\n";
    return 0;
  }
  if (argc == 13 && std::string(argv[1]) == "--prompt-lookup") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    auto initial = parse_token_ids(argv[9]);
    const auto proposal_tokens = parse_optional_size(argv[10]);
    const auto maximum_ngram = parse_optional_size(argv[11]);
    const auto maximum_positions = parse_optional_size(argv[12]);
    if (!proposal_tokens || !maximum_ngram || !maximum_positions)
      throw std::runtime_error("prompt lookup arguments");
    const whisper_interface::PromptLookupSearch configuration{
        *proposal_tokens, *maximum_ngram};
    if (!configuration.valid())
      throw std::runtime_error("prompt lookup configuration ADT invariant");
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    CachedDecoder cached(tensors, execution);
    auto result = prompt_lookup_search(cached, memory, initial, configuration,
                                       *maximum_positions, execution);
    execution.require_all();
    TokenDecoder token_decoder(argv[7], argv[8]);
    std::cout << "WHISPER_CPP23_PROMPT_LOOKUP initial_positions="
              << initial.size() << " tokens=";
    for (std::size_t index = 0; index < result.tokens.size(); ++index) {
      if (index)
        std::cout << ',';
      std::cout << result.tokens[index];
    }
    std::cout << " proposal_rounds=" << result.proposal_rounds
              << " proposed_tokens=" << result.proposed_tokens
              << " accepted_candidates="
              << result.accepted_candidate_tokens
              << " correction_tokens=" << result.correction_tokens
              << " target_evaluations=" << result.target_evaluations
              << " first_proposal=";
    for (std::size_t index = 0; index < result.first_proposal.size(); ++index) {
      if (index)
        std::cout << ',';
      std::cout << result.first_proposal[index];
    }
    std::cout << " first_accepted=" << result.first_accepted
              << " terminated_by_eos=" << result.terminated_by_eos
              << " graph_nodes_visited=" << execution.visited()
              << " text=" << std::quoted(token_decoder.decode(result.tokens))
              << "\n";
    return 0;
  }
  if (argc == 11 && std::string(argv[1]) == "--stop-string-search") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    const auto maximum_positions = parse_optional_size(argv[9]);
    const whisper_interface::StopStringSet stop_strings{split(argv[10], '|')};
    if (!maximum_positions || !stop_strings.valid())
      throw std::runtime_error("stop string search arguments");
    const auto window = read_f32(argv[5]);
    const auto filters = read_f32(argv[6]);
    TokenDecoder token_decoder(argv[7], argv[8]);
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], window, filters, execution);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    Decoder decoder(tensors, execution);
    CachedDecoder cached(tensors, execution);
    const StopStringPredicate matcher =
        [&](std::span<const std::int32_t> prior, std::int32_t token) {
          return stop_string_overlaps_last_token(token_decoder, stop_strings,
                                                 prior, token);
        };
    auto generated = generate(
        decoder, cached, memory, Selection::Greedy, 0.0f, 0, false, {},
        generated_whisper::forced_prefix, whisper_interface::NoTimestamps{},
        nullptr, nullptr, execution, nullptr, *maximum_positions, &matcher);
    execution.require_all();
    std::cout << "WHISPER_CPP23_STOP_STRING tokens=";
    for (std::size_t index = 0; index < generated.tokens.size(); ++index) {
      if (index)
        std::cout << ',';
      std::cout << generated.tokens[index];
    }
    std::cout << " terminated_by_stop_string="
              << generated.terminated_by_stop_string
              << " terminated_by_eos=" << generated.terminated_by_eos
              << " graph_nodes_visited=" << execution.visited()
              << " text=" << std::quoted(token_decoder.decode(generated.tokens))
              << "\n";
    return 0;
  }
  if (argc == 4 && std::string(argv[1]) == "--generation-applicability") {
    const std::string field = argv[2], value = argv[3];
    if (field == "dola_layers") {
      const whisper_interface::RejectDolaForEncoderDecoder rejection{
          parse_dola_layers(value)};
      if (!rejection.valid())
        throw std::runtime_error("DoLa rejection ADT invariant");
      std::cout << "WHISPER_CPP23_GENERATION_APPLICABILITY field=" << field
                << " status=MODEL_REJECTED reason=encoder_decoder_model\n";
      return 0;
    }
    if (field == "guidance_scale") {
      const auto scale = parse_optional_f32(value);
      if (!scale)
        throw std::runtime_error("guidance scale argument");
      const whisper_interface::RejectUnbatchedGuidanceForMelEncoder rejection{
          *scale};
      if (!rejection.valid())
        throw std::runtime_error("guidance rejection ADT invariant");
      std::cout << "WHISPER_CPP23_GENERATION_APPLICABILITY field=" << field
                << " status=MODEL_REJECTED reason=unconditional_branch_requires_mel_features\n";
      return 0;
    }
    if (field == "encoder_repetition_penalty") {
      const auto factor = parse_optional_f32(value);
      if (!factor)
        throw std::runtime_error("encoder repetition argument");
      const whisper_interface::IgnoreEncoderTokenPenaltyWithoutEncoderTokenIds
          ignored{*factor, 0};
      if (!ignored.valid())
        throw std::runtime_error("encoder repetition ignored ADT invariant");
      std::cout << "WHISPER_CPP23_GENERATION_APPLICABILITY field=" << field
                << " status=MODEL_IGNORED reason=continuous_encoder_has_no_input_ids\n";
      return 0;
    }
    if (field == "encoder_no_repeat_ngram_size") {
      const auto order = parse_optional_size(value, 448);
      if (!order || *order == 0)
        throw std::runtime_error("encoder ngram argument");
      const whisper_interface::IgnoreEncoderTokenPenaltyWithoutEncoderTokenIds
          ignored{1.0f, *order};
      if (!ignored.valid())
        throw std::runtime_error("encoder ngram ignored ADT invariant");
      std::cout << "WHISPER_CPP23_GENERATION_APPLICABILITY field=" << field
                << " status=MODEL_IGNORED reason=continuous_encoder_has_no_input_ids\n";
      return 0;
    }
    throw std::runtime_error("unknown generation applicability field");
  }
  if (argc == 12 && std::string(argv[1]) == "--contrastive-search") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    const auto candidates = parse_optional_size(argv[9], 51864);
    const auto penalty = parse_optional_f32(argv[10]);
    const auto maximum_positions = parse_optional_size(argv[11]);
    if (!candidates || !penalty || !maximum_positions)
      throw std::runtime_error("contrastive search arguments");
    const whisper_interface::ContrastiveSearch configuration{
        *candidates, *penalty,
        whisper_interface::SequentialCandidateEvaluation{}};
    if (!configuration.valid())
      throw std::runtime_error("contrastive search configuration ADT invariant");
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    CachedDecoder cached(tensors, execution);
    auto result = contrastive_search(
        cached, memory, generated_whisper::forced_prefix, configuration,
        *maximum_positions, execution);
    execution.require_all();
    std::cout << "WHISPER_CPP23_CONTRASTIVE_SEARCH candidates=" << *candidates
              << " penalty=" << std::setprecision(9) << *penalty
              << " tokens=";
    for (std::size_t index = 0; index < result.tokens.size(); ++index) {
      if (index)
        std::cout << ',';
      std::cout << result.tokens[index];
    }
    std::cout << " candidate_branches=" << result.candidate_branches
              << " cosine_edges=" << result.cosine_edges
              << " maximum_cosine=" << result.maximum_cosine
              << " first_candidates=";
    for (std::size_t index = 0; index < result.first_candidate_tokens.size();
         ++index) {
      if (index)
        std::cout << ',';
      std::cout << result.first_candidate_tokens[index];
    }
    std::cout << " first_probabilities=";
    for (std::size_t index = 0; index < result.first_probabilities.size();
         ++index) {
      if (index)
        std::cout << ',';
      std::cout << result.first_probabilities[index];
    }
    std::cout << " first_degenerations=";
    for (std::size_t index = 0; index < result.first_degenerations.size();
         ++index) {
      if (index)
        std::cout << ',';
      std::cout << result.first_degenerations[index];
    }
    std::cout << " first_scores=";
    for (std::size_t index = 0; index < result.first_scores.size(); ++index) {
      if (index)
        std::cout << ',';
      std::cout << result.first_scores[index];
    }
    std::cout << " first_selected_rank=" << result.first_selected_rank
              << " graph_nodes_visited=" << execution.visited() << "\n";
    return 0;
  }
  if (argc == 18 && std::string(argv[1]) == "--beam-sample-mass") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    const auto beams = parse_optional_size(argv[9], 32);
    const auto temperature = parse_optional_f32(argv[10]);
    if (!beams || *beams <= 1 || !temperature || *temperature <= 0.0f)
      throw std::runtime_error("beam sample mass arguments");
    const auto filters = parse_sampling_filters(argv[11], argv[12], argv[13],
                                                argv[14], argv[15], argv[16]);
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    CachedDecoder cached(tensors, execution);
    DecoderKVState state;
    F32 logits;
    for (auto token : generated_whisper::forced_prefix) {
      execution.hit(generated_whisper::nodes[73]);
      logits = cached.step(token, memory, state);
    }
    execution.hit(generated_whisper::nodes[70]);
    auto active = log_softmax_logits(logits);
    active = policy_logits(active, 0);
    for (auto &value : active)
      value /= *temperature;
    apply_sampling_filters(active, filters, 2);
    F32 flattened(*beams * 51864, -1.0e9f);
    std::copy(active.begin(), active.end(), flattened.begin());
    execution.hit(generated_whisper::nodes[71]);
    softmax_rows(flattened, 1, flattened.size());
    execution.require_range(0, 72);
    write_f32(argv[17], flattened);
    const auto support =
        std::count_if(flattened.begin(), flattened.end(),
                      [](auto probability) { return probability > 0.0f; });
    std::cout << "WHISPER_CPP23_BEAM_SAMPLE_MASS beams=" << *beams
              << " support=" << support << " sum=" << std::setprecision(17)
              << std::accumulate(flattened.begin(), flattened.end(), 0.0)
              << " graph_nodes_visited=" << execution.visited() << "\n";
    return 0;
  }
  if ((argc == 22 || argc == 29) && std::string(argv[1]) == "--beam-sample") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    const auto beams = parse_optional_size(argv[9], 32);
    const auto return_sequences = parse_optional_size(argv[10], 32);
    const auto maximum_positions = parse_optional_size(argv[11]);
    const auto length_penalty = parse_optional_f32(argv[12]);
    const auto temperature = parse_optional_f32(argv[14]);
    std::size_t seed_used = 0;
    const auto seed = std::stoull(argv[15], &seed_used);
    if (!beams || !return_sequences || !maximum_positions ||
        !length_penalty || !temperature || *temperature <= 0.0f ||
        seed_used != std::string(argv[15]).size())
      throw std::runtime_error("beam sample arguments");
    whisper_interface::BeamStoppingPolicy stopping;
    if (std::string(argv[13]) == "heuristic")
      stopping = whisper_interface::HeuristicBeamStopping{};
    else if (std::string(argv[13]) == "all-finished")
      stopping = whisper_interface::StopWhenAllBeamsFinished{};
    else if (std::string(argv[13]) == "canonical")
      stopping = whisper_interface::CanonicalBeamStopping{};
    else
      throw std::runtime_error("beam sample stopping policy");
    whisper_interface::SampledBeamSearch configuration{
        *beams,
        *return_sequences,
        *temperature,
        seed,
        *length_penalty,
        stopping,
        parse_sampling_filters(argv[16], argv[17], argv[18], argv[19],
                               argv[20], argv[21])};
    if (!configuration.valid())
      throw std::runtime_error("beam sample configuration ADT invariant");
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    CachedDecoder cached(tensors, execution);
    std::optional<whisper_interface::GenerationLogitPolicies> synthid_policies;
    if (argc == 29) {
      synthid_policies.emplace();
      synthid_policies->watermark = parse_synthid_watermark_policy(
          argv[22], argv[23], argv[24], argv[25], argv[26], argv[27], argv[28]);
    }
    auto result = sampled_beam_search(
        cached, memory, generated_whisper::forced_prefix, configuration,
        *maximum_positions, execution,
        synthid_policies ? &*synthid_policies : nullptr);
    execution.require_all();
    std::cout << "WHISPER_CPP23_BEAM_SAMPLE beams=" << *beams
              << " returned=" << result.sequences.size() << " sequences=";
    for (std::size_t sequence = 0; sequence < result.sequences.size();
         ++sequence) {
      if (sequence)
        std::cout << ';';
      for (std::size_t token = 0;
           token < result.sequences[sequence].tokens.size(); ++token) {
        if (token)
          std::cout << ',';
        std::cout << result.sequences[sequence].tokens[token];
      }
    }
    std::cout << " scores=" << std::setprecision(9);
    for (std::size_t sequence = 0; sequence < result.sequences.size();
         ++sequence) {
      if (sequence)
        std::cout << ',';
      std::cout << result.sequences[sequence].score;
    }
    std::cout << " sampled_candidates=" << result.sampled_candidates
              << " expanded_candidates=" << result.expanded_candidates
              << " cache_branches=" << result.cache_branches
              << " unique_draw_sets=" << result.every_draw_set_unique
              << " mass_error=" << result.maximum_conditional_mass_error
              << " synthid_state_rows=" << result.synthid_context_hashes.size()
              << " synthid_hashes=";
    for (std::size_t i = 0; i < result.synthid_context_hashes.size(); ++i) {
      if (i) std::cout << ',';
      std::cout << result.synthid_context_hashes[i];
    }
    std::cout << " synthid_repeated=";
    for (bool value : result.synthid_repeated) std::cout << int(value);
    std::cout << " synthid_skipped=";
    for (bool value : result.synthid_skipped) std::cout << int(value);
    std::cout
              << " graph_nodes_visited=" << execution.visited() << "\n";
    return 0;
  }
  if (argc == 22 &&
      std::string(argv[1]) == "--constrained-beam-search-synthid") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    const auto beams = parse_optional_size(argv[9], 32);
    const auto return_sequences = parse_optional_size(argv[10], 32);
    const auto maximum_positions = parse_optional_size(argv[11]);
    const auto length_penalty = parse_optional_f32(argv[12]);
    if (!beams || !return_sequences || !maximum_positions || !length_penalty)
      throw std::runtime_error("SynthID constrained beam arguments");
    whisper_interface::BeamStoppingPolicy stopping;
    if (std::string(argv[13]) == "heuristic")
      stopping = whisper_interface::HeuristicBeamStopping{};
    else if (std::string(argv[13]) == "all-finished")
      stopping = whisper_interface::StopWhenAllBeamsFinished{};
    else if (std::string(argv[13]) == "canonical")
      stopping = whisper_interface::CanonicalBeamStopping{};
    else
      throw std::runtime_error("SynthID constrained beam stopping policy");
    whisper_interface::ConstrainedBeamSearch configuration{
        *beams, *return_sequences, *length_penalty, stopping,
        parse_positive_constraints(argv[14])};
    whisper_interface::GenerationLogitPolicies policies{};
    policies.watermark = parse_synthid_watermark_policy(
        argv[15], argv[16], argv[17], argv[18], argv[19], argv[20], argv[21]);
    if (!configuration.valid() || !policies.valid())
      throw std::runtime_error("SynthID constrained beam request invariant");
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    CachedDecoder cached(tensors, execution);
    auto result = constrained_beam_search(
        cached, memory, generated_whisper::forced_prefix, configuration,
        *maximum_positions, execution, &policies);
    execution.require_all();
    if (result.synthid_context_hashes.size() % *beams != 0 ||
        result.synthid_repeated.size() !=
            result.synthid_context_hashes.size() ||
        result.synthid_skipped.size() != result.synthid_context_hashes.size())
      throw std::runtime_error("SynthID constrained beam trace shape");
    std::cout << "WHISPER_CPP23_SYNTHID_CONSTRAINED_BEAM beams=" << *beams
              << " returned=" << result.sequences.size() << " sequences=";
    for (std::size_t sequence = 0; sequence < result.sequences.size();
         ++sequence) {
      if (sequence)
        std::cout << ';';
      for (std::size_t token = 0;
           token < result.sequences[sequence].tokens.size(); ++token) {
        if (token)
          std::cout << ',';
        std::cout << result.sequences[sequence].tokens[token];
      }
    }
    std::cout << " scores=" << std::setprecision(9);
    for (std::size_t sequence = 0; sequence < result.sequences.size();
         ++sequence) {
      if (sequence)
        std::cout << ',';
      std::cout << result.sequences[sequence].score;
    }
    std::cout << " calls=" << result.synthid_context_hashes.size() / *beams
              << " context_hashes=";
    for (std::size_t index = 0; index < result.synthid_context_hashes.size();
         ++index) {
      if (index)
        std::cout << ',';
      std::cout << result.synthid_context_hashes[index];
    }
    std::cout << " repeated=";
    for (auto value : result.synthid_repeated)
      std::cout << (value ? '1' : '0');
    std::cout << " skipped=";
    for (auto value : result.synthid_skipped)
      std::cout << (value ? '1' : '0');
    std::cout << " expanded_candidates=" << result.expanded_candidates
              << " cache_branches=" << result.cache_branches
              << " graph_nodes_visited=" << execution.visited() << "\n";
    return 0;
  }
  if (argc == 15 && std::string(argv[1]) == "--constrained-beam-search") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    const auto beams = parse_optional_size(argv[9], 32);
    const auto return_sequences = parse_optional_size(argv[10], 32);
    const auto maximum_positions = parse_optional_size(argv[11]);
    const auto length_penalty = parse_optional_f32(argv[12]);
    if (!beams || !return_sequences || !maximum_positions || !length_penalty)
      throw std::runtime_error("constrained beam search arguments");
    whisper_interface::BeamStoppingPolicy stopping;
    if (std::string(argv[13]) == "heuristic")
      stopping = whisper_interface::HeuristicBeamStopping{};
    else if (std::string(argv[13]) == "all-finished")
      stopping = whisper_interface::StopWhenAllBeamsFinished{};
    else if (std::string(argv[13]) == "canonical")
      stopping = whisper_interface::CanonicalBeamStopping{};
    else
      throw std::runtime_error("constrained beam stopping policy");
    whisper_interface::ConstrainedBeamSearch configuration{
        *beams, *return_sequences, *length_penalty, stopping,
        parse_positive_constraints(argv[14])};
    if (!configuration.valid())
      throw std::runtime_error(
          "constrained beam search configuration ADT invariant");
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    CachedDecoder cached(tensors, execution);
    auto result = constrained_beam_search(
        cached, memory, generated_whisper::forced_prefix, configuration,
        *maximum_positions, execution);
    execution.require_all();
    std::cout << "WHISPER_CPP23_CONSTRAINED_BEAM_SEARCH beams=" << *beams
              << " returned=" << result.sequences.size() << " sequences=";
    for (std::size_t sequence = 0; sequence < result.sequences.size();
         ++sequence) {
      if (sequence)
        std::cout << ';';
      for (std::size_t token = 0;
           token < result.sequences[sequence].tokens.size(); ++token) {
        if (token)
          std::cout << ',';
        std::cout << result.sequences[sequence].tokens[token];
      }
    }
    std::cout << " scores=" << std::setprecision(9);
    for (std::size_t sequence = 0; sequence < result.sequences.size();
         ++sequence) {
      if (sequence)
        std::cout << ',';
      std::cout << result.sequences[sequence].score;
    }
    std::cout << " expanded_candidates=" << result.expanded_candidates
              << " cache_branches=" << result.cache_branches
              << " graph_nodes_visited=" << execution.visited() << "\n";
    return 0;
  }
  if (argc == 23 &&
      std::string(argv[1]) == "--group-beam-search-synthid") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    const auto beams = parse_optional_size(argv[9], 32);
    const auto groups = parse_optional_size(argv[10], 32);
    const auto return_sequences = parse_optional_size(argv[11], 32);
    const auto maximum_positions = parse_optional_size(argv[12]);
    const auto length_penalty = parse_optional_f32(argv[13]);
    const auto diversity_penalty = parse_optional_f32(argv[14]);
    if (!beams || !groups || !return_sequences || !maximum_positions ||
        !length_penalty || !diversity_penalty)
      throw std::runtime_error("SynthID group beam arguments");
    whisper_interface::BeamStoppingPolicy stopping;
    if (std::string(argv[15]) == "heuristic")
      stopping = whisper_interface::HeuristicBeamStopping{};
    else if (std::string(argv[15]) == "all-finished")
      stopping = whisper_interface::StopWhenAllBeamsFinished{};
    else if (std::string(argv[15]) == "canonical")
      stopping = whisper_interface::CanonicalBeamStopping{};
    else
      throw std::runtime_error("SynthID group beam stopping policy");
    whisper_interface::DiverseGroupBeamSearch configuration{
        *beams, *groups, *return_sequences, *diversity_penalty,
        *length_penalty, stopping};
    whisper_interface::GenerationLogitPolicies policies{};
    policies.watermark = parse_synthid_watermark_policy(
        argv[16], argv[17], argv[18], argv[19], argv[20], argv[21], argv[22]);
    if (!configuration.valid() || !policies.valid())
      throw std::runtime_error("SynthID group beam request invariant");
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    CachedDecoder cached(tensors, execution);
    auto result = group_beam_search(
        cached, memory, generated_whisper::forced_prefix, configuration,
        *maximum_positions, execution, &policies);
    execution.require_all();
    const auto group_size = *beams / *groups;
    if (group_size == 0 || result.synthid_context_hashes.size() % group_size != 0 ||
        result.synthid_repeated.size() !=
            result.synthid_context_hashes.size() ||
        result.synthid_skipped.size() != result.synthid_context_hashes.size())
      throw std::runtime_error("SynthID group beam trace shape");
    std::cout << "WHISPER_CPP23_SYNTHID_GROUP_BEAM beams=" << *beams
              << " groups=" << *groups
              << " returned=" << result.sequences.size() << " sequences=";
    for (std::size_t sequence = 0; sequence < result.sequences.size();
         ++sequence) {
      if (sequence)
        std::cout << ';';
      for (std::size_t token = 0;
           token < result.sequences[sequence].tokens.size(); ++token) {
        if (token)
          std::cout << ',';
        std::cout << result.sequences[sequence].tokens[token];
      }
    }
    std::cout << " scores=" << std::setprecision(9);
    for (std::size_t sequence = 0; sequence < result.sequences.size();
         ++sequence) {
      if (sequence)
        std::cout << ',';
      std::cout << result.sequences[sequence].score;
    }
    std::cout << " calls=" << result.synthid_context_hashes.size() / group_size
              << " context_hashes=";
    for (std::size_t index = 0; index < result.synthid_context_hashes.size();
         ++index) {
      if (index)
        std::cout << ',';
      std::cout << result.synthid_context_hashes[index];
    }
    std::cout << " repeated=";
    for (auto value : result.synthid_repeated)
      std::cout << (value ? '1' : '0');
    std::cout << " skipped=";
    for (auto value : result.synthid_skipped)
      std::cout << (value ? '1' : '0');
    std::cout << " expanded_candidates=" << result.expanded_candidates
              << " cache_branches=" << result.cache_branches
              << " graph_nodes_visited=" << execution.visited() << "\n";
    return 0;
  }
  if (argc == 16 && std::string(argv[1]) == "--group-beam-search") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    const auto beams = parse_optional_size(argv[9], 32);
    const auto groups = parse_optional_size(argv[10], 32);
    const auto return_sequences = parse_optional_size(argv[11], 32);
    const auto maximum_positions = parse_optional_size(argv[12]);
    const auto length_penalty = parse_optional_f32(argv[13]);
    const auto diversity_penalty = parse_optional_f32(argv[14]);
    if (!beams || !groups || !return_sequences || !maximum_positions ||
        !length_penalty || !diversity_penalty)
      throw std::runtime_error("group beam search arguments");
    whisper_interface::BeamStoppingPolicy stopping;
    if (std::string(argv[15]) == "heuristic")
      stopping = whisper_interface::HeuristicBeamStopping{};
    else if (std::string(argv[15]) == "all-finished")
      stopping = whisper_interface::StopWhenAllBeamsFinished{};
    else if (std::string(argv[15]) == "canonical")
      stopping = whisper_interface::CanonicalBeamStopping{};
    else
      throw std::runtime_error("group beam stopping policy");
    whisper_interface::DiverseGroupBeamSearch configuration{
        *beams, *groups, *return_sequences, *diversity_penalty,
        *length_penalty, stopping};
    if (!configuration.valid())
      throw std::runtime_error("group beam search configuration ADT invariant");
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    CachedDecoder cached(tensors, execution);
    auto result = group_beam_search(
        cached, memory, generated_whisper::forced_prefix, configuration,
        *maximum_positions, execution);
    execution.require_all();
    std::cout << "WHISPER_CPP23_GROUP_BEAM_SEARCH beams=" << *beams
              << " groups=" << *groups
              << " returned=" << result.sequences.size() << " sequences=";
    for (std::size_t sequence = 0; sequence < result.sequences.size();
         ++sequence) {
      if (sequence)
        std::cout << ';';
      for (std::size_t token = 0;
           token < result.sequences[sequence].tokens.size(); ++token) {
        if (token)
          std::cout << ',';
        std::cout << result.sequences[sequence].tokens[token];
      }
    }
    std::cout << " scores=" << std::setprecision(9);
    for (std::size_t sequence = 0; sequence < result.sequences.size();
         ++sequence) {
      if (sequence)
        std::cout << ',';
      std::cout << result.sequences[sequence].score;
    }
    std::cout << " expanded_candidates=" << result.expanded_candidates
              << " cache_branches=" << result.cache_branches
              << " graph_nodes_visited=" << execution.visited() << "\n";
    return 0;
  }
  if (argc >= 9 && std::string(argv[1]) == "--transcribe-batch") {
    whisper_interface::ShortFormGreedyBatchRequest request;
    for (int i = 8; i < argc; ++i)
      request.audio.items.push_back({argv[i]});
    if (!request.valid())
      throw std::runtime_error("short-form batch request ADT invariant");
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    const auto window = read_f32(argv[4]);
    const auto filters = read_f32(argv[5]);
    TokenDecoder token_decoder(argv[6], argv[7]);
    const bool verify_recompute =
        std::getenv("WHISPER_VERIFY_RECOMPUTE") != nullptr;
    for (std::size_t item = 0; item < request.audio.items.size(); ++item) {
      const auto result = transcribe_short_form_item(
          tensors, request.audio.items[item].path, window, filters,
          token_decoder, verify_recompute);
      std::cout << "WHISPER_CPP23_BATCH_ITEM item=" << item << " tokens=";
      for (std::size_t token = 0; token < result.tokens.size(); ++token) {
        if (token)
          std::cout << ',';
        std::cout << result.tokens[token];
      }
      std::cout << " cache_logit_error=" << result.maximum_cache_logit_error
                << " graph_nodes_visited=" << result.graph_nodes_visited
                << " text=" << std::quoted(result.text) << "\n";
    }
    std::cout << "WHISPER_CPP23_BATCH_OK items=" << request.audio.items.size()
              << " shared_checkpoint=1 isolated_item_state=1 peak_rss_bytes="
              << peak_rss_bytes() << "\n";
    return 0;
  }
  if (argc == 21 && std::string(argv[1]) == "--beam-search-synthid") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    const auto beams = parse_optional_size(argv[9], 32);
    const auto return_sequences = parse_optional_size(argv[10], 32);
    const auto maximum_positions = parse_optional_size(argv[11]);
    const auto length_penalty = parse_optional_f32(argv[12]);
    if (!beams || !return_sequences || !maximum_positions || !length_penalty)
      throw std::runtime_error("SynthID beam search arguments");
    whisper_interface::BeamStoppingPolicy stopping;
    if (std::string(argv[13]) == "heuristic")
      stopping = whisper_interface::HeuristicBeamStopping{};
    else if (std::string(argv[13]) == "all-finished")
      stopping = whisper_interface::StopWhenAllBeamsFinished{};
    else if (std::string(argv[13]) == "canonical")
      stopping = whisper_interface::CanonicalBeamStopping{};
    else
      throw std::runtime_error("SynthID beam stopping policy");
    whisper_interface::StandardBeamSearch configuration{
        *beams, *return_sequences, *length_penalty, stopping};
    whisper_interface::GenerationLogitPolicies policies{};
    policies.watermark = parse_synthid_watermark_policy(
        argv[14], argv[15], argv[16], argv[17], argv[18], argv[19], argv[20]);
    if (!configuration.valid() || !policies.valid())
      throw std::runtime_error("SynthID beam request ADT invariant");
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    CachedDecoder cached(tensors, execution);
    auto result = beam_search(cached, memory, generated_whisper::forced_prefix,
                              configuration, *maximum_positions, execution,
                              &policies);
    execution.require_all();
    if (result.synthid_context_hashes.size() % *beams != 0 ||
        result.synthid_repeated.size() !=
            result.synthid_context_hashes.size() ||
        result.synthid_skipped.size() != result.synthid_context_hashes.size())
      throw std::runtime_error("SynthID beam trace shape");
    std::cout << "WHISPER_CPP23_SYNTHID_BEAM beams=" << *beams
              << " returned=" << result.sequences.size() << " sequences=";
    for (std::size_t sequence = 0; sequence < result.sequences.size();
         ++sequence) {
      if (sequence)
        std::cout << ';';
      for (std::size_t token = 0;
           token < result.sequences[sequence].tokens.size(); ++token) {
        if (token)
          std::cout << ',';
        std::cout << result.sequences[sequence].tokens[token];
      }
    }
    std::cout << " scores=" << std::setprecision(9);
    for (std::size_t sequence = 0; sequence < result.sequences.size();
         ++sequence) {
      if (sequence)
        std::cout << ',';
      std::cout << result.sequences[sequence].score;
    }
    std::cout << " calls=" << result.synthid_context_hashes.size() / *beams
              << " context_hashes=";
    for (std::size_t index = 0; index < result.synthid_context_hashes.size();
         ++index) {
      if (index)
        std::cout << ',';
      std::cout << result.synthid_context_hashes[index];
    }
    std::cout << " repeated=";
    for (auto value : result.synthid_repeated)
      std::cout << (value ? '1' : '0');
    std::cout << " skipped=";
    for (auto value : result.synthid_skipped)
      std::cout << (value ? '1' : '0');
    std::cout << " expanded_candidates=" << result.expanded_candidates
              << " cache_branches=" << result.cache_branches
              << " graph_nodes_visited=" << execution.visited() << "\n";
    return 0;
  }
  if (argc == 14 && std::string(argv[1]) == "--beam-search") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    const auto beams = parse_optional_size(argv[9], 32);
    const auto return_sequences = parse_optional_size(argv[10], 32);
    const auto maximum_positions = parse_optional_size(argv[11]);
    const auto length_penalty = parse_optional_f32(argv[12]);
    if (!beams || !return_sequences || !maximum_positions || !length_penalty)
      throw std::runtime_error("beam search arguments");
    whisper_interface::BeamStoppingPolicy stopping;
    if (std::string(argv[13]) == "heuristic")
      stopping = whisper_interface::HeuristicBeamStopping{};
    else if (std::string(argv[13]) == "all-finished")
      stopping = whisper_interface::StopWhenAllBeamsFinished{};
    else if (std::string(argv[13]) == "canonical")
      stopping = whisper_interface::CanonicalBeamStopping{};
    else
      throw std::runtime_error("beam stopping policy");
    whisper_interface::StandardBeamSearch configuration{
        *beams, *return_sequences, *length_penalty, stopping};
    if (!configuration.valid())
      throw std::runtime_error("beam search configuration ADT invariant");
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    CachedDecoder cached(tensors, execution);
    auto result = beam_search(cached, memory, generated_whisper::forced_prefix,
                              configuration, *maximum_positions, execution);
    execution.require_all();
    std::cout << "WHISPER_CPP23_BEAM_SEARCH beams=" << *beams
              << " returned=" << result.sequences.size() << " sequences=";
    for (std::size_t sequence = 0; sequence < result.sequences.size();
         ++sequence) {
      if (sequence)
        std::cout << ";";
      for (std::size_t token = 0;
           token < result.sequences[sequence].tokens.size(); ++token) {
        if (token)
          std::cout << ",";
        std::cout << result.sequences[sequence].tokens[token];
      }
    }
    std::cout << " scores=" << std::setprecision(9);
    for (std::size_t sequence = 0; sequence < result.sequences.size();
         ++sequence) {
      if (sequence)
        std::cout << ",";
      std::cout << result.sequences[sequence].score;
    }
    std::cout << " expanded_candidates=" << result.expanded_candidates
              << " cache_branches=" << result.cache_branches
              << " graph_nodes_visited=" << execution.visited() << "\n";
    return 0;
  }
  if (argc == 16 && std::string(argv[1]) == "--transcribe-score-policies") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    whisper_interface::GenerationLogitPolicies policies{};
    policies.sequence_bias = parse_sequence_bias(argv[9]);
    policies.forced_beginning = parse_forced_beginning(argv[10]);
    policies.forced_ending = parse_forced_ending(argv[11]);
    policies.invalid_logits =
        parse_bool01(argv[12])
            ? whisper_interface::
                  InvalidLogitPolicy{whisper_interface::RepairInvalidLogits{}}
            : whisper_interface::InvalidLogitPolicy{
                  whisper_interface::PreserveInvalidLogits{}};
    policies.exponential_eos = parse_exponential_eos(argv[13]);
    policies.normalization =
        parse_bool01(argv[14])
            ? whisper_interface::
                  LogitNormalizationPolicy{whisper_interface::
                                               NormalizeLogProbabilities{}}
            : whisper_interface::LogitNormalizationPolicy{
                  whisper_interface::PreserveLogitScale{}};
    const auto maximum_positions = parse_optional_size(argv[15]);
    if (!maximum_positions || !policies.valid() ||
        *maximum_positions <= generated_whisper::forced_prefix.size())
      throw std::runtime_error("score policy request ADT invariant");
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    Decoder decoder(tensors, execution);
    CachedDecoder cached(tensors, execution);
    auto generated = generate(
        decoder, cached, memory, Selection::Greedy, 0.0f, 0, false, {},
        generated_whisper::forced_prefix, whisper_interface::NoTimestamps{},
        nullptr, nullptr, execution, &policies, *maximum_positions);
    execution.require_all();
    std::cout << "WHISPER_CPP23_SCORE_POLICIES tokens=";
    for (std::size_t i = 0; i < generated.tokens.size(); ++i) {
      if (i)
        std::cout << ",";
      std::cout << generated.tokens[i];
    }
    std::cout << " terminated_by_eos=" << generated.terminated_by_eos
              << " graph_nodes_visited=" << execution.visited() << "\n";
    return 0;
  }
  if (argc == 17 && std::string(argv[1]) == "--sampling-mass") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    const auto temperature_value = parse_optional_f32(argv[9]);
    if (!temperature_value || *temperature_value <= 0.0f)
      throw std::runtime_error("sampling temperature ADT invariant");
    const auto filters = parse_sampling_filters(argv[10], argv[11], argv[12],
                                                argv[13], argv[14], argv[15]);
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    CachedDecoder cached(tensors, execution);
    DecoderKVState state;
    F32 logits;
    for (auto token : generated_whisper::forced_prefix) {
      execution.hit(generated_whisper::nodes[73]);
      logits = cached.step(token, memory, state);
    }
    execution.hit(generated_whisper::nodes[70]);
    auto mass = policy_logits(logits, 0);
    for (auto &value : mass)
      value /= *temperature_value;
    apply_sampling_filters(mass, filters);
    execution.hit(generated_whisper::nodes[71]);
    softmax_rows(mass, 1, mass.size());
    execution.require_range(0, 72);
    if (execution.visited() != 73)
      throw std::runtime_error("sampling mass graph coverage");
    write_f32(argv[16], mass);
    const auto support = std::count_if(mass.begin(), mass.end(),
                                       [](auto value) { return value > 0.0f; });
    const auto selected =
        std::max_element(mass.begin(), mass.end()) - mass.begin();
    std::cout << "WHISPER_CPP23_SAMPLING_MASS support=" << support
              << " selected=" << selected << " sum=" << std::setprecision(17)
              << std::accumulate(mass.begin(), mass.end(), 0.0)
              << " graph_nodes_visited=" << execution.visited() << "\n";
    return 0;
  }
  if (argc == 11 && std::string(argv[1]) == "--transcribe-length-limit") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    auto count = parse_optional_size(argv[10]);
    if (!count)
      throw std::runtime_error("generation length count");
    whisper_interface::GenerationLengthLimit length;
    std::size_t maximum_positions = 0;
    if (std::string(argv[9]) == "max-length") {
      length = whisper_interface::MaximumTotalPositions{*count};
      maximum_positions = *count;
    } else if (std::string(argv[9]) == "max-new-tokens") {
      length = whisper_interface::MaximumNewTokens{*count};
      maximum_positions = generated_whisper::forced_prefix.size() + *count;
    } else {
      throw std::runtime_error("generation length constructor");
    }
    if (!std::visit([](const auto &value) { return value.valid(); }, length) ||
        maximum_positions > 448)
      throw std::runtime_error("generation length ADT invariant");
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    Decoder decoder(tensors, execution);
    CachedDecoder cached(tensors, execution);
    auto generated = generate(decoder, cached, memory, Selection::Greedy, 0.0f,
                              0, false, {}, generated_whisper::forced_prefix,
                              whisper_interface::NoTimestamps{}, nullptr,
                              nullptr, execution, nullptr, maximum_positions);
    execution.require_all();
    std::cout << "WHISPER_CPP23_LENGTH_LIMIT kind=" << argv[9]
              << " count=" << *count << " tokens=";
    for (std::size_t i = 0; i < generated.tokens.size(); ++i) {
      if (i)
        std::cout << ",";
      std::cout << generated.tokens[i];
    }
    std::cout << " graph_nodes_visited=" << execution.visited() << "\n";
    return 0;
  }
  if (argc == 14 && std::string(argv[1]) == "--transcribe-logit-policies") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    auto repetition = parse_optional_f32(argv[9]);
    auto ngram = parse_optional_size(argv[10]);
    auto minimum_length = parse_optional_size(argv[11]);
    auto minimum_new_tokens = parse_optional_size(argv[12]);
    whisper_interface::GenerationLogitPolicies policies{
        repetition
            ? whisper_interface::
                  RepetitionPolicy{whisper_interface::RepetitionPenalty{
                      *repetition}}
            : whisper_interface::
                  RepetitionPolicy{whisper_interface::NoRepetitionPenalty{}},
        ngram ? whisper_interface::NGramPolicy{whisper_interface::NoRepeatNGram{
                    *ngram}}
              : whisper_interface::
                    NGramPolicy{whisper_interface::AllowRepeatedNGrams{}},
        parse_forbidden_sequences(argv[13]),
        minimum_length
            ? whisper_interface::
                  MinimumLengthPolicy{whisper_interface::MinimumLength{
                      *minimum_length}}
            : whisper_interface::
                  MinimumLengthPolicy{whisper_interface::NoMinimumLength{}},
        minimum_new_tokens
            ? whisper_interface::
                  MinimumNewTokenPolicy{whisper_interface::MinimumNewTokens{
                      *minimum_new_tokens}}
            : whisper_interface::MinimumNewTokenPolicy{
                  whisper_interface::NoMinimumNewTokens{}}};
    if (!policies.valid())
      throw std::runtime_error("generation logit policies ADT invariant");
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    Decoder decoder(tensors, execution);
    CachedDecoder cached(tensors, execution);
    auto generated = generate(decoder, cached, memory, Selection::Greedy, 0.0f,
                              0, false, {}, generated_whisper::forced_prefix,
                              whisper_interface::NoTimestamps{}, nullptr,
                              nullptr, execution, &policies);
    execution.require_all();
    std::cout << "WHISPER_CPP23_LOGIT_POLICIES tokens=";
    for (std::size_t i = 0; i < generated.tokens.size(); ++i) {
      if (i)
        std::cout << ",";
      std::cout << generated.tokens[i];
    }
    std::cout << " graph_nodes_visited=" << execution.visited() << "\n";
    return 0;
  }
  if (argc == 14 && std::string(argv[1]) == "--transcribe-long-fallback") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    auto flag = std::string(argv[9]);
    if (flag != "0" && flag != "1")
      throw std::runtime_error("long-form condition flag");
    whisper_interface::FallbackThresholds fallback{
        parse_optional_f32(argv[10]), parse_optional_f32(argv[11]),
        parse_optional_f32(argv[12]), parse_f32_csv(argv[13])};
    if (!fallback.valid())
      throw std::runtime_error("fallback ADT invariant");
    GraphExecutionAudit execution;
    auto result = execute_long_form(
        tensors, argv[4], read_f32(argv[5]), read_f32(argv[6]), flag == "1", {},
        whisper_interface::FirstSegmentPrompt{}, nullptr, &fallback, execution);
    std::vector<std::int32_t> sequence;
    for (const auto &segment : result.segments)
      sequence.insert(sequence.end(), segment.tokens.begin(),
                      segment.tokens.end());
    execution.require_all();
    std::cout << "WHISPER_CPP23_LONG_FALLBACK generation_calls="
              << result.generation_calls
              << " fallback_attempts=" << result.fallback.size()
              << " skipped_windows=" << result.skipped_windows
              << " total_frames=" << result.total_frames << " tokens=";
    for (std::size_t i = 0; i < sequence.size(); ++i) {
      if (i)
        std::cout << ",";
      std::cout << sequence[i];
    }
    std::cout << " segments=";
    if (result.segments.empty())
      std::cout << "-";
    else {
      std::size_t token_offset = 0;
      for (std::size_t i = 0; i < result.segments.size(); ++i) {
        if (i)
          std::cout << ",";
        std::cout << std::fixed << std::setprecision(2)
                  << result.segments[i].start_seconds << ":"
                  << result.segments[i].end_seconds << ":" << token_offset
                  << ":" << token_offset + result.segments[i].tokens.size();
        token_offset += result.segments[i].tokens.size();
      }
    }
    std::cout << " seeks=";
    for (std::size_t i = 0; i < result.seeks.size(); ++i) {
      if (i)
        std::cout << ",";
      std::cout << result.seeks[i];
    }
    std::cout << " observations=" << std::defaultfloat << std::setprecision(9);
    for (std::size_t i = 0; i < result.fallback.size(); ++i) {
      if (i)
        std::cout << ",";
      const auto &o = result.fallback[i];
      std::cout << o.seek << ":" << o.attempt << ":" << o.temperature << ":"
                << o.compression << ":" << o.average_logprob << ":"
                << o.no_speech_probability << ":" << o.needs_fallback << ":"
                << o.should_skip;
    }
    std::cout << " graph_nodes_visited=" << execution.visited() << "\n";
    return 0;
  }
  if (argc == 10 && std::string(argv[1]) == "--transcribe-prefix-allowed") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    auto forced = parse_token_ids(argv[9]);
    whisper_interface::PrefixAllowedTokensFn constraint{
        [&](std::size_t step, std::span<const std::int32_t>,
            std::int32_t token) {
          return step >= forced.size() || token == forced[step];
        }};
    if (!constraint.valid())
      throw std::runtime_error("prefix constraint ADT invariant");
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    Decoder decoder(tensors, execution);
    CachedDecoder cached(tensors, execution);
    auto generated = generate(decoder, cached, memory, Selection::Greedy, 0.0f,
                              0, false, {}, generated_whisper::forced_prefix,
                              whisper_interface::NoTimestamps{}, &constraint,
                              nullptr, execution);
    execution.require_all();
    std::cout << "WHISPER_CPP23_PREFIX_ALLOWED forced=" << forced.size()
              << " tokens=";
    for (std::size_t i = 0; i < generated.tokens.size(); ++i) {
      if (i)
        std::cout << ",";
      std::cout << generated.tokens[i];
    }
    std::cout << " graph_nodes_visited=" << execution.visited() << "\n";
    return 0;
  }
  if (argc == 12 && std::string(argv[1]) == "--transcribe-long-prompt") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    auto flag = std::string(argv[9]);
    if (flag != "0" && flag != "1")
      throw std::runtime_error("long-form condition flag");
    auto prompt = parse_token_ids(argv[10]);
    auto type_name = std::string(argv[11]);
    whisper_interface::PromptConditionType type;
    if (type_name == "first-segment")
      type = whisper_interface::FirstSegmentPrompt{};
    else if (type_name == "all-segments")
      type = whisper_interface::AllSegmentsPrompt{};
    else
      throw std::runtime_error("long-form prompt condition");
    GraphExecutionAudit execution;
    auto result = execute_long_form(tensors, argv[4], read_f32(argv[5]),
                                    read_f32(argv[6]), flag == "1", prompt,
                                    type, nullptr, nullptr, execution);
    std::vector<std::int32_t> sequence;
    for (const auto &segment : result.segments)
      sequence.insert(sequence.end(), segment.tokens.begin(),
                      segment.tokens.end());
    execution.require_all();
    std::cout << "WHISPER_CPP23_LONG_PROMPT conditioned=" << (flag == "1")
              << " prompt_type=" << type_name
              << " generation_calls=" << result.generation_calls
              << " total_frames=" << result.total_frames << " tokens=";
    for (std::size_t i = 0; i < sequence.size(); ++i) {
      if (i)
        std::cout << ",";
      std::cout << sequence[i];
    }
    std::cout << " segments=";
    std::size_t token_offset = 0;
    for (std::size_t i = 0; i < result.segments.size(); ++i) {
      if (i)
        std::cout << ",";
      std::cout << std::fixed << std::setprecision(2)
                << result.segments[i].start_seconds << ":"
                << result.segments[i].end_seconds << ":" << token_offset << ":"
                << token_offset + result.segments[i].tokens.size();
      token_offset += result.segments[i].tokens.size();
    }
    std::cout << " seeks=";
    for (std::size_t i = 0; i < result.seeks.size(); ++i) {
      if (i)
        std::cout << ",";
      std::cout << result.seeks[i];
    }
    std::cout << " graph_nodes_visited=" << execution.visited() << "\n";
    return 0;
  }
  if (argc == 10 && std::string(argv[1]) == "--transcribe-long") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    auto flag = std::string(argv[9]);
    if (flag != "0" && flag != "1")
      throw std::runtime_error("long-form condition flag");
    GraphExecutionAudit execution;
    std::vector<std::pair<std::size_t, std::size_t>> progress_events;
    whisper_interface::MonitorProgress monitor{
        [&](std::size_t seek, std::size_t maximum) {
          progress_events.emplace_back(seek, maximum);
        }};
    auto result = execute_long_form(
        tensors, argv[4], read_f32(argv[5]), read_f32(argv[6]), flag == "1", {},
        whisper_interface::FirstSegmentPrompt{}, &monitor, nullptr, execution);
    std::vector<std::int32_t> sequence;
    for (const auto &segment : result.segments)
      sequence.insert(sequence.end(), segment.tokens.begin(),
                      segment.tokens.end());
    TokenDecoder token_decoder(argv[7], argv[8]);
    auto text = token_decoder.decode(sequence);
    while (!text.empty() &&
           std::isspace(static_cast<unsigned char>(text.front())))
      text.erase(text.begin());
    while (!text.empty() &&
           std::isspace(static_cast<unsigned char>(text.back())))
      text.pop_back();
    execution.require_all();
    std::cout << "WHISPER_CPP23_LONG_TRANSCRIPT conditioned=" << (flag == "1")
              << " generation_calls=" << result.generation_calls
              << " total_frames=" << result.total_frames << " tokens=";
    for (std::size_t i = 0; i < sequence.size(); ++i) {
      if (i)
        std::cout << ",";
      std::cout << sequence[i];
    }
    std::cout << " segments=";
    std::size_t token_offset = 0;
    for (std::size_t i = 0; i < result.segments.size(); ++i) {
      if (i)
        std::cout << ",";
      std::cout << std::fixed << std::setprecision(2)
                << result.segments[i].start_seconds << ":"
                << result.segments[i].end_seconds << ":" << token_offset << ":"
                << token_offset + result.segments[i].tokens.size();
      token_offset += result.segments[i].tokens.size();
    }
    std::cout << " seeks=";
    for (std::size_t i = 0; i < result.seeks.size(); ++i) {
      if (i)
        std::cout << ",";
      std::cout << result.seeks[i];
    }
    std::cout << " progress=";
    for (std::size_t i = 0; i < progress_events.size(); ++i) {
      if (i)
        std::cout << ",";
      std::cout << progress_events[i].first << ":" << progress_events[i].second;
    }
    std::cout << " graph_nodes_visited=" << execution.visited() << " text=\""
              << text << "\"\n";
    return 0;
  }
  if (argc == 9 && std::string(argv[1]) == "--transcribe-token-timestamps") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    GraphExecutionAudit execution;
    auto pcm = read_wav_pcm16(argv[4]);
    auto feature_frames = std::min<std::size_t>(3000, (pcm.size() + 159) / 160);
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    Decoder decoder(tensors, execution);
    CachedDecoder cached(tensors, execution);
    AlignmentTrace alignment;
    whisper_interface::TokenTimestamps time_output{0.02f};
    if (!time_output.valid())
      throw std::runtime_error("token timestamp ADT invariant");
    auto generated = generate(decoder, cached, memory, Selection::Greedy, 0.0f,
                              0, false, {}, generated_whisper::forced_prefix,
                              time_output, nullptr, &alignment, execution);
    auto timestamps = aligned_token_timestamps(
        alignment, generated_whisper::forced_prefix.size(), feature_frames,
        time_output.seconds_per_token);
    std::vector<std::int32_t> sequence(generated_whisper::forced_prefix.begin(),
                                       generated_whisper::forced_prefix.end());
    sequence.insert(sequence.end(), generated.tokens.begin(),
                    generated.tokens.end());
    sequence.push_back(generated_whisper::eos_token);
    if (sequence.size() != timestamps.size())
      throw std::runtime_error("token timestamp output shape");
    execution.require_all();
    std::cout << "WHISPER_CPP23_TOKEN_TIMESTAMPS tokens=";
    for (std::size_t i = 0; i < sequence.size(); ++i) {
      if (i)
        std::cout << ",";
      std::cout << sequence[i];
    }
    std::cout << " timestamps=";
    for (std::size_t i = 0; i < timestamps.size(); ++i) {
      if (i)
        std::cout << ",";
      std::cout << std::fixed << std::setprecision(2) << timestamps[i];
    }
    std::cout << " alignment_positions=" << alignment.positions
              << " source_positions=" << feature_frames / 2
              << " graph_nodes_visited=" << execution.visited() << "\n";
    return 0;
  }
  if (argc == 9 && std::string(argv[1]) == "--transcribe-timestamps") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    Decoder decoder(tensors, execution);
    CachedDecoder cached(tensors, execution);
    whisper_interface::Segments time_output{0.02f, 0.01f};
    if (!time_output.valid())
      throw std::runtime_error("timestamp ADT invariant");
    auto timestamp_prefix =
        std::span<const std::int32_t>(generated_whisper::forced_prefix)
            .first(1);
    auto generated =
        generate(decoder, cached, memory, Selection::Greedy, 0.0f, 0, false, {},
                 timestamp_prefix, time_output, nullptr, nullptr, execution);
    auto segments = timestamp_segments(generated.tokens,
                                       time_output.seconds_per_timestamp_token);
    TokenDecoder token_decoder(argv[7], argv[8]);
    auto text = token_decoder.decode(generated.tokens);
    while (!text.empty() &&
           std::isspace(static_cast<unsigned char>(text.front())))
      text.erase(text.begin());
    while (!text.empty() &&
           std::isspace(static_cast<unsigned char>(text.back())))
      text.pop_back();
    execution.require_all();
    std::cout << "WHISPER_CPP23_TIMESTAMP_TRANSCRIPT tokens=";
    for (std::size_t i = 0; i < generated.tokens.size(); ++i) {
      if (i)
        std::cout << ",";
      std::cout << generated.tokens[i];
    }
    std::cout << " segments=";
    for (std::size_t i = 0; i < segments.size(); ++i) {
      if (i)
        std::cout << ",";
      std::cout << std::fixed << std::setprecision(2)
                << segments[i].start_seconds << ":" << segments[i].end_seconds
                << ":" << segments[i].token_begin << ":"
                << segments[i].token_end;
    }
    std::cout << " graph_nodes_visited=" << execution.visited() << " text=\""
              << text << "\"\n";
    return 0;
  }
  if (argc == 10 && std::string(argv[1]) == "--transcribe-prompt") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    auto prompt_values = parse_token_ids(argv[9]);
    whisper_interface::PromptTokens prompt{{std::move(prompt_values)}};
    if (!prompt.tokens.valid() || prompt.tokens.values.size() + 2 >= 448)
      throw std::runtime_error("prompt ADT invariant");
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    Decoder decoder(tensors, execution);
    CachedDecoder cached(tensors, execution);
    auto generated = generate(
        decoder, cached, memory, Selection::Greedy, 0.0f, 0, false,
        prompt.tokens.values, generated_whisper::forced_prefix,
        whisper_interface::NoTimestamps{}, nullptr, nullptr, execution);
    TokenDecoder token_decoder(argv[7], argv[8]);
    auto text = token_decoder.decode(generated.tokens);
    while (!text.empty() &&
           std::isspace(static_cast<unsigned char>(text.front())))
      text.erase(text.begin());
    while (!text.empty() &&
           std::isspace(static_cast<unsigned char>(text.back())))
      text.pop_back();
    execution.require_all();
    std::cout << "WHISPER_CPP23_PROMPT_TRANSCRIPT prompt_tokens="
              << prompt.tokens.values.size() << " tokens=";
    for (std::size_t i = 0; i < generated.tokens.size(); ++i) {
      if (i)
        std::cout << ",";
      std::cout << generated.tokens[i];
    }
    std::cout << " graph_nodes_visited=" << execution.visited() << " text=\""
              << text << "\"\n";
    return 0;
  }
  if (argc == 11 && std::string(argv[1]) == "--forward-attentions") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    auto ids = parse_token_ids(argv[7]);
    AttentionWriter writer(argv[8], argv[9]);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace, {}, &writer);
    Decoder decoder(tensors, execution);
    auto logits = decoder.run_attentions(ids, memory, writer, trace);
    write_f32(argv[10], logits);
    execution.require_range(0, 70);
    std::cout << "WHISPER_CPP23_ATTENTIONS_OK decoder_positions=" << ids.size()
              << " attention_tensors=12 logits=" << logits.size()
              << " graph_nodes_visited=" << execution.visited() << "\n";
    return 0;
  }
  if (argc == 11 && std::string(argv[1]) == "--forward-hidden-states") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    auto ids = parse_token_ids(argv[7]);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    Decoder decoder(tensors, execution);
    auto logits = decoder.run(ids, memory, trace);
    const std::vector<std::string> names = {
        "conv2_gelu_position", "encoder_layer_0", "encoder_layer_1",
        "encoder_layer_2",     "encoder_final",   "decoder_embed_position",
        "decoder_layer_0",     "decoder_layer_1", "decoder_layer_2",
        "decoder_final"};
    write_named_tensors(argv[8], argv[9], trace, names);
    write_f32(argv[10], logits);
    execution.require_range(0, 70);
    std::cout << "WHISPER_CPP23_HIDDEN_STATES_OK decoder_positions="
              << ids.size() << " hidden_tensors=" << names.size()
              << " logits=" << logits.size()
              << " graph_nodes_visited=" << execution.visited() << "\n";
    return 0;
  }
  if (argc == 12 && std::string(argv[1]) == "--forward-head-masks") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    auto ids = parse_token_ids(argv[7]);
    auto encoder_mask_values = parse_f32_csv(argv[8]);
    auto decoder_mask_values = parse_f32_csv(argv[9]);
    auto cross_mask_values = parse_f32_csv(argv[10]);
    whisper_interface::HeadMasks masks{
        {encoder_mask_values}, {decoder_mask_values}, {cross_mask_values}};
    if (!masks.valid())
      throw std::runtime_error("head masks ADT invariant");
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace, masks.encoder.layer_head_mass);
    Decoder decoder(tensors, execution);
    auto logits =
        decoder.run_head_masked(ids, memory, masks.decoder.layer_head_mass,
                                masks.cross.layer_head_mass, trace);
    execution.require_range(0, 70);
    write_f32(argv[11], logits);
    std::cout << "WHISPER_CPP23_HEAD_MASK_FORWARD_OK decoder_positions="
              << ids.size() << " logits=" << logits.size()
              << " graph_nodes_visited=" << execution.visited()
              << " output=" << argv[11] << "\n";
    return 0;
  }
  if (argc == 9 && std::string(argv[1]) == "--forward-memory-masked") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    GraphExecutionAudit execution;
    auto memory_values = read_f32(argv[4]);
    whisper_interface::SuppliedEncoderMemory encoder_input{
        {std::span<const float>(memory_values)}};
    if (!encoder_input.hidden.valid())
      throw std::runtime_error("encoder memory ADT invariant");
    auto ids = parse_token_ids(argv[5]);
    whisper_interface::TokenIds decoder_input{{std::move(ids)}};
    auto mask_values = parse_token_ids(argv[6]);
    whisper_interface::DecoderAttentionMask mask{{std::move(mask_values)}};
    auto position_values = parse_token_ids(argv[7]);
    whisper_interface::SuppliedPositionIds positions{
        {std::move(position_values)}};
    if (!decoder_input.tokens.valid() || !mask.keep.valid() ||
        !positions.positions.valid() ||
        mask.keep.values.size() != decoder_input.tokens.values.size() ||
        positions.positions.values.size() != decoder_input.tokens.values.size())
      throw std::runtime_error("masked decoder ADT invariant");
    std::vector<std::pair<std::string, F32>> trace;
    Decoder decoder(tensors, execution);
    auto logits =
        decoder.run_masked(decoder_input.tokens.values, memory_values,
                           mask.keep.values, positions.positions.values, trace);
    execution.require_range(30, 70);
    write_f32(argv[8], logits);
    std::cout << "WHISPER_CPP23_MASKED_FORWARD_OK decoder_positions="
              << decoder_input.tokens.values.size()
              << " logits=" << logits.size()
              << " graph_nodes_visited=" << execution.visited()
              << " output=" << argv[8] << "\n";
    return 0;
  }
  if (argc == 6 && std::string(argv[1]) == "--loss-memory-labels") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    GraphExecutionAudit execution;
    auto memory_values = read_f32(argv[4]);
    whisper_interface::SuppliedEncoderMemory encoder_input{
        {std::span<const float>(memory_values)}};
    if (!encoder_input.hidden.valid())
      throw std::runtime_error("encoder memory ADT invariant");
    auto objective = parse_labels(argv[5]);
    auto decoder_ids = shift_labels_right(objective);
    std::vector<std::pair<std::string, F32>> trace;
    Decoder decoder(tensors, execution);
    auto logits = decoder.run(decoder_ids, memory_values, trace);
    auto loss = labelled_cross_entropy(logits, objective);
    execution.require_range(30, 70);
    std::cout << "WHISPER_CPP23_LABELLED_LOSS_OK positions="
              << objective.labels.size() << " loss=" << std::setprecision(17)
              << loss << " graph_nodes_visited=" << execution.visited() << "\n";
    return 0;
  }
  if (argc == 10 && std::string(argv[1]) == "--cached-step-memory") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    GraphExecutionAudit execution;
    auto memory_values = read_f32(argv[4]);
    whisper_interface::SuppliedEncoderMemory encoder_input{
        {std::span<const float>(memory_values)}};
    if (!encoder_input.hidden.valid())
      throw std::runtime_error("encoder memory ADT invariant");
    auto cache_values = read_f32(argv[5]);
    auto position = std::stoull(argv[6]);
    auto state = deserialize_cache(cache_values, position);
    auto token_values = parse_token_ids(argv[7]);
    if (token_values.size() != 1)
      throw std::runtime_error("cached step requires one token");
    CachedDecoder decoder(tensors, execution);
    auto logits = decoder.step(token_values.front(), memory_values, state);
    execution.require_range(30, 70);
    write_f32(argv[8], logits);
    auto output_cache = serialize_cache(state);
    write_f32(argv[9], output_cache);
    std::cout << "WHISPER_CPP23_CACHED_STEP_OK input_position=" << position
              << " output_position=" << state.position
              << " logits=" << logits.size()
              << " cache_floats=" << output_cache.size()
              << " graph_nodes_visited=" << execution.visited() << "\n";
    return 0;
  }
  if (argc == 8 && std::string(argv[1]) == "--forward-memory-embeddings") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    GraphExecutionAudit execution;
    auto memory_values = read_f32(argv[4]);
    whisper_interface::SuppliedEncoderMemory encoder_input{
        {std::span<const float>(memory_values)}};
    if (!encoder_input.hidden.valid())
      throw std::runtime_error("encoder memory ADT invariant");
    auto embedding_values = read_f32(argv[5]);
    auto positions = std::stoull(argv[6]);
    whisper_interface::SuppliedDecoderEmbeddings decoder_input{
        std::span<const float>(embedding_values), positions};
    if (!decoder_input.valid())
      throw std::runtime_error("decoder embeddings ADT invariant");
    std::vector<std::pair<std::string, F32>> trace;
    Decoder decoder(tensors, execution);
    auto logits = decoder.run_embeddings(
        decoder_input.values, decoder_input.positions, memory_values, trace);
    execution.require_range(30, 70);
    write_f32(argv[7], logits);
    std::cout << "WHISPER_CPP23_EMBEDDINGS_FORWARD_OK decoder_positions="
              << decoder_input.positions << " logits=" << logits.size()
              << " graph_nodes_visited=" << execution.visited()
              << " output=" << argv[7] << "\n";
    return 0;
  }
  if (argc == 7 && std::string(argv[1]) == "--forward-memory-tokens") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    GraphExecutionAudit execution;
    auto memory_values = read_f32(argv[4]);
    whisper_interface::SuppliedEncoderMemory encoder_input{
        {std::span<const float>(memory_values)}};
    if (!encoder_input.hidden.valid())
      throw std::runtime_error("encoder memory ADT invariant");
    auto ids = parse_token_ids(argv[5]);
    whisper_interface::TokenIds decoder_input{{std::move(ids)}};
    if (!decoder_input.tokens.valid())
      throw std::runtime_error("decoder input ADT invariant");
    std::vector<std::pair<std::string, F32>> trace;
    Decoder decoder(tensors, execution);
    auto logits =
        decoder.run(decoder_input.tokens.values, memory_values, trace);
    execution.require_range(30, 70);
    write_f32(argv[6], logits);
    std::cout << "WHISPER_CPP23_MEMORY_FORWARD_OK decoder_positions="
              << decoder_input.tokens.values.size()
              << " logits=" << logits.size()
              << " graph_nodes_visited=" << execution.visited()
              << " output=" << argv[6] << "\n";
    return 0;
  }
  if (argc == 9 && std::string(argv[1]) == "--forward-tokens") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    Decoder decoder(tensors, execution);
    auto ids = parse_token_ids(argv[7]);
    whisper_interface::TokenIds decoder_input{{std::move(ids)}};
    if (!decoder_input.tokens.valid())
      throw std::runtime_error("decoder input ADT invariant");
    auto logits = decoder.run(decoder_input.tokens.values, memory, trace);
    execution.require_range(0, 70);
    write_f32(argv[8], logits);
    std::cout << "WHISPER_CPP23_FORWARD_OK decoder_positions="
              << decoder_input.tokens.values.size()
              << " logits=" << logits.size()
              << " graph_nodes_visited=" << execution.visited()
              << " output=" << argv[8] << "\n";
    return 0;
  }
  if (argc == 9 && std::string(argv[1]) == "--transcribe") {
    TensorStore tensors(argv[2], argv[3]);
    audit_graph(tensors);
    GraphExecutionAudit execution;
    auto mel = execute_frontend(argv[4], read_f32(argv[5]), read_f32(argv[6]),
                                execution);
    std::vector<std::pair<std::string, F32>> trace;
    Encoder encoder(tensors, execution);
    auto memory = encoder.run(mel, trace);
    Decoder decoder(tensors, execution);
    CachedDecoder cached(tensors, execution);
    auto generated = generate(
        decoder, cached, memory, Selection::Greedy, 0.0f, 0,
        std::getenv("WHISPER_VERIFY_RECOMPUTE") != nullptr, {},
        generated_whisper::forced_prefix, whisper_interface::NoTimestamps{},
        nullptr, nullptr, execution);
    TokenDecoder token_decoder(argv[7], argv[8]);
    auto text = token_decoder.decode(generated.tokens);
    while (!text.empty() &&
           std::isspace(static_cast<unsigned char>(text.front())))
      text.erase(text.begin());
    while (!text.empty() &&
           std::isspace(static_cast<unsigned char>(text.back())))
      text.pop_back();
    std::cout << "WHISPER_CPP23_TRANSCRIPT tokens=";
    for (std::size_t i = 0; i < generated.tokens.size(); ++i) {
      if (i)
        std::cout << ',';
      std::cout << generated.tokens[i];
    }
    execution.require_all();
    std::cout << " cache_logit_error=" << generated.max_cache_logit_error
              << " peak_rss_bytes=" << peak_rss_bytes()
              << " graph_nodes_visited=" << execution.visited() << " text=\""
              << text << "\"\n";
    return 0;
  }
  if (argc != 12)
    throw std::runtime_error(
        "usage: whisper_graph_cpp23 checkpoint tensor_manifest wav "
        "mel_reference hann mel_filters encoder_reference decoder_ids "
        "decoder_reference token_manifest token_bytes");
  TensorStore tensors(argv[1], argv[2]);
  audit_graph(tensors);
  auto validated = tensors.validate_all();
  GraphExecutionAudit execution;
  auto mel = execute_frontend(argv[3], read_f32(argv[5]), read_f32(argv[6]),
                              execution);
  auto mel_reference = read_f32(argv[4]);
  auto mel_error = compare(mel, mel_reference);
  std::cout << "log_mel max_abs=" << mel_error.max_abs
            << " rmse=" << mel_error.rmse << " cosine=" << mel_error.cosine
            << "\n";
  auto reference = read_f32(argv[7]);
  if (mel.size() != 80 * 3000)
    throw std::runtime_error("mel shape");
  std::vector<std::pair<std::string, F32>> trace;
  Encoder encoder(tensors, execution);
  auto output = encoder.run(mel, trace);
  std::size_t offset = 0;
  double worst = mel_error.max_abs;
  for (auto &[name, value] : trace) {
    std::span<const float> expected(reference.data() + offset, value.size());
    auto e = compare(value, expected);
    std::cout << name << " max_abs=" << e.max_abs << " rmse=" << e.rmse
              << " cosine=" << e.cosine << "\n";
    worst = std::max(worst, e.max_abs);
    offset += value.size();
  }
  if (offset != reference.size())
    throw std::runtime_error("encoder reference coverage");
  auto ids = read_i32(argv[8]);
  auto decoder_reference = read_f32(argv[9]);
  std::vector<std::pair<std::string, F32>> decoder_trace;
  Decoder decoder(tensors, execution);
  decoder.run(ids, output, decoder_trace);
  offset = 0;
  for (auto &[name, value] : decoder_trace) {
    std::span<const float> expected(decoder_reference.data() + offset,
                                    value.size());
    auto e = compare(value, expected);
    std::cout << name << " max_abs=" << e.max_abs << " rmse=" << e.rmse
              << " cosine=" << e.cosine << "\n";
    worst = std::max(worst, e.max_abs);
    offset += value.size();
  }
  if (offset != decoder_reference.size())
    throw std::runtime_error("decoder reference coverage");
  if (worst > 3e-3)
    throw std::runtime_error("numerical error threshold");
  CachedDecoder cached(tensors, execution);
  auto generated =
      generate(decoder, cached, output, Selection::Greedy, 0.0f, 0, true, {},
               generated_whisper::forced_prefix,
               whisper_interface::NoTimestamps{}, nullptr, nullptr, execution);
  if (!std::equal(generated.tokens.begin(), generated.tokens.end(),
                  generated_whisper::expected_sample_tokens.begin(),
                  generated_whisper::expected_sample_tokens.end()))
    throw std::runtime_error("generated token mismatch");
  CachedDecoder sample_cache(tensors, execution);
  auto sampled =
      generate(decoder, sample_cache, output, Selection::Sample, 1.0f, 0x5eed,
               true, {}, generated_whisper::forced_prefix,
               whisper_interface::NoTimestamps{}, nullptr, nullptr, execution);
  if (generated.max_mass_sum_error > 2e-6 ||
      generated.max_selected_mass_error > 2e-4 ||
      generated.max_cache_logit_error > 3e-3 ||
      sampled.max_mass_sum_error > 2e-6 || sampled.max_cache_logit_error > 3e-3)
    throw std::runtime_error("probability/cache verification");
  TokenDecoder tokens(argv[10], argv[11]);
  auto text = tokens.decode(generated.tokens);
  while (!text.empty() &&
         std::isspace(static_cast<unsigned char>(text.front())))
    text.erase(text.begin());
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())))
    text.pop_back();
  const std::string expected =
      "Mr. Quilter is the apostle of the middle classes, and we are glad to "
      "welcome his gospel.";
  if (text != expected)
    throw std::runtime_error("decoded text mismatch: " + text);
  execution.require_all();
  std::cout << "WHISPER_CPP23_WAV_TO_TEXT_CACHED_PROBABILISTIC_OK graph_nodes="
            << generated_whisper::nodes.size() << " tensors=" << validated
            << " generated_tokens=" << generated.tokens.size()
            << " sampled_tokens=" << sampled.tokens.size()
            << " cache_positions=" << generated.cache_positions
            << " cache_logit_error=" << generated.max_cache_logit_error
            << " mass_sum_error=" << generated.max_mass_sum_error
            << " selected_mass_error=" << generated.max_selected_mass_error
            << " mel_max_abs=" << mel_error.max_abs
            << " worst_max_abs=" << worst
            << " graph_nodes_visited=" << execution.visited()
            << " transcript=\"" << text << "\"\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << "WHISPER_CPP23_GRAPH_FAIL " << e.what() << "\n";
  return 1;
}
