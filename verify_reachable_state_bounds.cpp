#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Rec {
  std::string name, kind;
  std::uint64_t begin{}, end{}, rows{}, cols{};
  long double bound{};
};

static std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> out; std::stringstream ss(s); std::string x;
  while (std::getline(ss, x, c)) out.push_back(x);
  return out;
}

static long double json_number(const std::string& text, const std::string& key) {
  const auto p = text.find("\"" + key + "\"");
  if (p == std::string::npos) throw std::runtime_error("missing JSON key " + key);
  const auto colon = text.find(':', p);
  const auto begin = text.find_first_of("-0123456789", colon + 1);
  const auto end = text.find_first_not_of("+-.0123456789eE", begin);
  return std::stold(text.substr(begin, end - begin));
}

int main(int argc, char** argv) try {
  if (argc != 4) throw std::runtime_error("usage: verifier model.safetensors manifest.tsv bounds.json");
  std::ifstream mf(argv[2]); if (!mf) throw std::runtime_error("cannot open manifest");
  std::string line; std::getline(mf, line);
  std::vector<Rec> records;
  while (std::getline(mf, line)) {
    auto f = split(line, '\t'); if (f.size() != 7) throw std::runtime_error("bad manifest row");
    records.push_back({f[0], f[1], std::stoull(f[2]), std::stoull(f[3]),
                       std::stoull(f[4]), std::stoull(f[5]), std::stold(f[6])});
  }
  std::ifstream bin(argv[1], std::ios::binary); if (!bin) throw std::runtime_error("cannot open checkpoint");
  std::map<std::string,long double> norm;
  long double worst_slack = std::numeric_limits<long double>::infinity();
  for (const auto& r : records) {
    if (r.end - r.begin != r.rows * r.cols * 4) throw std::runtime_error("shape/offset mismatch " + r.name);
    bin.seekg(static_cast<std::streamoff>(r.begin));
    std::vector<std::uint32_t> bits(r.rows * r.cols);
    bin.read(reinterpret_cast<char*>(bits.data()), static_cast<std::streamsize>(bits.size() * 4));
    if (!bin) throw std::runtime_error("checkpoint read failed " + r.name);
    long double actual = 0;
    if (r.kind == "maxabs") {
      for (auto u : bits) actual = std::max(actual, std::fabs(static_cast<long double>(std::bit_cast<float>(u))));
    } else if (r.kind == "row_l1") {
      for (std::uint64_t row = 0; row < r.rows; ++row) {
        long double sum = 0;
        for (std::uint64_t col = 0; col < r.cols; ++col)
          sum += std::fabs(static_cast<long double>(std::bit_cast<float>(bits[row*r.cols+col])));
        actual = std::max(actual, sum);
      }
    } else throw std::runtime_error("unknown norm kind");
    if (!(actual <= r.bound)) throw std::runtime_error("non-conservative norm bound " + r.name);
    worst_slack = std::min(worst_slack, r.bound - actual);
    norm[r.name] = r.bound;
  }
  const long double sd = std::sqrt(512.0L);
  auto rms = [&](const std::string& w) { return sd * norm.at(w); };
  auto att = [&](const std::string& p, long double x) {
    return norm.at(p+".o.weight") * norm.at(p+".v.weight") * x;
  };
  auto mlp = [&](const std::string& p, long double x) {
    return norm.at(p+".wo.weight") * norm.at(p+".wi_0.weight") * x * norm.at(p+".wi_1.weight") * x;
  };
  long double enc = norm.at("shared.weight");
  for (int i=0;i<8;++i) {
    auto p="encoder.block."+std::to_string(i);
    enc += att(p+".layer.0.SelfAttention", rms(p+".layer.0.layer_norm.weight"));
    enc += mlp(p+".layer.1.DenseReluDense", rms(p+".layer.1.layer_norm.weight"));
  }
  const long double memory = rms("encoder.final_layer_norm.weight");
  long double dec = norm.at("shared.weight");
  for (int i=0;i<8;++i) {
    auto p="decoder.block."+std::to_string(i);
    dec += att(p+".layer.0.SelfAttention", rms(p+".layer.0.layer_norm.weight"));
    dec += att(p+".layer.1.EncDecAttention", memory);
    dec += mlp(p+".layer.2.DenseReluDense", rms(p+".layer.2.layer_norm.weight"));
  }
  const long double readout = rms("decoder.final_layer_norm.weight");
  const long double logits = norm.at("lm_head.weight") * readout;
  std::ifstream jf(argv[3]); std::string js((std::istreambuf_iterator<char>(jf)), {});
  if (enc > json_number(js,"encoder_hidden") || memory > json_number(js,"encoder_memory") || dec > json_number(js,"decoder_hidden") ||
      readout > json_number(js,"readout_hidden") || logits > json_number(js,"logit_abs"))
    throw std::runtime_error("recurrence exceeds serialized certificate");
  std::cout << "REACHABLE_STATE_BOUNDS_CPP23_OK tensors=" << records.size()
            << " logit_abs=" << static_cast<double>(logits)
            << " min_norm_slack=" << static_cast<double>(worst_slack) << "\n";
  return 0;
} catch (const std::exception& e) {
  std::cerr << "REACHABLE_STATE_BOUNDS_CPP23_FAIL " << e.what() << "\n"; return 1;
}
