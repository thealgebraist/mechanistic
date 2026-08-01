#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
constexpr std::uint16_t states = 7;
constexpr std::uint16_t tokens = 8;
constexpr std::uint16_t denominator = 16;
constexpr std::array<const char*, tokens> token_name{
    "key", "keys", "is", "are", "found", "missing", ".", "<eos>"};
constexpr std::array<const char*, states> state_name{
    "START", "SINGULAR", "PLURAL", "PREDICATE_A", "PREDICATE_B",
    "TERMINAL_A", "TERMINAL_B"};

// Exact categorical masses.  Each row sums to denominator.
constexpr std::array<std::array<std::uint16_t, tokens>, states> emission{{
    {{8, 8, 0, 0, 0, 0, 0, 0}},
    {{0, 0, 14, 0, 1, 1, 0, 0}},
    {{0, 0, 0, 14, 1, 1, 0, 0}},
    {{0, 0, 0, 0, 10, 5, 1, 0}},
    {{0, 0, 0, 0, 10, 5, 1, 0}},
    {{0, 0, 0, 0, 0, 0, 16, 0}},
    {{0, 0, 0, 0, 0, 0, 16, 0}},
}};

// Token-conditioned one-hot recurrent transition h' = T_token h.  States 3/4
// and 5/6 are not byte-identical, but become equivalent after quotienting.
constexpr std::array<std::array<std::uint16_t, tokens>, states> successor{{
    {{1, 2, 5, 5, 5, 5, 5, 5}},
    {{5, 5, 3, 5, 5, 5, 5, 5}},
    {{6, 6, 6, 4, 6, 6, 6, 6}},
    {{5, 5, 5, 5, 5, 5, 5, 5}},
    {{6, 6, 6, 6, 6, 6, 6, 6}},
    {{5, 5, 5, 5, 5, 5, 5, 5}},
    {{6, 6, 6, 6, 6, 6, 6, 6}},
}};

void put16(std::ofstream& out, std::uint16_t value) {
  const char bytes[2]{static_cast<char>(value & 255),
                      static_cast<char>((value >> 8) & 255)};
  out.write(bytes, 2);
}

void put_string(std::ofstream& out, const std::string& value) {
  if (value.size() > 255) throw std::runtime_error("label too long");
  out.put(static_cast<char>(value.size()));
  out.write(value.data(), static_cast<std::streamsize>(value.size()));
}
}  // namespace

int main(int argc, char** argv) try {
  const std::string path = argc == 2 ? argv[1] : "outputs/toy_text_model.ptm";
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot open output");
  out.write("PTM1", 4);
  put16(out, 1);             // format version
  put16(out, states);
  put16(out, tokens);
  put16(out, denominator);
  put16(out, 0);             // initial state
  for (const auto* name : token_name) put_string(out, name);
  for (const auto* name : state_name) put_string(out, name);
  for (const auto& row : emission)
    for (const auto value : row) put16(out, value);
  for (const auto& row : successor)
    for (const auto value : row) put16(out, value);
  out.close();
  if (!out) throw std::runtime_error("write failed");
  std::cout << "TOY_TEXT_MODEL_BINARY_OK states=" << states
            << " tokens=" << tokens << " denominator=" << denominator << '\n';
  return 0;
} catch (const std::exception& error) {
  std::cerr << "TOY_TEXT_MODEL_BINARY_FAIL " << error.what() << '\n';
  return 1;
}
