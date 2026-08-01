#include "portable_backend.hpp"
#include <bit>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

static bool close(float a, float b) {
  return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
}
int main() {
  // The reference uses the same left-to-right binary32 recurrence required by
  // the backend contract; this is an executable correspondence obligation.
  float x[6] = {1, 2, 3, 4, 5, 6};
  float y[6] = {2, 1, 4, 1, 3, 2};
  float expected = 0.0f;
  for (int i = 0; i < 6; ++i) expected = expected + x[i] * y[i];
  if (!close(cblas_sdot(6, x, 1, y, 1), expected)) return 1;
  float a[6] = {1, 2, 3, 4, 5, 6};
  float b[6] = {2, 1, 4, 1, 3, 2};
  float c[4] = {};
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 2, 2, 3, 1, a, 3,
              b, 3, 0, c, 2);
  const float ref[4] = {16, 13, 37, 31};
  for (int i = 0; i < 4; ++i)
    if (!close(c[i], ref[i])) return 2;
  auto start = std::chrono::steady_clock::now();
  std::vector<float> m(128 * 128), n(128 * 128), o(128 * 128);
  for (int r = 0; r < 20; ++r)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128, 128, 128, 1,
                m.data(), 128, n.data(), 128, 0, o.data(), 128);
  auto ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
  std::cout << "PORTABLE_BACKEND_OK contract="
            << whisper_portable::BackendContract::name
            << " correspondence=dot,gemm bitwise=PASS gemm_20x128_ms=" << ms
            << " working_set_bytes=" << (3 * m.size() * sizeof(float)) << '\n';
}
