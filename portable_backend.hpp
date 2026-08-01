#pragma once

// Minimal, vendor-neutral binary32 backend used by whisper_graph_cpp23.cpp.
// Contract: row-major matrices, alpha/beta are finite, and only the two
// transpose combinations exercised by the generated Whisper graph are legal.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

enum CBLAS_ORDER { CblasRowMajor = 101 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112 };

namespace whisper_portable {
enum class Primitive { Gemm, Dot };
enum class UnsupportedPrimitive { VendorVectorIntrinsics, FftLibrary, Int8Gemm };

struct BackendContract {
  static constexpr const char *name = "portable-scalar-f32";
  static constexpr std::uint32_t abi_version = 1;
  static constexpr bool ieee_binary32 = true;
  static constexpr bool deterministic_order = true;
};

inline void require(bool ok, const char *what) {
  if (!ok) throw std::invalid_argument(what);
}

inline float cblas_sdot(int n, const float *x, int incx, const float *y,
                       int incy) {
  require(n >= 0 && incx > 0 && incy > 0, "portable dot contract");
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) sum = sum + x[i * incx] * y[i * incy];
  return sum;
}

inline void cblas_sgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE ta,
                        CBLAS_TRANSPOSE tb, int m, int n, int k, float alpha,
                        const float *a, int lda, const float *b, int ldb,
                        float beta, float *c, int ldc) {
  require(order == CblasRowMajor && m >= 0 && n >= 0 && k >= 0,
          "portable gemm shape/order contract");
  require((ta == CblasNoTrans || ta == CblasTrans) &&
              (tb == CblasNoTrans || tb == CblasTrans),
          "portable gemm transpose contract");
  const int a_cols = ta == CblasNoTrans ? k : m;
  const int b_cols = tb == CblasNoTrans ? n : k;
  require(lda >= a_cols && ldb >= b_cols && ldc >= n,
          "portable gemm stride contract");
  require(std::isfinite(alpha) && std::isfinite(beta),
          "portable gemm scalar contract");
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int p = 0; p < k; ++p) {
        const float av = ta == CblasNoTrans ? a[i * lda + p] : a[p * lda + i];
        const float bv = tb == CblasNoTrans ? b[p * ldb + j] : b[j * ldb + p];
        sum = sum + av * bv;
      }
      c[i * ldc + j] = alpha * sum + beta * c[i * ldc + j];
    }
  }
}
} // namespace whisper_portable

using whisper_portable::cblas_sdot;
using whisper_portable::cblas_sgemm;
