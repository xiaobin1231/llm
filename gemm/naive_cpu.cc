#include "gemm/launch_template.h"

namespace gemm {

void GemmCpuImpl(const MatrixDim& dim, const float* A, const float* B, float* C) {
  for (std::size_t m = 0u; m < dim.M; m++) {
    for (std::size_t n = 0u; n < dim.N; n++) {
      float s = 0.0f;
      for (std::size_t k = 0u; k < dim.K; k++) {
        s += A[m * dim.K + k] * B[k * dim.N + n];
      }
      C[m * dim.N + n] = s;
    }
  }
}

template<>
void GemmExec<float, float, k_naive_cpu_impl>(const MatrixDim& dim, const float* A, const float* B, float* C, cudaStream_t stream) {
  (void) stream;
  GemmCpuImpl(dim, A, B, C);
}

}  // namespace gemm
