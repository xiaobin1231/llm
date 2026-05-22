#ifndef GEMM_LAUNCH_TEMPLATE_H_
#define GEMM_LAUNCH_TEMPLATE_H_

#include <cstdlib>
#include <cuda_runtime.h>

namespace gemm {

enum GemmImplOperation : std::size_t {
  k_naive_cpu_impl = 0,
  k_naive_cuda_impl = 1,
  k_cuda_mma_impl = 2,
  k_cuda_wmma_impl = 3,
  k_cuda_wgmma_impl = 4,
  k_cuda_cute_impl = 5,
  k_cuda_multi_stage_cute_impl = 6,
};

struct MatrixDim {
  explicit MatrixDim(std::size_t m, std::size_t n, std::size_t k)
    : M(m), N(n), K(k) {}

  std::size_t M;
  std::size_t N;
  std::size_t K;
};

template<typename scalar_t>
void GemmCpuImpl(const MatrixDim& dim, const scalar_t* A, const scalar_t* B, scalar_t* C) {
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

template<typename scalar_t, GemmImplOperation op>
void GemmExec(const MatrixDim& dim, const scalar_t* A, const scalar_t* B, scalar_t* C, cudaStream_t stream = nullptr);

}  // namespace gemm

#endif  // GEMM_LAUNCH_TEMPLATE_H_
