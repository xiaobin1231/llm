#ifndef GEMM_LAUNCH_TEMPLATE_H_
#define GEMM_LAUNCH_TEMPLATE_H_

#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace gemm {

enum GemmImplOperation : std::size_t {
  k_naive_cpu_impl = 0,
  k_naive_cuda_impl = 1,
  k_tensor_core_mma_impl = 2,
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

void GemmCpuImpl(const MatrixDim& dim, const float* A, const float* B, float* C);

template<typename scalar_ab, typename scalar_c,  GemmImplOperation op>
void GemmExec(const MatrixDim& dim, const scalar_ab* A, const scalar_ab* B, scalar_c* C, cudaStream_t stream = nullptr);

}  // namespace gemm

#endif  // GEMM_LAUNCH_TEMPLATE_H_
