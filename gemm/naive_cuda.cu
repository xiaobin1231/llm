#include "gemm/kernel_impl.h"
#include "gemm/launch_template.h"

namespace gemm {

template<>
void GemmExec<float, k_naive_cuda_impl>(const MatrixDim& dim, const float* A, const float* B, float* C, cudaStream_t stream) {
  constexpr int BLOCK_SIZE = 16;
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((dim.N + blockDim.x - 1) / blockDim.x, (dim.M + blockDim.y - 1) / blockDim.y);
  auto kernel = &GemmNaiveCudaImplKernel<float, true, BLOCK_SIZE>;
  kernel<<<gridDim, blockDim, 0, stream>>>(A, B, C, dim.M, dim.N, dim.K);
}

// template<>
// void GemmExec<__half, k_naive_cuda_impl>(const MatrixDim& dim, const __half* A, const __half* B, __half* C, cudaStream_t stream) {
//   dim3 blockDim(16, 16);
//   dim3 gridDim((dim.N + blockDim.x - 1) / blockDim.x, (dim.M + blockDim.y - 1) / blockDim.y);
//   auto kernel = &GemmNaiveCudaImplKernel<__half>;
//   kernel<<<gridDim, blockDim, 0, stream>>>(A, B, C, dim.M, dim.N, dim.K);
// }

}  // namespace gemm
