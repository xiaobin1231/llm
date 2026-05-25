#include "gemm/kernel_impl.h"
#include "gemm/launch_template.h"

namespace gemm {

template<>
void GemmExec<half, float, k_tensor_core_mma_impl>(const MatrixDim& dim, const half* A, const half* B, float* C, cudaStream_t stream) {
  constexpr int BLOCK_M = 128;
  constexpr int BLOCK_N = 128;
  constexpr int BLOCK_K = 32;

  constexpr int WARP_TILE_M = 64;
  constexpr int WARP_TILE_N = 32;

  constexpr int kWarps = 8;
  constexpr int kThreads = kWarps * 32;

  dim3 blockDim(kThreads);
  dim3 gridDim((dim.N + BLOCK_N - 1) / BLOCK_N, (dim.M + BLOCK_M - 1) / BLOCK_M);
  auto kernel = &GemmMMAImpKernel_m16n8k16fp32fp16fp16fp32<BLOCK_M, BLOCK_N, BLOCK_K, WARP_TILE_M, WARP_TILE_N, kWarps>;
  kernel<<<gridDim, blockDim, 0, stream>>>(A, B, C, dim.M, dim.N, dim.K);
}

}  // namespace gemm
