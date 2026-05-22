#ifndef GEMM_KERNEL_IMPL_H_
#define GEMM_KERNEL_IMPL_H_

#include <cstdlib>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace gemm {

template<typename scalar_t>
__global__ void GemmNaiveCudaImplKernel(
    const scalar_t* A,
    const scalar_t* B,
    scalar_t* C,
    uint32_t M,
    uint32_t N,
    uint32_t K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) {
    return;
  }

  scalar_t s = static_cast<scalar_t>(0);
  for (int k = 0; k < K; k++) {
    s += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = s;
}

}  // namespace gemm

#endif  // GEMM_KERNEL_IMPL_H_
