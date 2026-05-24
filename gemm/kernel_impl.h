#ifndef GEMM_KERNEL_IMPL_H_
#define GEMM_KERNEL_IMPL_H_

#include <cstdlib>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace gemm {

template<typename scalar_t, bool apply_shm, int tile_size>
__global__ void GemmNaiveCudaImplKernel(
    const scalar_t* A,
    const scalar_t* B,
    scalar_t* C,
    uint32_t M,
    uint32_t N,
    uint32_t K) {
  if constexpr (apply_shm) {
    __shared__ scalar_t A_shm[tile_size][tile_size];
    __shared__ scalar_t B_shm[tile_size][tile_size];

    int row = blockIdx.y * tile_size + threadIdx.y;
    int col = blockIdx.x * tile_size + threadIdx.x;
    if (row >= M || col >= N) {
      return;
    }

    scalar_t s = static_cast<scalar_t>(0);
    for (int bidx = 0; bidx < (K + tile_size - 1) / tile_size; bidx++) {
      int A_col_idx = bidx * tile_size + threadIdx.x;
      if (A_col_idx < K) {
        A_shm[threadIdx.y][threadIdx.x] = A[row * K + A_col_idx];
      }
      int B_row_idx = bidx * tile_size + threadIdx.y;
      if (B_row_idx < K) {
        B_shm[threadIdx.y][threadIdx.x] = B[B_row_idx * N + col];
      }
      __syncthreads();

      for (int k = 0; k < tile_size; k++) {
        s += A_shm[threadIdx.y][k] * B_shm[k][threadIdx.x];
      }

      __syncthreads();
    }

    C[row * N + col] = s;
  } else {
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
}

}  // namespace gemm

#endif  // GEMM_KERNEL_IMPL_H_
