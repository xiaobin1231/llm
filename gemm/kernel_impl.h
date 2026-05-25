#ifndef GEMM_KERNEL_IMPL_H_
#define GEMM_KERNEL_IMPL_H_

#include <cstdlib>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace gemm {

__device__ __forceinline__ int GetSwizzleCol(int row, int col) {
  return col ^ ((row & 3) * 8);
}

template<bool apply_shm, int tile_size>
__global__ void GemmNaiveCudaImplKernel(
    const float* A,
    const float* B,
    float* C,
    uint32_t M,
    uint32_t N,
    uint32_t K) {
  if constexpr (apply_shm) {
    int row = blockIdx.y * tile_size + threadIdx.y;
    int col = blockIdx.x * tile_size + threadIdx.x;
    if (row >= M || col >= N) {
      return;
    }

    __shared__ float A_shm[tile_size][tile_size];
    __shared__ float B_shm[tile_size][tile_size];

    float s = 0.0f;
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

    float s = 0.0f;
    for (int k = 0; k < K; k++) {
      s += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = s;
  }
}

template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_TILE_M, int WARP_TILE_N, int kWarps>
__global__ void GemmMMAImpKernel_m16n8k16fp32fp16fp16fp32(
    const half* A,
    const half* B,
    float* C,
    uint32_t M,
    uint32_t N,
    uint32_t K) {
  constexpr int kThreads = kWarps * 32;
  constexpr int kWarpRows = BLOCK_M / WARP_TILE_M;
  constexpr int kWarpCols = BLOCK_N / WARP_TILE_N;
  static_assert((kWarps == kWarpRows * kWarpCols), "Hyper parameter kWarps not equals kWarpRows * kWarpCols.");

  constexpr int kAtomicMMA_m = 16;
  constexpr int kAtomicMMA_n = 8;
  constexpr int kAtomicMMA_k = 16;

  constexpr int M_TILE = WARP_TILE_M / kAtomicMMA_m;
  constexpr int N_TILE = WARP_TILE_N / kAtomicMMA_n;

  int block_row = blockIdx.y * BLOCK_M;
  int block_col = blockIdx.x * BLOCK_N;

  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;

  int warp_row_idx = warp_id / kWarpCols;
  int warp_col_idx = warp_id % kWarpCols;

  __shared__ half smem_A[BLOCK_M * BLOCK_K];
  __shared__ half smem_B[BLOCK_K * BLOCK_N];

  constexpr int kRegCPerThread = kAtomicMMA_m * kAtomicMMA_n / 32;
  float reg_c[M_TILE][N_TILE][kRegCPerThread] = {0.0f};

  for (int kidx = 0; kidx < (K + BLOCK_K - 1) / BLOCK_K; kidx++) {
    // Naive loading method.
    // constexpr int kLoadAToShmCount = BLOCK_M * BLOCK_K / kThreads;
    // for (int i = 0; i < kLoadAToShmCount; i++) {
    //   int m = (i * kThreads + tid) / BLOCK_K;
    //   int n = (i * kThreads + tid) % BLOCK_K;
    //   smem_A[m * BLOCK_K + n] = A[(block_row + m) * K + kidx * BLOCK_K + n];
    // }
    // constexpr int kLoadBToShmCount = BLOCK_K * BLOCK_N / kThreads;
    // for (int i = 0; i < kLoadBToShmCount; i++) {
    //   int m = (i * kThreads + tid) / BLOCK_N;
    //   int n = (i * kThreads + tid) % BLOCK_N;
    //   smem_B[m * BLOCK_N + n] = B[(kidx * BLOCK_K + m) * N + block_col + n];
    // }

    // Vectorized loading method.
    constexpr int kVecSize = 8;
    constexpr int kCarryACount = BLOCK_M * BLOCK_K / (kThreads * kVecSize);
    constexpr int kSmemACols = BLOCK_K / kVecSize;
    #pragma unroll
    for (int i = 0; i < kCarryACount; i++) {
      int flatten_idx = i * kThreads + tid;
      int r = flatten_idx / kSmemACols;
      int c = flatten_idx % kSmemACols;
      const float4* src = reinterpret_cast<const float4*>(&A[(block_row + r) * K + kidx * BLOCK_K + c * kVecSize]);
      float4* dst = reinterpret_cast<float4*>(&smem_A[r * BLOCK_K + GetSwizzleCol(r, c * kVecSize)]);
      *dst = *src;
    }
    constexpr int kCarryBCount = (BLOCK_K * BLOCK_N) / (kThreads * kVecSize);
    constexpr int kSmemBCols = BLOCK_N / kVecSize;
    #pragma unroll
    for (int i = 0; i < kCarryBCount; i++) {
      int flatten_idx = i * kThreads + tid;
      int r = flatten_idx / kSmemBCols;
      int c = flatten_idx % kSmemBCols;
      const float4* src = reinterpret_cast<const float4*>(&B[(kidx * BLOCK_K + r) * N + block_col + c * kVecSize]);
      float4* dst = reinterpret_cast<float4*>(&smem_B[r * BLOCK_N + GetSwizzleCol(r, c * kVecSize)]);
      *dst = *src;
    }

    __syncthreads();

    constexpr int kRegAPerThread = (kAtomicMMA_m * kAtomicMMA_k / 32) / 2;
    constexpr int kRegBPerThread = (kAtomicMMA_k * kAtomicMMA_n / 32) / 2;
    uint32_t reg_a[M_TILE][kRegAPerThread];
    uint32_t reg_b[N_TILE][kRegBPerThread];

    for (int kfrag = 0; kfrag < BLOCK_K; kfrag += kAtomicMMA_k) {
      #pragma unroll
      for (int m = 0; m < M_TILE; m++) {
        int row = warp_row_idx * WARP_TILE_M + m * kAtomicMMA_m + (lane_id % 16);
        int col = kfrag + (lane_id / 16) * 8;
        uint32_t smem_Aptr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_A[row * BLOCK_K + GetSwizzleCol(row, col)]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(reg_a[m][0]), "=r"(reg_a[m][1]), "=r"(reg_a[m][2]), "=r"(reg_a[m][3])
                     : "r"(smem_Aptr));
      }
      #pragma unroll
      for (int n = 0; n < N_TILE; n++) {
        int row = kfrag + (lane_id % 16);
        int col = warp_col_idx * WARP_TILE_N + n * kAtomicMMA_n;
        uint32_t smem_Bptr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_B[row * BLOCK_N + GetSwizzleCol(row, col)]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                     : "=r"(reg_b[n][0]), "=r"(reg_b[n][1]) : "r"(smem_Bptr));
      }

      #pragma unroll
      for (int m = 0; m < M_TILE; m++) {
        #pragma unroll
        for (int n = 0; n < N_TILE; n++) {
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                       : "=f"(reg_c[m][n][0]), "=f"(reg_c[m][n][1]), "=f"(reg_c[m][n][2]), "=f"(reg_c[m][n][3])
                       : "r"(reg_a[m][0]), "r"(reg_a[m][1]), "r"(reg_a[m][2]), "r"(reg_a[m][3]),
                         "r"(reg_b[n][0]), "r"(reg_b[n][1]),
                         "f"(reg_c[m][n][0]), "f"(reg_c[m][n][1]), "f"(reg_c[m][n][2]), "f"(reg_c[m][n][3]));
        }
      }
    }

    __syncthreads();
  }

  int r0 = lane_id / 4;
  int r1 = r0 + 8;
  int c0 = (lane_id % 4) * 2;
  int c1 = c0 + 1;

  for (int m = 0; m < M_TILE; ++m) {
    for (int n = 0; n < N_TILE; ++n) {
      int g_r0 = block_row + warp_row_idx * WARP_TILE_M + m * kAtomicMMA_m + r0;
      int g_r1 = block_row + warp_row_idx * WARP_TILE_M + m * kAtomicMMA_m + r1;
      int g_c0 = block_col + warp_col_idx * WARP_TILE_N + n * kAtomicMMA_n + c0;
      int g_c1 = block_col + warp_col_idx * WARP_TILE_N + n * kAtomicMMA_n + c1;

      C[g_r0 * N + g_c0] = reg_c[m][n][0];
      C[g_r0 * N + g_c1] = reg_c[m][n][1];
      C[g_r1 * N + g_c0] = reg_c[m][n][2];
      C[g_r1 * N + g_c1] = reg_c[m][n][3];
    }
  }
}

}  // namespace gemm

#endif  // GEMM_KERNEL_IMPL_H_
