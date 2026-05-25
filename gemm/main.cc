#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "gemm/launch_template.h"

namespace {
inline float RandUniform(float low, float high) {
  float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  return low + r * (high - low);
}

void InitializeMatrix(float* data, std::size_t n, float low, float high) {
  for (std::size_t i = 0; i < n; i++) {
    data[i] = RandUniform(low, high);
  }
}
void InitializeMatrixWithFP32(half* dst, float* src, std::size_t n) {
  for (std::size_t i = 0; i < n; i++) {
    dst[i] = __float2half(src[i]);
  }
}
}  // namespace

int main(int argc, char* argv[]) {
  constexpr std::size_t M = 1024u;
  constexpr std::size_t N = 4096u;
  constexpr std::size_t K = 2048u;

  // FP32
  float* host_fp32_A = static_cast<float*>(malloc(M * K * sizeof(float)));
  InitializeMatrix(host_fp32_A, M * K, -1.0f, 1.0f);
  float* host_fp32_B = static_cast<float*>(malloc(K * N * sizeof(float)));
  InitializeMatrix(host_fp32_B, K * N, -1.0f, 1.0f);

  float* device_fp32_A;
  float* device_fp32_B;
  cudaMalloc(&device_fp32_A, M * K * sizeof(float));
  cudaMalloc(&device_fp32_B, K * N * sizeof(float));
  cudaMemcpy(device_fp32_A, host_fp32_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_fp32_B, host_fp32_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

  // FP16
  half* host_fp16_A = static_cast<half*>(malloc(M * K * sizeof(half)));
  InitializeMatrixWithFP32(host_fp16_A, host_fp32_A, M * K);
  half* host_fp16_B = static_cast<half*>(malloc(K * N * sizeof(half)));
  InitializeMatrixWithFP32(host_fp16_B, host_fp32_B, K * N);

  half* device_fp16_A;
  half* device_fp16_B;
  cudaMalloc(&device_fp16_A, M * K * sizeof(half));
  cudaMalloc(&device_fp16_B, K * N * sizeof(half));
  cudaMemcpy(device_fp16_A, host_fp16_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(device_fp16_B, host_fp16_B, K * N * sizeof(half), cudaMemcpyHostToDevice);

  float* d_C;
  cudaMalloc(&d_C, M * N * sizeof(float));
  cudaMemset(d_C, 0, M * N * sizeof(float));

  gemm::MatrixDim dim(M, N, K);

  // gemm::GemmExec<float, float, gemm::k_naive_cuda_impl>(dim, device_fp32_A, device_fp32_B, d_C);
  gemm::GemmExec<half, float, gemm::k_tensor_core_mma_impl>(dim, device_fp16_A, device_fp16_B, d_C);

  float* h_C_cpu = static_cast<float*>(malloc(M * N * sizeof(float)));
  gemm::GemmExec<float, float, gemm::k_naive_cpu_impl>(dim, host_fp32_A, host_fp32_B, h_C_cpu);

  cudaDeviceSynchronize();
  float* h_C_gpu = static_cast<float*>(malloc(M * N * sizeof(float)));
  cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  bool match = true;
  for (std::size_t n = 0u; n < N; n++) {
    if (std::fabs(h_C_cpu[n] - h_C_gpu[n]) >= 5e-2) {
      match = false;
      break;
    }
  }

  if (match) {
    printf("Gpu and Cpu result are matched.\n");
  } else {
    printf("Gpu and Cpu result not match.\n");
  }

  free(host_fp32_A);
  free(host_fp32_B);
  free(host_fp16_A);
  free(host_fp16_B);
  free(h_C_cpu);
  free(h_C_gpu);
  cudaFree(device_fp32_A);
  cudaFree(device_fp32_B);
  cudaFree(device_fp16_A);
  cudaFree(device_fp16_B);
  cudaFree(d_C);

  return 0;
}
