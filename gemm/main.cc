#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "gemm/launch_template.h"

namespace {
inline float RandUniform(float low, float high) {
  float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  return low + r * (high - low);
}

void InitializeMatrix(float* data, size_t n, float low, float high) {
  for (size_t i = 0; i < n; i++) {
    data[i] = RandUniform(low, high);
  }
}
}  // namespace

int main(int argc, char* argv[]) {
  constexpr std::size_t M = 1024u;
  constexpr std::size_t N = 4096u;
  constexpr std::size_t K = 2048u;

  using scalar_t = float;

  scalar_t* h_A = static_cast<scalar_t*>(malloc(M * K * sizeof(scalar_t)));
  InitializeMatrix(h_A, M * K, -1, 1);
  scalar_t* h_B = static_cast<scalar_t*>(malloc(K * N * sizeof(scalar_t)));
  InitializeMatrix(h_B, K * N, -1, 1);

  scalar_t* h_C_cpu = static_cast<scalar_t*>(malloc(M * N * sizeof(scalar_t)));
  scalar_t* h_C_gpu = static_cast<scalar_t*>(malloc(M * N * sizeof(scalar_t)));

  scalar_t* d_A;
  scalar_t* d_B;
  scalar_t* d_C;
  cudaMalloc(&d_A, M * K * sizeof(scalar_t));
  cudaMalloc(&d_B, K * N * sizeof(scalar_t));
  cudaMalloc(&d_C, M * N * sizeof(scalar_t));
  cudaMemset(&d_C, 0, M * N * sizeof(scalar_t));

  cudaMemcpy(d_A, h_A, M * K * sizeof(scalar_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(scalar_t), cudaMemcpyHostToDevice);

  gemm::MatrixDim dim(M, N, K);
  gemm::GemmExec<scalar_t, gemm::k_naive_cpu_impl>(dim, h_A, h_B, h_C_cpu);

  gemm::GemmExec<scalar_t, gemm::k_naive_cuda_impl>(dim, d_A, d_B, d_C);
  cudaDeviceSynchronize();
  cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(scalar_t), cudaMemcpyDeviceToHost);

  bool match = true;
  for (std::size_t n = 0u; n < N; n++) {
    if (std::fabs(h_C_cpu[n] - h_C_gpu[n]) >= 1e-4) {
      match = false;
    }
  }

  if (match) {
    printf("Gpu and Cpu result are matched.\n");
  } else {
    printf("Gpu and Cpu result not match.\n");
  }

  free(h_A);
  free(h_B);
  free(h_C_cpu);
  free(h_C_gpu);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
