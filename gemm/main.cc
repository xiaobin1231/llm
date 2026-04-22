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

  float* MatrixA = static_cast<float*>(malloc(M * K * sizeof(float)));
  InitializeMatrix(MatrixA, M * K, -1, 1);
  float* MatrixB = static_cast<float*>(malloc(K * N * sizeof(float)));
  InitializeMatrix(MatrixB, K * N, -1, 1);
  float* MatrixC = static_cast<float*>(malloc(M * N * sizeof(float)));
  InitializeMatrix(MatrixC, M * N, -1, 1);

  gemm::MatrixDim dim(M, N, K);
  gemm::GemmExec<float, gemm::k_naive_cpu_impl>(dim, MatrixA, MatrixB, MatrixC);

  printf("MatrixC[0]:\n");
  for (std::size_t n = 0u; n < N; n++) {
    printf("%.2f, ", MatrixC[n]);
  }
  printf("\n");

  free(MatrixA);
  free(MatrixB);
  free(MatrixC);

  return 0;
}
