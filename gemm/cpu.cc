#include <cstdlib>
#include "gemm/launch_template.h"

namespace gemm {

template<>
void GemmExec<float, k_naive_cpu_impl>(const MatrixDim& dim, const float* A, const float* B, float* C) {
  GemmCpuImpl<float>(dim, A, B, C);
}

template<>
void GemmExec<__half, k_naive_cpu_impl>(const MatrixDim& dim, const float* A, const float* B, float* C) {
  GemmCpuImpl<__half>(dim, A, B, C);
}

}  // namespace gemm
