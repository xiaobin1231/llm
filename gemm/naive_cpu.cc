#include "gemm/launch_template.h"

namespace gemm {

template<>
void GemmExec<float, k_naive_cpu_impl>(const MatrixDim& dim, const float* A, const float* B, float* C, cudaStream_t stream) {
  (void) stream;
  GemmCpuImpl<float>(dim, A, B, C);
}

}  // namespace gemm
