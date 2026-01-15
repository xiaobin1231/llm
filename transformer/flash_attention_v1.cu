#include <cuda_runtime.h>
#include <cmath>

namespace llm {

/***
  @brief Flash attention v1 forward implementation. Sadly, there are some limitations as follows. \
    1. The parameter Br must equals Bc.
    2. 越界问题
    3. Causual Mask
***/
template<typename Scalar_t, int seq_len, int d_model, int Br, int Bc, int Tr, int Tc>
__gloabl__ void flash_attention_v1_fwd(Scalar_t* Q, Scalar_t* K, Scalar_t* V, Scalar_t* M, Scalar_t* L) {
  int tx = threadIdx.x;
  int batch_idx = blockIdx.x;
  int head_idx = blockIdx.y;

  int qkv_offset = batch_idx * blockDim.y * seq_len * d_model + head_idx * seq_len * d_model;
  Scalar_t* q_block = Q[qkv_offset];
  Scalar_t* k_block = K[qkv_offset];
  Scalar_t* v_block = V[qkv_offset];

  int ml_offset = batch_idx * blockDim.y * seq_len + head_idx * seq_len;
  Scalar_t* m_block = M[ml_offset];
  Scalar_t* l_block = L[ml_offset];

  extern __shared__ Scalar_t shm[];
  const size_t query_tile_size = Br * d_model;
  const size_t kv_tile_size = Bc * d_model;
  Scalar_t* q_shm = shm;
  Scalar_t* k_shm = shm + query_tile_size;
  Scalar_t* v_shm = shm + query_tile_size + kv_tile_size;
  Scalar_t* s_shm = shm + query_tile_size + 2 * kv_tile_size;

  const float scale = 1.0f / sqrtf(d_model);

  for (int j = 0; j < Tc; j++) {
    // Load k v tile in shm.
    for (int d = 0; d < d_model; d++) {
      k_shm[tx * d_model + d] = k_block[j * kv_tile_size + tx * d_model + d];
      v_shm[tx * d_model + d] = v_block[j * kv_tile_size + tx * d_model + d];
    }
    __syncthreads();

    for (int i = 0; i < Tr; i++) {
      // Load q tile in shm.
      for (int d = 0; d < d_model; d++) {
        q_shm[tx * d_model + d] = q_block[i * query_tile_size + tx * d_model + d];
      }

      Scalar_t prev_m_value = m_block[i * Br + tx];
      Scalar_t prev_l_value = l_block[i * Br + tx];

      // Q @ K.T
      Scalar_t m_value = -FLT_MAX;
      for (int bc_idx = 0; bc_idx < Bc; bc_idx++) {
        if (i * Br + tx < j * Bc + bc_idx) {
          s_shm[tx * Bc + bc_idx] = -FLT_MAX;
        } else {
          Scalar_t p = 0.f;
          for (int d = 0; d < d_model; d++) {
            p += q_shm[tx * d_model + d] * k_shm[bc_idx * d_model + d];
          }
          p *= scale;
          if (m_value < p) {
            m_value = p;
          }
          s_shm[tx * Bc + bc_idx] = p;
        }
      }

      Scalar_t l_value = 0.f;
      for (int bc_idx = 0; bc_idx < Bc; bc_idx++) {
        if (i * Br + tx < j * Bc + bc_idx) {
          s_shm[tx * Bc + bc_idx] = 0.f;
        } else {
          s_shm[tx * Bc + bc_idx] = __expf(s_shm[tx * Bc + bc_idx] - m_value);
        }
        l_value += s_shm[tx * Bc + bc_idx];
      }

      Scalar_t new_m_value = max(prev_m_value, m_value);
      Scalar_t new_l_value = __expf(m_value - prev_m_value) * prev_l_value + __expf() * l_value;


    }
  }
}

}  // namespace llm

int main(int argc, char* argv[]) {
  llm::flash_attention_v1_fwd<>();
}
