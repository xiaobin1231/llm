#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

namespace {
inline float RandUniform(float low, float high) {
  float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  return low + r * (high - low);
}

void InitRandom(float* data, size_t n, float low, float high) {
  for (size_t i = 0; i < n; i++) {
    data[i] = RandUniform(low, high);
  }
}

void PrintOutputTensor(const float* output, uint32_t batch_size, uint32_t n_head, uint32_t seq_len, uint32_t d_k) {
  for (uint32_t bs = 0u; bs < batch_size; bs++) {
    printf("Output batch %u:\n", bs);
    for (uint32_t n = 0u; n < n_head; n++) {
      printf(" head %u:\n", n);
      for (uint32_t l = 0u; l < seq_len; l++) {
        printf("  token %u: [", l);
        const float* row = output + bs * n_head * seq_len * d_k + n * seq_len * d_k + l * d_k;
        for (uint32_t i = 0u; i < d_k; i++) {
          printf("%8.5f", row[i]);
          if (i + 1 < d_k) {
            printf(", ");
          }
        }
        printf("]\n");
      }
      printf("]\n");
    }
    printf("\n");
  }
}

}  // namespace

namespace llm {

/***
 * @brief Flash attention v1 forward implementation. Sadly, there are some limitations as follows. \
 *  1. The parameter Br must equals Bc.
 *  2. Br value must equals 32 or a multiple of 32.
***/
template<int seq_len, int d_k, int Br, int Bc, int Tr, int Tc>
__global__ void flash_attention_v1_fwd(float* Q, float* K, float* V, float* M, float* L, float* O, float scale) {
  int tx = threadIdx.x;
  int batch_idx = blockIdx.x;
  int head_idx = blockIdx.y;

  int qkv_offset = batch_idx * gridDim.y * seq_len * d_k + head_idx * seq_len * d_k;
  float* q_block = Q + qkv_offset;
  float* k_block = K + qkv_offset;
  float* v_block = V + qkv_offset;
  float* o_block = O + qkv_offset;

  int ml_offset = batch_idx * gridDim.y * seq_len + head_idx * seq_len;
  float* m_block = M + ml_offset;
  float* l_block = L + ml_offset;

  extern __shared__ float shm[];
  const size_t query_tile_size = Br * d_k;
  const size_t kv_tile_size = Bc * d_k;
  float* q_shm = shm;
  float* k_shm = shm + query_tile_size;
  float* v_shm = shm + query_tile_size + kv_tile_size;
  float* s_shm = shm + query_tile_size + 2 * kv_tile_size;

  for (int j = 0; j < Tc; j++) {
    // Load k v tile in shm.
    for (int d = 0; d < d_k; d++) {
      k_shm[tx * d_k + d] = k_block[j * kv_tile_size + tx * d_k + d];
      v_shm[tx * d_k + d] = v_block[j * kv_tile_size + tx * d_k + d];
    }
    __syncthreads();

    for (int i = 0; i < Tr; i++) {
      if (i * Br + tx >= seq_len) {
        break;
      }

      // Load q tile in shm.
      for (int d = 0; d < d_k; d++) {
        q_shm[tx * d_k + d] = q_block[i * query_tile_size + tx * d_k + d];
      }

      float prev_m_value = m_block[i * Br + tx];
      float prev_l_value = l_block[i * Br + tx];

      // Q @ K.T
      float curr_m_value = -FLT_MAX;
      for (int bc_idx = 0; bc_idx < Bc; bc_idx++) {
        if (j * Bc + bc_idx >= seq_len) {
          break;
        }

        float p = 0.0f;
        for (int d = 0; d < d_k; d++) {
          p += q_shm[tx * d_k + d] * k_shm[bc_idx * d_k + d];
        }

        p *= scale;
        if (curr_m_value < p) {
          curr_m_value = p;
        }
        s_shm[tx * Bc + bc_idx] = p;

        if (i * Br + tx < j * Bc + bc_idx) {
          s_shm[tx * Bc + bc_idx] = -FLT_MAX;
        }
      }

      float curr_l_value = 0.0f;
      for (int bc_idx = 0; bc_idx < Bc; bc_idx++) {
        if (j * Bc + bc_idx >= seq_len) {
          break;
        }

        s_shm[tx * Bc + bc_idx] = __expf(s_shm[tx * Bc + bc_idx] - curr_m_value);
        if (i * Br + tx < j * Bc + bc_idx) {
          s_shm[tx * Bc + bc_idx] = 0.f;
        }
        curr_l_value += s_shm[tx * Bc + bc_idx];
      }

      // Online softmax update.
      float new_m_value = max(prev_m_value, curr_m_value);
      float new_l_value = __expf(prev_m_value - new_m_value) * prev_l_value + __expf(curr_m_value - new_m_value) * curr_l_value;

      for (int d = 0; d < d_k; d++) {
        float pv = 0.0f;
        for (int bc_idx = 0; bc_idx < Bc; bc_idx++) {
          if (j * Bc + bc_idx >= seq_len) {
            break;
          }

          pv += s_shm[tx * Bc + bc_idx] * v_shm[bc_idx * d_k + d];
        }
        o_block[query_tile_size * i + tx * d_k + d] = (1 / new_l_value) * \
          ((__expf(prev_m_value - new_m_value) * prev_l_value * o_block[query_tile_size * i + tx * d_k + d]) + \
            __expf(curr_m_value - new_m_value) * pv);
      }

      m_block[i * Br + tx] = new_m_value;
      l_block[i * Br + tx] = new_l_value;
    }
    __syncthreads();
  }
}

}  // namespace llm

int main(int argc, char* argv[]) {
  constexpr int seq_len = 128;
  constexpr int d_model = 64;
  constexpr int Br = 32;
  constexpr int Bc = 32;
  constexpr int Tr = seq_len / Br;
  constexpr int Tc = seq_len / Bc;

  const int batch_size = 2;
  const int num_heads = 4;
  const int d_k = d_model / num_heads;

  const float scale = 1.0f / sqrtf(static_cast<float>(d_k));

  // (batch_size, seq_len, d_model)
  float* token_embeddings = static_cast<float*>(malloc(batch_size * seq_len * d_model * sizeof(float)));
  InitRandom(token_embeddings, batch_size * seq_len * d_model, -0.5f, 0.5f);

  // (3, d_model, d_model)
  float* qkv_weights = static_cast<float*>(malloc(3u * d_model * d_model * sizeof(float)));
  InitRandom(qkv_weights, 3u * d_model * d_model, -1.0f, 1.0f);
  const float* query_weights = qkv_weights + 0u * d_model * d_model;
  const float* key_weights = qkv_weights + 1u * d_model * d_model;
  const float* value_weights = qkv_weights + 2u * d_model * d_model;

  float *Q, *K, *V, *O;
  const size_t qkv_size = batch_size * num_heads * seq_len * d_k * sizeof(float);
  cudaMallocManaged(&Q, qkv_size);
  memset(Q, 0, qkv_size);
  cudaMallocManaged(&K, qkv_size);
  memset(K, 0, qkv_size);
  cudaMallocManaged(&V, qkv_size);
  memset(V, 0, qkv_size);
  cudaMallocManaged(&O, qkv_size);
  memset(O, 0, qkv_size);

  for (uint32_t bs = 0u; bs < batch_size; bs++) {
    for (uint32_t n = 0u; n < num_heads; n++) {
      for (uint32_t l = 0u; l < seq_len; l++) {
        float* query = Q + bs * num_heads * seq_len * d_k + n * seq_len * d_k + l * d_k;
        float* key = K + bs * num_heads * seq_len * d_k + n * seq_len * d_k + l * d_k;
        float* value = V + bs * num_heads * seq_len * d_k + n * seq_len * d_k + l * d_k;

        const float* token_embd = token_embeddings + bs * seq_len * d_model + l * d_model;
        for (uint32_t i = 0u; i < d_k; i++) {
          float q_sum = 0.0f;
          float k_sum = 0.0f;
          float v_sum = 0.0f;
          for (uint32_t j = 0u; j < d_model; j++) {
            // x @ Q.T
            q_sum += token_embd[j] * query_weights[j * d_model + n * d_k + i];
            // x @ K.T
            k_sum += token_embd[j] * key_weights[j * d_model + n * d_k + i];
            // x @ V.T
            v_sum += token_embd[j] * value_weights[j * d_model + n * d_k + i];
          }
          query[i] = q_sum;
          key[i] = k_sum;
          value[i] = v_sum;
        }
      }
    }
  }

  float *M, *L;
  const size_t ml_size = batch_size * num_heads * seq_len * sizeof(float);
  cudaMallocManaged(&M, ml_size);
  for (size_t i = 0; i < batch_size * num_heads * seq_len; ++i) {
    M[i] = -FLT_MAX;
  }

  cudaMallocManaged(&L, ml_size);
  memset(L, 0, ml_size);

  dim3 grid(batch_size, num_heads);
  dim3 block(Br);

  const size_t query_tile_size = Br * d_k;
  const size_t kv_tile_size = Bc * d_k;
  const size_t s_tile_size = Br * Bc;
  const size_t shared_mem_size = (query_tile_size + 2 * kv_tile_size + s_tile_size) * sizeof(float);

  printf("Launching kernel with:\n");
  printf("  grid: (%u, %u)\n", grid.x, grid.y);
  printf("  block: %u\n", block.x);
  printf("  shared memory: %ld bytes\n", shared_mem_size);

  llm::flash_attention_v1_fwd<seq_len, d_k, Br, Bc, Tr, Tc>
      <<<grid, block, shared_mem_size>>>(Q, K, V, M, L, O, scale);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s.\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();

  PrintOutputTensor(O, batch_size, num_heads, seq_len, d_k);

  cudaFree(Q);
  cudaFree(K);
  cudaFree(V);
  cudaFree(M);
  cudaFree(L);
  cudaFree(O);

  return 0;
}
