#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

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

template<int seq_len, int d_k, int Br, int Bc, int THREAD_WORKERS>
__global__ void flash_attention_v2_warp_level_fwd(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* O, float scale) {
  int q_tile_idx = blockIdx.x;
  int batch_idx = blockIdx.y;
  int head_idx = blockIdx.z;

  constexpr int WARP_SIZE = 32;
  constexpr int WARP_NUM = THREAD_WORKERS / WARP_SIZE;
  constexpr int ROWS_PER_WARP = Br / WARP_NUM;
  constexpr int COLS_PER_THREAD = d_k / WARP_SIZE;
  static_assert(THREAD_WORKERS % WARP_SIZE == 0, "Threads must be multiple of 32");
  static_assert(Br % WARP_NUM == 0, "Br must be divisible by WARP_NUM");
  static_assert(d_k % WARP_SIZE == 0, "d_k must be divisible by 32");

  int tx = threadIdx.x;
  int warp_id = tx / WARP_SIZE;
  int lane_id = tx % WARP_SIZE;

  int qkv_offset = batch_idx * gridDim.z * seq_len * d_k + head_idx * seq_len * d_k;
  const float* q_block = Q + qkv_offset;
  const float* k_block = K + qkv_offset;
  const float* v_block = V + qkv_offset;

  extern __shared__ float shm[];
  float* q_shm = shm;
  float* k_shm = shm + Br * d_k;
  float* v_shm = shm + Br * d_k + Bc * d_k;

  // Load q tile in shm.
  int q_tile_size = Br * d_k;
  for (int i = tx; i < q_tile_size; i += THREAD_WORKERS) {
    int row = i / d_k;
    int col = i % d_k;
    q_shm[row * d_k + col] = 0.0f;

    int global_q_row = q_tile_idx * Br + row;
    if (global_q_row < seq_len) {
      q_shm[row * d_k + col] = q_block[global_q_row * d_k + col];
    }
  }
  __syncthreads();

  float m_row[ROWS_PER_WARP];
  float l_row[ROWS_PER_WARP];
  float o_row[ROWS_PER_WARP][COLS_PER_THREAD];
  #pragma unroll
  for (int r = 0; r < ROWS_PER_WARP; r++) {
    m_row[r] = -FLT_MAX;
    l_row[r] = 0.0f;
    #pragma unroll
    for (int c = 0; c < COLS_PER_THREAD; c++) {
      o_row[r][c] = 0.0f;
    }
  }

  int num_kv_tile = (seq_len + Bc - 1) / Bc;
  int kv_tile_size = Bc * d_k;
  for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tile; kv_tile_idx++) {
    // Load kv tile in shm.
    for (int i = tx; i < kv_tile_size; i += THREAD_WORKERS) {
      int row = i / d_k;
      int col = i % d_k;
      int global_kv_row = kv_tile_idx * Bc + row;

      k_shm[row * d_k + col] = 0.0f;
      v_shm[row * d_k + col] = 0.0f;
      if (global_kv_row < seq_len) {
        k_shm[row * d_k + col] = k_block[global_kv_row * d_k + col];
        v_shm[row * d_k + col] = v_block[global_kv_row * d_k + col];
      }
    }
    __syncthreads();

    for (int r = 0; r < ROWS_PER_WARP; r++) {
      int q_row_idx = warp_id * ROWS_PER_WARP + r;
      int global_q_row = q_tile_idx * Br + q_row_idx;
      if (global_q_row >= seq_len) {
        continue;
      }

      float m_tile = -FLT_MAX;
      float s_tile[Bc] = {0.0f};

      // Q @ K.T
      for (int k_row_idx = 0; k_row_idx < Bc; k_row_idx++) {
        int global_k_row = kv_tile_idx * Bc + k_row_idx;
        // Causal mask.
        if (global_q_row < global_k_row || global_k_row >= seq_len) {
          s_tile[k_row_idx] = -FLT_MAX;
          continue;
        }

        float s_partial = 0.0f;
        #pragma unroll
        for (int c = 0; c < COLS_PER_THREAD; c++) {
          int d = c * WARP_SIZE + lane_id;
          s_partial += q_shm[q_row_idx * d_k + d] * k_shm[k_row_idx * d_k + d];
        }

        unsigned mask = __activemask();
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
          s_partial += __shfl_down_sync(mask, s_partial, offset);
        }
        float s = __shfl_sync(mask, s_partial, 0);

        s *= scale;
        s_tile[k_row_idx] = s;
        m_tile = fmaxf(m_tile, s);
      }

      // Softmax(Q @ K.T / scale)
      float l_tile = 0.0f;
      for (int k_row_idx = 0; k_row_idx < Bc; k_row_idx++) {
        float p = __expf(s_tile[k_row_idx] - m_tile);
        s_tile[k_row_idx] = p;
        l_tile += p;
      }

      // Online softmax update.
      float m_new = fmaxf(m_row[r], m_tile);
      float exp_m_prev = __expf(m_row[r] - m_new);
      float exp_m_curr = __expf(m_tile - m_new);
      float l_new = exp_m_curr * l_tile + exp_m_prev * l_row[r];

      // O = S @ V
      #pragma unroll
      for (int c = 0; c < COLS_PER_THREAD; c++) {
        int d = c * WARP_SIZE + lane_id;
        float pv_sum = 0.0f;
        for (int v_row_idx = 0; v_row_idx < Bc; v_row_idx++) {
          pv_sum += s_tile[v_row_idx] * v_shm[v_row_idx * d_k + d];
        }
        o_row[r][c] = exp_m_prev * o_row[r][c] + exp_m_curr * pv_sum;
      }

      m_row[r] = m_new;
      l_row[r] = l_new;
    }

    __syncthreads();
  }

  for (int r = 0; r < ROWS_PER_WARP; r++) {
    int q_row_idx = warp_id * ROWS_PER_WARP + r;
    int global_q_row = q_tile_idx * Br + q_row_idx;
    if (global_q_row < seq_len) {
      float* O_ptr = O + qkv_offset + global_q_row * d_k;
      #pragma unroll
      for (int c = 0; c < COLS_PER_THREAD; c++) {
        int d = c * WARP_SIZE + lane_id;
        O_ptr[d] = o_row[r][c] / l_row[r];
      }
    }
  }
}

}  // namespace llm

int main(int argc, char* argv[]) {
  constexpr int seq_len = 128;
  constexpr int d_model = 128;
  constexpr int Br = 32;
  constexpr int Bc = 16;

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

  int num_q_tile = (seq_len + Br - 1) / Br;
  constexpr int thread_workers = 128;
  dim3 grid(num_q_tile, batch_size, num_heads);
  dim3 block(thread_workers);

  const size_t query_tile_size = Br * d_k;
  const size_t kv_tile_size = Bc * d_k;
  const size_t shared_mem_size = (query_tile_size + 2 * kv_tile_size) * sizeof(float);

  printf("Launching kernel with:\n");
  printf("  grid: (%u, %u, %u)\n", grid.x, grid.y, grid.z);
  printf("  block: %u\n", block.x);
  printf("  shared memory: %ld bytes\n", shared_mem_size);

  llm::flash_attention_v2_warp_level_fwd<seq_len, d_k, Br, Bc, thread_workers>
      <<<grid, block, shared_mem_size>>>(Q, K, V, O, scale);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s.\n", cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();

  PrintOutputTensor(O, batch_size, num_heads, seq_len, d_k);

  cudaFree(Q);
  cudaFree(K);
  cudaFree(V);
  cudaFree(O);

  return 0;
}
