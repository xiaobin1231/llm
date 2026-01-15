#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>

namespace {
inline float rand_uniform(float low, float high) {
  float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  return low + r * (high - low);
}

void init_random(float* data, size_t n, float low, float high) {
  for (size_t i = 0; i < n; i++) {
    data[i] = rand_uniform(low, high);
  }
}

void print_attention_scores(const float* attn,
                            uint32_t batch_size,
                            uint32_t n_head,
                            uint32_t seq_len) {
  for (uint32_t bs = 0; bs < batch_size; bs++) {
    for (uint32_t h = 0; h < n_head; h++) {
      printf("Attention [batch=%u, head=%u]:\n", bs, h);
      for (uint32_t l = 0; l < seq_len; l++) {
        printf("  l=%u: [", l);
        const float* row =
            attn
            + bs * n_head * seq_len * seq_len
            + h * seq_len * seq_len
            + l * seq_len;
        for (uint32_t l2 = 0; l2 < seq_len; l2++) {
          printf("%7.5f", row[l2]);
          if (l2 + 1 < seq_len) printf(", ");
        }
        printf("]\n");
      }
      printf("\n");
    }
  }
}

void print_output_tensor(const float* output,
                         uint32_t batch_size,
                         uint32_t seq_len,
                         uint32_t d_model) {
  for (uint32_t bs = 0; bs < batch_size; bs++) {
    printf("Output batch %u:\n", bs);
    for (uint32_t l = 0; l < seq_len; l++) {
      printf("  token %u: [", l);
      const float* row =
          output
          + bs * seq_len * d_model
          + l * d_model;
      for (uint32_t i = 0; i < d_model; i++) {
        printf("%8.5f", row[i]);
        if (i + 1 < d_model) printf(", ");
      }
      printf("]\n");
    }
    printf("\n");
  }
}

}  // namespace

namespace llm {

/***
 * @brief Execute self-attention forward on cpu.
 * @param token_embeddings: (batch_size, seq_len, d_model) (2, 3, 16)
 * @param qkv_weights: (3, d_model, d_model) (3, 16, 16)
 * @param multi_head_qkv_matrixs: (3, batch_size, seq_len ,d_model) (3, 2, 3, 16)
 * @param attention_scores: (batch_size, n_head, seq_len, seq_len) (2, 4, 3, 3)
 * @param output_tensor: (batch_size, seq_len, d_model) (2, 3, 16)
 * @param n_head: 4
***/
void self_attention_on_cpu(
    const float* token_embeddings,
    const float* qkv_weights,
    float* multi_head_qkv_matrixs, float* attention_scores,
    float* output_tensor, uint32_t batch_size, uint32_t seq_len, uint32_t d_model, uint32_t n_head) {
  const float* query_weights = qkv_weights + 0u * d_model * d_model;  // 16 x 16
  const float* key_weights = qkv_weights + 1u * d_model * d_model;  // 16 x 16
  const float* value_weights = qkv_weights + 2u * d_model * d_model;  // 16 x 16

  float* query_matrix = multi_head_qkv_matrixs + 0u * batch_size * seq_len * d_model;  // 2 x 3 x 16
  float* key_matrix = multi_head_qkv_matrixs + 1u * batch_size * seq_len * d_model;  // 2 x 3 x 16
  float* value_matrix = multi_head_qkv_matrixs + 2u * batch_size * seq_len * d_model;  // 2 x 3 x 16

  const uint32_t head_size = d_model / n_head;  // 4
  const float scale = 1.0f / sqrtf(head_size);

  for (uint32_t bs = 0u; bs < batch_size; bs++) {
    for (uint32_t l = 0u; l < seq_len; l++) {
      float* multi_head_query = query_matrix + bs * seq_len * d_model + l * d_model;  // 1 x 16
      float* multi_head_key = key_matrix + bs * seq_len * d_model + l * d_model;  // 1 x 16
      float* multi_head_value = value_matrix + bs * seq_len * d_model + l * d_model;  // 1 x 16

      const float* token_embd = token_embeddings + bs * seq_len * d_model + l * d_model;  // 1 x 16
      for (uint32_t i = 0u; i < d_model; i++) {
        for (uint32_t j = 0u; j < d_model; j++) {
          multi_head_query[i] += token_embd[j] * query_weights[j * d_model + i];  // 1 x 16
          multi_head_key[i] += token_embd[j] * key_weights[j * d_model + i];  // 1 x 16
          multi_head_value[i] += token_embd[j] * value_weights[j * d_model + i];  // 1 x 16
        }
      }
    }
  }

  for (uint32_t bs = 0u; bs < batch_size; bs++) {
    for (uint32_t l = 0u; l < seq_len; l++) {
      for (uint32_t h = 0u; h < n_head; h++) {
        float* query = query_matrix + bs * seq_len * d_model + l * d_model + h * head_size;  // 1 x 4
        float* attn_s = attention_scores + bs * n_head * seq_len * seq_len + h * seq_len * seq_len;  // 3 x 3

        float max_value = -FLT_MAX;
        for (uint32_t l2 = 0u; l2 <= l; l2++) {
          float* key = key_matrix + bs * seq_len * d_model + l2 * d_model + h * head_size;  // 1 x 4
          for (uint32_t hs = 0u; hs < head_size; hs++) {
            attn_s[l * seq_len + l2] += query[hs] * key[hs];
          }
          attn_s[l * seq_len + l2] *= scale;
          if (attn_s[l * seq_len + l2] > max_value) {
            max_value = attn_s[l * seq_len + l2];
          }
        }

        float exp_sum_value = 0.0f;
        for (uint32_t l2 = 0; l2 <= l; l2++) {
            attn_s[l * seq_len + l2] = expf(attn_s[l * seq_len + l2] - max_value);
            exp_sum_value += attn_s[l * seq_len + l2];
        }

        float exp_sum_value_inv = (exp_sum_value == 0.0f) ? 0.0f : (1.0f / exp_sum_value);
        for (uint32_t l2 = 0u; l2 <= l; l2++) {
          attn_s[l * seq_len + l2] *= exp_sum_value_inv;
        }
      }
    }
  }

  for (uint32_t bs = 0u; bs < batch_size; bs++) {
    for (uint32_t l = 0u; l < seq_len; l++) {
      for (uint32_t h = 0u; h < n_head; h++) {
        float* attn_s = attention_scores + bs * n_head * seq_len * seq_len + h * seq_len * seq_len + l * seq_len;  // 1 x 3
        float* out = output_tensor + bs * seq_len * d_model + l * d_model + h * head_size;  // 1 x 4
        for (uint32_t l2 = 0u; l2 <= l; l2++) {
          float* value = value_matrix + bs * seq_len * d_model + l2 * d_model + h * head_size;  // 1 x 4
          for (uint32_t hs = 0u; hs < head_size; hs++) {
            out[hs] += attn_s[l2] * value[hs];
          }
        }
      }
    }
  }
}

}  // namespace llm

int main(int argc, char* argv[]) {
  constexpr uint32_t batch_size = 2u;
  constexpr uint32_t seq_len = 3u;
  constexpr uint32_t d_model = 16u;
  constexpr uint32_t n_head = 4u;

  float* token_embeddings = static_cast<float*>(malloc(batch_size * seq_len * d_model * sizeof(float)));
  float* qkv_weights = static_cast<float*>(malloc(3u * d_model * d_model * sizeof(float)));
  float* multi_head_qkv_matrixs = static_cast<float*>(malloc(3u * batch_size * seq_len * d_model * sizeof(float)));
  float* attention_scores = static_cast<float*>(malloc(batch_size * n_head * seq_len * seq_len * sizeof(float)));
  float* output_tensor = static_cast<float*>(malloc(batch_size * seq_len * d_model * sizeof(float)));

  init_random(token_embeddings,
              batch_size * seq_len * d_model,
              -0.5f, 0.5f);

  init_random(qkv_weights,
              3u * d_model * d_model,
              -1.0f, 1.0f);

  memset(multi_head_qkv_matrixs, 0,
         3u * batch_size * seq_len * d_model * sizeof(float));
  memset(attention_scores, 0,
         batch_size * n_head * seq_len * seq_len * sizeof(float));
  memset(output_tensor, 0,
         batch_size * seq_len * d_model * sizeof(float));

  llm::self_attention_on_cpu(token_embeddings, qkv_weights, multi_head_qkv_matrixs,
                             attention_scores, output_tensor, batch_size, seq_len, d_model, n_head);

  print_attention_scores(attention_scores, batch_size, n_head, seq_len);
  print_output_tensor(output_tensor, batch_size, seq_len, d_model);

  free(token_embeddings);
  free(qkv_weights);
  free(multi_head_qkv_matrixs);
  free(attention_scores);
  free(output_tensor);

  return 0;
}
