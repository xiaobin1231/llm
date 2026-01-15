#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdint>

/***
 * @brief Execute self-attention forward on cpu.
 * @param token_embeddings: (batch_size, seq_len, d_model) (2, 3, 16)
 * @param qkv_weights: (3, d_model, d_model) (3, 16, 16)
 * @param out_tensor: (batch_size, seq_len, d_model) (2, 3, 16)
 * @param n_head: 4
***/
void self_attention_on_cpu(const float* token_embeddings, const float* qkv_weights, float* out_tensor,
                           uint32_t batch_size, uint32_t seq_len, uint32_t d_model, uint32_t n_head) {
  const float* query_weights = qkv_weights + 0u * d_model * d_model;  // 16 x 16
  const float* key_weights = qkv_weights + 1u * d_model * d_model;  // 16 x 16
  const float* value_weights = qkv_weights + 2u * d_model * d_model;  // 16 x 16

  // FIXME: Init param and remember to free mem.
  float* query_matrix = static_cast<float*>(malloc(batch_size * seq_len * d_model * sizeof(float)));  // 2 x 3 x 16
  float* key_matrix = static_cast<float*>(malloc(batch_size * seq_len * d_model * sizeof(float)));  // 2 x 3 x 16
  float* value_matrix = static_cast<float*>(malloc(batch_size * seq_len * d_model * sizeof(float)));  // 2 x 3 x 16

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

  // FIXME: Init param and remember to free mem.
  float* attention_scores = static_cast<float*>(malloc(batch_size * n_head * seq_len * seq_len * sizeof(float)));  // 2 x 4 x 3 x 3
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
        for (uint32_t l2 = 0u; l2 < l; l2++) {
          attn_s[l * seq_len + l2] *= exp_sum_value_inv;
        }
      }
    }
  }

  for (uint32_t bs = 0u; bs < batch_size; bs++) {
    for (uint32_t l = 0u; l < seq_len; l++) {
      for (uint32_t h = 0u; h < n_head; h++) {
        float* attn_s = attention_scores + bs * n_head * seq_len * seq_len + h * seq_len * seq_len + l * seq_len;  // 1 x 3
        float* out = out_tensor + bs * seq_len * d_model + l * d_model + h * head_size;  // 1 x 4
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

int main(int argc, char* argv[]) {


  return 0;
}
