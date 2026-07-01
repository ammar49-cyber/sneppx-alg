#include "inference_engine.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

ArixGenerationConfig arix_generation_config_default(void) {
    ArixGenerationConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_p = 0.9f;
    cfg.top_k = 40;
    cfg.max_new_tokens = 256;
    cfg.eos_token_id = -1;
    return cfg;
}

int arix_argmax(const float* logits, size_t n) {
    if (!logits || n == 0) return -1;
    size_t best = 0;
    for (size_t i = 1; i < n; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return (int)best;
}

int arix_sample_from_logits(const float* logits, size_t vocab_size, ArixGenerationConfig* cfg) {
    if (!logits || vocab_size == 0 || !cfg) return -1;

    size_t n = vocab_size;
    float* probs = (float*)arix_malloc(n * sizeof(float), 64);
    if (!probs) return arix_argmax(logits, n);

    int* indices = (int*)arix_malloc(n * sizeof(int), 64);
    if (!indices) { arix_free(probs, n * sizeof(float)); return arix_argmax(logits, n); }

    for (size_t i = 0; i < n; i++) {
        probs[i] = logits[i] / (cfg->temperature > 0.0f ? cfg->temperature : 1.0f);
    }

    float max_logit = probs[0];
    for (size_t i = 1; i < n; i++) if (probs[i] > max_logit) max_logit = probs[i];
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        probs[i] = expf(probs[i] - max_logit);
        sum += probs[i];
    }
    float inv_sum = 1.0f / (sum + 1e-10f);
    for (size_t i = 0; i < n; i++) probs[i] *= inv_sum;

    if (cfg->top_k > 0 && cfg->top_k < (int)n) {
        for (size_t i = 0; i < n; i++) indices[i] = (int)i;
        for (int i = 0; i < (int)n - 1; i++) {
            for (int j = i + 1; j < (int)n; j++) {
                if (probs[indices[i]] < probs[indices[j]]) {
                    int t = indices[i]; indices[i] = indices[j]; indices[j] = t;
                }
            }
        }
        float kth_prob = probs[indices[cfg->top_k - 1]];
        for (size_t i = 0; i < n; i++) {
            if (probs[i] < kth_prob) probs[i] = 0.0f;
        }
        float new_sum = 0.0f;
        for (size_t i = 0; i < n; i++) new_sum += probs[i];
        inv_sum = 1.0f / (new_sum + 1e-10f);
        for (size_t i = 0; i < n; i++) probs[i] *= inv_sum;
    }

    if (cfg->top_p > 0.0f && cfg->top_p < 1.0f) {
        for (size_t i = 0; i < n; i++) indices[i] = (int)i;
        for (size_t i = 0; i < n - 1; i++) {
            for (size_t j = i + 1; j < n; j++) {
                if (probs[indices[i]] < probs[indices[j]]) {
                    int t = indices[i]; indices[i] = indices[j]; indices[j] = t;
                }
            }
        }
        float cum = 0.0f;
        for (size_t i = 0; i < n; i++) {
            if (cum >= cfg->top_p) probs[indices[i]] = 0.0f;
            cum += probs[indices[i]];
        }
        float new_sum = 0.0f;
        for (size_t i = 0; i < n; i++) new_sum += probs[i];
        inv_sum = 1.0f / (new_sum + 1e-10f);
        for (size_t i = 0; i < n; i++) probs[i] *= inv_sum;
    }

    float r = (float)rand() / ((float)RAND_MAX + 1.0f);
    float cumulative = 0.0f;
    int result = 0;
    for (size_t i = 0; i < n; i++) {
        cumulative += probs[i];
        if (r < cumulative) { result = (int)i; break; }
    }

    arix_free(probs, n * sizeof(float));
    arix_free(indices, n * sizeof(int));
    return result;
}

int arix_generate_tokens(ArixTensor* embed_weight, ArixTensor* unembed_weight,
                         ArixAttentionWeights* attn,
                         const int* input_ids, size_t input_len,
                         int* output_ids, size_t max_output_len,
                         ArixGenerationConfig* cfg) {
    if (!embed_weight || !unembed_weight || !attn || !input_ids || !output_ids || !cfg) return -1;
    if (input_len == 0 || max_output_len == 0) return -1;

    size_t vocab_size = unembed_weight->shape[1];
    size_t d_model = embed_weight->shape[1];

    for (size_t i = 0; i < input_len && i < max_output_len; i++) {
        output_ids[i] = input_ids[i];
    }

    size_t generated = input_len;
    size_t seq_len = input_len;
    ArixKVCache* cache = arix_kv_cache_create(1, max_output_len, attn->config.num_heads, attn->config.head_dim);
    if (!cache) return -1;

    size_t pos_shape[] = {1};
    ArixTensor* pos_t = arix_tensor_empty(pos_shape, 1, ARIX_INT32);
    float* pos_d = (float*)pos_t->data;

    while (generated < max_output_len) {
        size_t current_len = (generated == input_len) ? input_len : 1;
        size_t* idx_ptr = (size_t*)(&output_ids[generated - current_len]);

        size_t emb_shape[] = {1, current_len, d_model};
        ArixTensor* emb = arix_tensor_empty(emb_shape, 3, ARIX_FLOAT32);
        if (!emb) break;

        ArixTensor idx_t;
        size_t idx_sh[] = {current_len};
        ArixTensor* idx_tensor = arix_tensor_create(idx_sh, 1, ARIX_INT32);
        if (!idx_tensor) { arix_tensor_destroy(emb); break; }
        int* idx_d = (int*)idx_tensor->data;
        for (size_t j = 0; j < current_len; j++) {
            idx_d[j] = (int)output_ids[generated - current_len + j];
        }

        ArixTensor* emb_raw = arix_tensor_embedding(embed_weight, idx_tensor);
        arix_tensor_destroy(idx_tensor);

        if (!emb_raw) { arix_tensor_destroy(emb); break; }
        float* emb_src = (float*)emb_raw->data;
        memcpy(emb->data, emb_src, current_len * d_model * sizeof(float));
        arix_tensor_destroy(emb_raw);

        ArixTensor* cos_t = arix_rope_precompute(current_len, attn->config.head_dim, attn->config.rope_base);
        ArixTensor* attn_out = NULL;
        if (generated == input_len) {
            attn_out = arix_attn_forward(attn, emb, cos_t, cos_t);
        } else {
            attn_out = arix_attn_forward_cached(attn, emb, cache, cos_t, cos_t);
        }
        arix_tensor_destroy(cos_t);
        arix_tensor_destroy(emb);

        if (!attn_out) break;

        size_t resh_shape[] = {d_model};
        ArixTensor* flat = arix_tensor_reshape(attn_out, resh_shape, 1);
        arix_tensor_destroy(attn_out);
        if (!flat) break;

        ArixTensor* logits = arix_tensor_matmul(flat, unembed_weight);
        arix_tensor_destroy(flat);

        if (!logits) break;
        float* ld = (float*)logits->data;
        size_t last_idx = logits->size / vocab_size - 1;
        float* last_logits = ld + last_idx * vocab_size;

        int next_id = arix_sample_from_logits(last_logits, vocab_size, cfg);
        arix_tensor_destroy(logits);

        if (next_id < 0) break;
        output_ids[generated] = next_id;
        generated++;

        if (cfg->eos_token_id >= 0 && next_id == cfg->eos_token_id) break;
    }

    arix_kv_cache_destroy(cache);
    return (int)generated;
}

char* arix_generate(ArixTensor* embed_weight, ArixTensor* unembed_weight,
                    ArixAttentionWeights* attn, ArixTokenizer* tok,
                    const char* prompt, ArixGenerationConfig* cfg) {
    if (!embed_weight || !unembed_weight || !attn || !tok || !prompt || !cfg) return NULL;

    int* ids = (int*)arix_malloc(4096 * sizeof(int), 64);
    if (!ids) return NULL;
    int input_len = arix_tokenizer_encode(tok, prompt, ids, 4096);
    if (input_len <= 0) { arix_free(ids, 4096 * sizeof(int)); return NULL; }

    int* output_ids = (int*)arix_malloc((size_t)(input_len + cfg->max_new_tokens + 1) * sizeof(int), 64);
    if (!output_ids) { arix_free(ids, 4096 * sizeof(int)); return NULL; }

    int total = arix_generate_tokens(embed_weight, unembed_weight, attn,
                                      ids, (size_t)input_len,
                                      output_ids, (size_t)(input_len + cfg->max_new_tokens),
                                      cfg);
    arix_free(ids, 4096 * sizeof(int));

    if (total <= 0) { arix_free(output_ids, (size_t)(input_len + cfg->max_new_tokens + 1) * sizeof(int)); return NULL; }

    char* result = arix_tokenizer_decode(tok, output_ids, (size_t)total);
    arix_free(output_ids, (size_t)(input_len + cfg->max_new_tokens + 1) * sizeof(int));
    return result;
}
