#ifndef ARIX_ATTENTION_H
#define ARIX_ATTENTION_H

#include "multidimensional_tensor_engine.h"
#include "automatic_differentiation_framework.h"

typedef struct {
    size_t num_heads;
    size_t head_dim;
    size_t d_model;
    float dropout;
    int use_causal_mask;
    int use_rope;
    float rope_base;
} ArixAttentionConfig;

typedef struct {
    ArixAttentionConfig config;
    ArixTensor* w_q; ArixTensor* b_q;
    ArixTensor* w_k; ArixTensor* b_k;
    ArixTensor* w_v; ArixTensor* b_v;
    ArixTensor* w_o; ArixTensor* b_o;
} ArixAttentionWeights;

typedef struct {
    ArixTensor* k_cache;
    ArixTensor* v_cache;
    size_t seq_len;
} ArixKVCache;

ArixAttentionConfig arix_attn_config_default(void);
ArixAttentionWeights* arix_attn_weights_create(ArixAttentionConfig cfg, unsigned int seed);
void arix_attn_weights_destroy(ArixAttentionWeights* w);
size_t arix_attn_num_params(const ArixAttentionWeights* w);
int arix_attn_get_params(const ArixAttentionWeights* w, ArixTensor** out, size_t max);

ArixTensor* arix_attn_forward(const ArixAttentionWeights* w, const ArixTensor* x,
                               ArixTensor* cos_t, ArixTensor* sin_t);
ArixTensor* arix_attn_forward_cached(const ArixAttentionWeights* w, const ArixTensor* x,
                                       ArixKVCache* cache, ArixTensor* cos_t, ArixTensor* sin_t);

ArixKVCache* arix_kv_cache_create(size_t max_batch, size_t max_seq, size_t num_heads, size_t head_dim);
void arix_kv_cache_destroy(ArixKVCache* cache);
void arix_kv_cache_clear(ArixKVCache* cache);

ArixTensor* arix_rope_precompute(size_t seq_len, size_t head_dim, float base);
void arix_rope_apply(ArixTensor* q, ArixTensor* k, const ArixTensor* cos, const ArixTensor* sin,
                     size_t offset);
ArixTensor* arix_tensor_rope(const ArixTensor* x, const ArixTensor* cos_table);

ArixTensor* arix_batched_matmul(const ArixTensor* a, const ArixTensor* b,
                                 int transpose_b, int transpose_a);

int arix_attn_build_train_graph(ArixAttentionWeights* w, ArixTape* tape,
                                 ArixVariable* input_var,
                                 ArixVariable** weight_vars, size_t num_weights,
                                 ArixVariable** output_var,
                                 ArixTensor* cos, ArixTensor* sin);

#endif
