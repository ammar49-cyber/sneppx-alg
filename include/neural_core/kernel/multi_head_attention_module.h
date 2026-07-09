#ifndef SNEPPX_ATTENTION_H
#define SNEPPX_ATTENTION_H

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
} SNEPPXAttentionConfig;

typedef struct {
    SNEPPXAttentionConfig config;
    SNEPPXTensor* w_q; SNEPPXTensor* b_q;
    SNEPPXTensor* w_k; SNEPPXTensor* b_k;
    SNEPPXTensor* w_v; SNEPPXTensor* b_v;
    SNEPPXTensor* w_o; SNEPPXTensor* b_o;
} SNEPPXAttentionWeights;

typedef struct {
    SNEPPXTensor* k_cache;
    SNEPPXTensor* v_cache;
    size_t seq_len;
} SNEPPXKVCache;

SNEPPXAttentionConfig SNEPPX_attn_config_default(void);
SNEPPXAttentionWeights* SNEPPX_attn_weights_create(SNEPPXAttentionConfig cfg, unsigned int seed);
void SNEPPX_attn_weights_destroy(SNEPPXAttentionWeights* w);
size_t SNEPPX_attn_num_params(const SNEPPXAttentionWeights* w);
int SNEPPX_attn_get_params(const SNEPPXAttentionWeights* w, SNEPPXTensor** out, size_t max);

SNEPPXTensor* SNEPPX_attn_forward(const SNEPPXAttentionWeights* w, const SNEPPXTensor* x,
                               SNEPPXTensor* cos_t, SNEPPXTensor* sin_t);
SNEPPXTensor* SNEPPX_attn_forward_cached(const SNEPPXAttentionWeights* w, const SNEPPXTensor* x,
                                       SNEPPXKVCache* cache, SNEPPXTensor* cos_t, SNEPPXTensor* sin_t);

SNEPPXKVCache* SNEPPX_kv_cache_create(size_t max_batch, size_t max_seq, size_t num_heads, size_t head_dim);
void SNEPPX_kv_cache_destroy(SNEPPXKVCache* cache);
void SNEPPX_kv_cache_clear(SNEPPXKVCache* cache);

SNEPPXTensor* SNEPPX_rope_precompute(size_t seq_len, size_t head_dim, float base);
void SNEPPX_rope_apply(SNEPPXTensor* q, SNEPPXTensor* k, const SNEPPXTensor* cos, const SNEPPXTensor* sin,
                     size_t offset);
SNEPPXTensor* SNEPPX_tensor_rope(const SNEPPXTensor* x, const SNEPPXTensor* cos_table);

SNEPPXTensor* SNEPPX_batched_matmul(const SNEPPXTensor* a, const SNEPPXTensor* b,
                                 int transpose_b, int transpose_a);

int SNEPPX_attn_build_train_graph(SNEPPXAttentionWeights* w, SNEPPXTape* tape,
                                 SNEPPXVariable* input_var,
                                 SNEPPXVariable** weight_vars, size_t num_weights,
                                 SNEPPXVariable** output_var,
                                 SNEPPXTensor* cos, SNEPPXTensor* sin);

#endif
