#ifndef SNEPPX_SPARSE_ATTENTION_H
#define SNEPPX_SPARSE_ATTENTION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SNEPPX_SPARSE_TOP_K,
    SNEPPX_SPARSE_STRIDED,
    SNEPPX_SPARSE_RANDOM,
    SNEPPX_SPARSE_BLOCK_LOCAL,
} SNEPPXSparseAttnPattern;

typedef struct {
    SNEPPXSparseAttnPattern pattern;
    int block_size;
    int top_k;
    int stride;
    int window_size;
    unsigned int seed;
} SNEPPXSparseAttnConfig;

int SNEPPX_sparse_attn_forward(const float* q, const float* k, const float* v, float* output, const SNEPPXSparseAttnConfig* cfg, int batch, int heads, int seq, int dim, float scale);
int SNEPPX_sparse_attn_build_mask(int* mask, const SNEPPXSparseAttnConfig* cfg, int seq);

#ifdef __cplusplus
}
#endif

#endif
