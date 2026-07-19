#ifndef SNEPPX_POSITION_ENCODING_H
#define SNEPPX_POSITION_ENCODING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int SNEPPX_rope_precompute_freqs(float* cos, float* sin, int max_seq, int dim, float base);
int SNEPPX_rope_apply_freqs(const float* x, float* output, const float* cos, const float* sin, int batch, int seq, int heads, int dim);
int SNEPPX_rope_apply_freqs_inplace(float* x, const float* cos, const float* sin, int batch, int seq, int heads, int dim);

float* SNEPPX_rope_precompute_tensor(int max_seq, int dim, float base);

int SNEPPX_alibi_compute_slopes(float* slopes, int num_heads, float slope_base);
int SNEPPX_alibi_apply(float* attn_scores, const float* slopes, int batch, int heads, int seq_q, int seq_k);

int SNEPPX_yarn_precompute(float* cos, float* sin, int max_seq, int dim, float base, float scale, float alpha, float beta);
int SNEPPX_yarn_apply(const float* x, float* output, const float* cos, const float* sin, int batch, int seq, int heads, int dim, float alpha, float beta);

#ifdef __cplusplus
}
#endif

#endif
