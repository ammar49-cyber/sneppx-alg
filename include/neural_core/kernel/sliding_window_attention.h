#ifndef SNEPPX_SLIDING_WINDOW_ATTENTION_H
#define SNEPPX_SLIDING_WINDOW_ATTENTION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int SNEPPX_swa_forward(const float* q, const float* k, const float* v, float* output, int batch, int heads, int seq, int dim, int window_size, float scale);
int SNEPPX_swa_backward(const float* q, const float* k, const float* v, const float* grad_out, float* grad_q, float* grad_k, float* grad_v, int batch, int heads, int seq, int dim, int window_size, float scale);

#ifdef __cplusplus
}
#endif

#endif
