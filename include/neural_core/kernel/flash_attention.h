#ifndef SNEPPX_FLASH_ATTENTION_H
#define SNEPPX_FLASH_ATTENTION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int SNEPPX_flash_attn_forward(const float* q, const float* k, const float* v, float* output, int batch, int heads, int seq, int dim, float scale);
int SNEPPX_flash_attn_backward(const float* q, const float* k, const float* v, const float* grad_out, float* grad_q, float* grad_k, float* grad_v, int batch, int heads, int seq, int dim, float scale);

#ifdef __cplusplus
}
#endif

#endif
