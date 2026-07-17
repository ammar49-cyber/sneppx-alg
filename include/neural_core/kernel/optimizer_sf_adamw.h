#ifndef SNEPPX_OPTIMIZER_SF_ADAMW_H
#define SNEPPX_OPTIMIZER_SF_ADAMW_H

#include <stddef.h>

typedef struct {
    float* z;
    float* m;
    float* v;
    float beta1, beta2;
    float eps;
    float lr;
    float weight_decay;
    int t;
    size_t size;
} SNEPPXSFRAdamWState;

void SNEPPX_sf_adamw_init(SNEPPXSFRAdamWState* state, size_t size, float lr, float beta1, float beta2, float eps, float weight_decay);
int  SNEPPX_sf_adamw_step(SNEPPXSFRAdamWState* state, float* params, const float* grads);
void SNEPPX_sf_adamw_destroy(SNEPPXSFRAdamWState* state);

#endif
