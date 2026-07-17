#ifndef SNEPPX_OPTIMIZER_RADAM_H
#define SNEPPX_OPTIMIZER_RADAM_H

#include <stddef.h>

typedef struct {
    float* m;
    float* v;
    float beta1, beta2;
    float eps;
    float lr;
    int t;
    size_t size;
} SNEPPXRAdamState;

void SNEPPX_radam_init(SNEPPXRAdamState* state, size_t size, float lr, float beta1, float beta2, float eps);
int  SNEPPX_radam_step(SNEPPXRAdamState* state, float* params, const float* grads);
void SNEPPX_radam_destroy(SNEPPXRAdamState* state);

#endif
