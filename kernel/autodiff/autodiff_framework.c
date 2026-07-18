#include "automatic_differentiation_framework.h"
#include <stdlib.h>

typedef struct SNEPPXAutogradEngine {
    int dummy;
} SNEPPXAutogradEngine;

SNEPPXAutogradEngine* SNEPPX_autograd_create(void) {
    return (SNEPPXAutogradEngine*)calloc(1, sizeof(SNEPPXAutogradEngine));
}

void SNEPPX_autograd_destroy(SNEPPXAutogradEngine* engine) {
    free(engine);
}

int SNEPPX_autograd_record(SNEPPXAutogradEngine* engine, SNEPPXTensor* tensor, const char* name) {
    (void)engine; (void)tensor; (void)name;
    return 0;
}

int SNEPPX_autograd_backward(SNEPPXAutogradEngine* engine, SNEPPXTensor* loss) {
    (void)engine; (void)loss;
    return 0;
}

int SNEPPX_autograd_zero_grad(SNEPPXAutogradEngine* engine) {
    (void)engine;
    return 0;
}

int SNEPPX_autograd_set_grad(SNEPPXAutogradEngine* engine, SNEPPXTensor* tensor, const SNEPPXTensor* grad) {
    (void)engine; (void)tensor; (void)grad;
    return 0;
}

SNEPPXTensor* SNEPPX_autograd_get_grad(const SNEPPXAutogradEngine* engine, const SNEPPXTensor* tensor) {
    (void)engine; (void)tensor;
    return NULL;
}

size_t SNEPPX_autograd_num_ops(const SNEPPXAutogradEngine* engine) {
    (void)engine;
    return 0;
}
