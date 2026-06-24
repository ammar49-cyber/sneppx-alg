#include "arix_optimizer.h"
#include "arix_memory.h"
#include <string.h>
#include <math.h>

ArixOptimizerConfig arix_optimizer_config_default(void) {
    ArixOptimizerConfig cfg;
    cfg.learning_rate = 0.01f;
    cfg.momentum = 0.9f;
    cfg.weight_decay = 1e-4f;
    cfg.grad_clip = 1.0f;
    return cfg;
}

ArixOptimizer* arix_optimizer_create(const ArixOptimizerConfig* config) {
    if (!config) return NULL;
    ArixOptimizer* opt = (ArixOptimizer*)arix_malloc(sizeof(ArixOptimizer), 64);
    if (!opt) return NULL;
    memset(opt, 0, sizeof(ArixOptimizer));
    opt->learning_rate = config->learning_rate;
    opt->momentum = config->momentum;
    opt->weight_decay = config->weight_decay;
    opt->grad_clip = config->grad_clip;
    opt->num_params = 0;
    opt->momentum_buffers = NULL;
    opt->step_count = 0;
    return opt;
}

void arix_optimizer_destroy(ArixOptimizer* opt) {
    if (!opt) return;
    if (opt->momentum_buffers) {
        for (size_t i = 0; i < opt->num_params; i++) {
            if (opt->momentum_buffers[i]) arix_tensor_destroy(opt->momentum_buffers[i]);
        }
        arix_free(opt->momentum_buffers, opt->num_params * sizeof(ArixTensor*));
    }
    arix_free(opt, sizeof(ArixOptimizer));
}

void arix_optimizer_step(ArixOptimizer* opt, ArixTensor** params, ArixTensor** grads, size_t num_params) {
    if (!opt || !params || !grads || num_params == 0) return;

    if (opt->num_params != num_params) {
        if (opt->momentum_buffers) {
            for (size_t i = 0; i < opt->num_params; i++) {
                if (opt->momentum_buffers[i]) arix_tensor_destroy(opt->momentum_buffers[i]);
            }
            arix_free(opt->momentum_buffers, opt->num_params * sizeof(ArixTensor*));
        }
        opt->momentum_buffers = (ArixTensor**)arix_malloc(num_params * sizeof(ArixTensor*), 64);
        if (!opt->momentum_buffers) return;
        memset(opt->momentum_buffers, 0, num_params * sizeof(ArixTensor*));
        for (size_t i = 0; i < num_params; i++) {
            if (params[i]) {
                opt->momentum_buffers[i] = arix_tensor_zeros(params[i]->shape, params[i]->ndim, ARIX_FLOAT32);
            }
        }
        opt->num_params = num_params;
    }

    opt->step_count++;

    for (size_t i = 0; i < num_params; i++) {
        if (!params[i] || !grads[i]) continue;

        float* pd = (float*)params[i]->data;
        float* gd = (float*)grads[i]->data;
        float* md = (float*)opt->momentum_buffers[i]->data;
        float lr = opt->learning_rate;
        float mu = opt->momentum;
        float wd = opt->weight_decay;
        float clip = opt->grad_clip;
        size_t sz = params[i]->size;

        for (size_t j = 0; j < sz; j++) {
            float g = gd[j];
            if (g > clip) g = clip;
            if (g < -clip) g = -clip;
            g += wd * pd[j];
            md[j] = mu * md[j] - lr * g;
            pd[j] += md[j];
        }
    }
}
