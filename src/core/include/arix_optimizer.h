#ifndef ARIX_OPTIMIZER_H
#define ARIX_OPTIMIZER_H

#include "arix_tensor.h"
#include <stddef.h>

typedef struct {
    float learning_rate;
    float momentum;
    float weight_decay;
    float grad_clip;
} ArixOptimizerConfig;

typedef struct {
    float learning_rate;
    float momentum;
    float weight_decay;
    float grad_clip;
    ArixTensor** momentum_buffers;
    size_t num_params;
    size_t step_count;
} ArixOptimizer;

ArixOptimizerConfig arix_optimizer_config_default(void);
ArixOptimizer* arix_optimizer_create(const ArixOptimizerConfig* config);
void arix_optimizer_destroy(ArixOptimizer* opt);
void arix_optimizer_step(ArixOptimizer* opt, ArixTensor** params, ArixTensor** grads, size_t num_params);

#endif /* ARIX_OPTIMIZER_H */
