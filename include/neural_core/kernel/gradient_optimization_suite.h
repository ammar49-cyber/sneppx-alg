#ifndef ARIX_OPTIMIZER_H
#define ARIX_OPTIMIZER_H

#include "multidimensional_tensor_engine.h"
#include <stddef.h>

typedef enum {
    ARIX_OPTIMIZER_SGD = 0,
    ARIX_OPTIMIZER_ADAM,
    ARIX_OPTIMIZER_ADAMW,
    ARIX_OPTIMIZER_ADAMAX,
    ARIX_OPTIMIZER_RMSPROP,
    ARIX_OPTIMIZER_ADAGRAD,
    ARIX_OPTIMIZER_ADADELTA,
} ArixOptimizerType;

typedef struct {
    float learning_rate;
    float momentum;
    float weight_decay;
    float grad_clip;
    ArixOptimizerType type;
    float beta1, beta2;
    float epsilon;
    float dampening;
    int nesterov;
    float rho;
} ArixOptimizerConfig;

typedef struct {
    float learning_rate;
    float momentum;
    float weight_decay;
    float grad_clip;
    ArixTensor** momentum_buffers;
    size_t num_params;
    size_t step_count;
    ArixOptimizerType type;
    ArixTensor** state_buf2;
    ArixTensor** state_buf3;
    float beta1, beta2;
    float epsilon;
    float dampening;
    int nesterov;
    float rho;
} ArixOptimizer;

ArixOptimizerConfig arix_optimizer_config_default(void);
ArixOptimizer* arix_optimizer_create(const ArixOptimizerConfig* config);
void arix_optimizer_destroy(ArixOptimizer* opt);
void arix_optimizer_step(ArixOptimizer* opt, ArixTensor** params, ArixTensor** grads, size_t num_params);

ArixOptimizer* arix_sgd_create(float lr, float momentum, float weight_decay);
ArixOptimizer* arix_adam_create(float lr, float beta1, float beta2, float eps, float weight_decay);
ArixOptimizer* arix_adamw_create(float lr, float beta1, float beta2, float eps, float weight_decay);
ArixOptimizer* arix_rmsprop_create(float lr, float alpha, float eps, float momentum, float weight_decay);
ArixOptimizer* arix_adagrad_create(float lr, float eps, float weight_decay);

typedef enum {
    ARIX_LR_STEP,
    ARIX_LR_EXPONENTIAL,
    ARIX_LR_COSINE,
    ARIX_LR_REDUCE_ON_PLATEAU,
} ArixLRSchedulerType;

typedef struct ArixLRScheduler {
    ArixLRSchedulerType type;
    float* lr_ptr;
    float gamma;
    float min_lr, max_lr;
    size_t step_size;
    float factor;
    size_t patience;
    size_t total_steps;
    size_t last_epoch;
    float best_loss;
    int wait_count;
    int mode_min;
} ArixLRScheduler;

ArixLRScheduler* arix_lr_scheduler_step_lr(float* lr_ptr, float gamma, size_t step_size);
ArixLRScheduler* arix_lr_scheduler_exponential(float* lr_ptr, float gamma);
ArixLRScheduler* arix_lr_scheduler_cosine(float* lr_ptr, float min_lr, float max_lr, size_t total_steps);
ArixLRScheduler* arix_lr_scheduler_reduce_on_plateau(float* lr_ptr, float factor, size_t patience, int mode_min);
void arix_lr_scheduler_destroy(ArixLRScheduler* sched);
void arix_lr_scheduler_step(ArixLRScheduler* sched, float current_loss);

#endif
