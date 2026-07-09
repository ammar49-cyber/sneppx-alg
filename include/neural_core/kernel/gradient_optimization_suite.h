#ifndef SNEPPX_OPTIMIZER_H
#define SNEPPX_OPTIMIZER_H

#include "multidimensional_tensor_engine.h"
#include <stddef.h>

typedef enum {
    SNEPPX_OPTIMIZER_SGD = 0,
    SNEPPX_OPTIMIZER_ADAM,
    SNEPPX_OPTIMIZER_ADAMW,
    SNEPPX_OPTIMIZER_ADAMAX,
    SNEPPX_OPTIMIZER_RMSPROP,
    SNEPPX_OPTIMIZER_ADAGRAD,
    SNEPPX_OPTIMIZER_ADADELTA,
} SNEPPXOptimizerType;

typedef struct {
    float learning_rate;
    float momentum;
    float weight_decay;
    float grad_clip;
    SNEPPXOptimizerType type;
    float beta1, beta2;
    float epsilon;
    float dampening;
    int nesterov;
    float rho;
} SNEPPXOptimizerConfig;

typedef struct {
    float learning_rate;
    float momentum;
    float weight_decay;
    float grad_clip;
    SNEPPXTensor** momentum_buffers;
    size_t num_params;
    size_t step_count;
    SNEPPXOptimizerType type;
    SNEPPXTensor** state_buf2;
    SNEPPXTensor** state_buf3;
    float beta1, beta2;
    float epsilon;
    float dampening;
    int nesterov;
    float rho;
} SNEPPXOptimizer;

SNEPPXOptimizerConfig SNEPPX_optimizer_config_default(void);
SNEPPXOptimizer* SNEPPX_optimizer_create(const SNEPPXOptimizerConfig* config);
void SNEPPX_optimizer_destroy(SNEPPXOptimizer* opt);
void SNEPPX_optimizer_step(SNEPPXOptimizer* opt, SNEPPXTensor** params, SNEPPXTensor** grads, size_t num_params);

SNEPPXOptimizer* SNEPPX_sgd_create(float lr, float momentum, float weight_decay);
SNEPPXOptimizer* SNEPPX_adam_create(float lr, float beta1, float beta2, float eps, float weight_decay);
SNEPPXOptimizer* SNEPPX_adamw_create(float lr, float beta1, float beta2, float eps, float weight_decay);
SNEPPXOptimizer* SNEPPX_rmsprop_create(float lr, float alpha, float eps, float momentum, float weight_decay);
SNEPPXOptimizer* SNEPPX_adagrad_create(float lr, float eps, float weight_decay);

typedef enum {
    SNEPPX_LR_STEP,
    SNEPPX_LR_EXPONENTIAL,
    SNEPPX_LR_COSINE,
    SNEPPX_LR_REDUCE_ON_PLATEAU,
} SNEPPXLRSchedulerType;

typedef struct SNEPPXLRScheduler {
    SNEPPXLRSchedulerType type;
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
} SNEPPXLRScheduler;

SNEPPXLRScheduler* SNEPPX_lr_scheduler_step_lr(float* lr_ptr, float gamma, size_t step_size);
SNEPPXLRScheduler* SNEPPX_lr_scheduler_exponential(float* lr_ptr, float gamma);
SNEPPXLRScheduler* SNEPPX_lr_scheduler_cosine(float* lr_ptr, float min_lr, float max_lr, size_t total_steps);
SNEPPXLRScheduler* SNEPPX_lr_scheduler_reduce_on_plateau(float* lr_ptr, float factor, size_t patience, int mode_min);
void SNEPPX_lr_scheduler_destroy(SNEPPXLRScheduler* sched);
void SNEPPX_lr_scheduler_step(SNEPPXLRScheduler* sched, float current_loss);

#endif
