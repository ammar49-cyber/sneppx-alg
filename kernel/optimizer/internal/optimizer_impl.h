#ifndef ARIX_OPTIMIZER_INTERNAL_H
#define ARIX_OPTIMIZER_INTERNAL_H
/*
 * Optimizer Internal — v0.5
 *
 * PURPOSE: Per-parameter update kernels, gradient clipping, weight decay,
 * and learning-rate schedule internals.  The external API defines the
 * optimizer struct; this header holds the step implementations and state
 * tracking for SGD, Adam, AdamW, and LAMB.
 *
 * DEPENDENCIES: gradient_optimization_suite.h, multidimensional_tensor_engine.h
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void**     param_data;
    void**     grad_data;
    void**     momentum_buf;
    void**     velocity_buf;
    size_t     num_params;
    size_t     param_size;
    float      lr;
    float      beta1;
    float      beta2;
    float      eps;
    float      weight_decay;
    uint64_t   step;
} ArixOptimizerState;

typedef enum {
    ARIX_SCHEDULE_CONSTANT,
    ARIX_SCHEDULE_COSINE,
    ARIX_SCHEDULE_LINEAR,
    ARIX_SCHEDULE_STEP_DECAY,
    ARIX_SCHEDULE_WARMUP_COSINE,
} ArixLRSchedule;

typedef struct {
    ArixLRSchedule type;
    float          base_lr;
    float          min_lr;
    size_t         warmup_steps;
    size_t         total_steps;
    size_t         step_count;
    float          current_lr;
} ArixLRScheduler;

int  arix_optimizer_state_init(ArixOptimizerState* state, size_t num_params, size_t param_size);
void arix_optimizer_state_destroy(ArixOptimizerState* state);

int  arix_optimizer_sgd_step(ArixOptimizerState* state);
int  arix_optimizer_adam_step(ArixOptimizerState* state);
int  arix_optimizer_adamw_step(ArixOptimizerState* state);
int  arix_optimizer_lamb_step(ArixOptimizerState* state);

int  arix_gradient_clip_norm(void** grads, size_t num_params, size_t param_size, float max_norm);
int  arix_gradient_clip_value(void** grads, size_t num_params, size_t param_size, float min_val, float max_val);

int  arix_lr_scheduler_init(ArixLRScheduler* sched, ArixLRSchedule type);
float arix_lr_scheduler_step(ArixLRScheduler* sched);
void  arix_lr_scheduler_reset(ArixLRScheduler* sched);

#ifdef __cplusplus
}
#endif

#endif /* ARIX_OPTIMIZER_INTERNAL_H */
