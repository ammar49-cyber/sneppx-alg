#ifndef SNEPPX_OPTIMIZER_INTERNAL_H
#define SNEPPX_OPTIMIZER_INTERNAL_H
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
} SNEPPXOptimizerState;

typedef enum {
    SNEPPX_SCHEDULE_CONSTANT,
    SNEPPX_SCHEDULE_COSINE,
    SNEPPX_SCHEDULE_LINEAR,
    SNEPPX_SCHEDULE_STEP_DECAY,
    SNEPPX_SCHEDULE_WARMUP_COSINE,
} SNEPPXLRSchedule;

typedef struct {
    SNEPPXLRSchedule type;
    float          base_lr;
    float          min_lr;
    size_t         warmup_steps;
    size_t         total_steps;
    size_t         step_count;
    float          current_lr;
} SNEPPXLRScheduler;

int  SNEPPX_optimizer_state_init(SNEPPXOptimizerState* state, size_t num_params, size_t param_size);
void SNEPPX_optimizer_state_destroy(SNEPPXOptimizerState* state);

int  SNEPPX_optimizer_sgd_step(SNEPPXOptimizerState* state);
int  SNEPPX_optimizer_adam_step(SNEPPXOptimizerState* state);
int  SNEPPX_optimizer_adamw_step(SNEPPXOptimizerState* state);
int  SNEPPX_optimizer_lamb_step(SNEPPXOptimizerState* state);

int  SNEPPX_gradient_clip_norm(void** grads, size_t num_params, size_t param_size, float max_norm);
int  SNEPPX_gradient_clip_value(void** grads, size_t num_params, size_t param_size, float min_val, float max_val);

int  SNEPPX_lr_scheduler_init(SNEPPXLRScheduler* sched, SNEPPXLRSchedule type);
float SNEPPX_impl_lr_scheduler_step(SNEPPXLRScheduler* sched);
void  SNEPPX_lr_scheduler_reset(SNEPPXLRScheduler* sched);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_OPTIMIZER_INTERNAL_H */
