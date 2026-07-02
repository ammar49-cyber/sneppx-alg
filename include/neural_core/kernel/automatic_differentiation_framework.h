#ifndef ARIX_AUTODIFF_H
#define ARIX_AUTODIFF_H

#include "multidimensional_tensor_engine.h"
#include <stddef.h>

typedef void (*BackwardFn)(void* ctx, ArixTensor* grad_output);

typedef void* (*RecomputeCtxFn)(struct ArixVariable* var, size_t* params, size_t param_count);

typedef struct ArixVariable {
    ArixTensor* data;
    ArixTensor* grad;
    int requires_grad;
    int checkpointed;
    BackwardFn backward_fn;
    void* backward_ctx;
    void (*free_ctx)(void*);
    RecomputeCtxFn recompute_ctx;
    int ref_count;
    struct ArixVariable** parents;
    size_t num_parents;
    size_t params[8];
    size_t param_count;
} ArixVariable;

typedef struct {
    ArixVariable** vars;
    size_t num_vars;
    size_t capacity;
    int checkpointing;
} ArixTape;

ArixVariable* arix_variable_create(ArixTensor* data, int requires_grad);
void          arix_variable_destroy(ArixVariable* var);
void          arix_variable_set_requires_grad(ArixVariable* var, int requires_grad);
ArixVariable* arix_variable_detach(ArixVariable* var);
ArixVariable* arix_variable_copy(ArixVariable* var);
void          arix_variable_zero_grad(ArixVariable* var);
float         arix_variable_item(ArixVariable* var);
size_t        arix_variable_numel(ArixVariable* var);

ArixTape* arix_tape_create(void);
void      arix_tape_destroy(ArixTape* tape);
void      arix_tape_record(ArixTape* tape, ArixVariable* var);
void      arix_tape_backward(ArixTape* tape, ArixVariable* loss);
void      arix_tape_zero_grad(ArixTape* tape);
float     arix_tape_global_norm(ArixTape* tape);
void      arix_tape_clip_grad_norm(ArixTape* tape, float max_norm);

void  arix_no_grad_enter(void);
void  arix_no_grad_exit(void);
int   arix_no_grad_is_active(void);

void  arix_tape_checkpoint_begin(ArixTape* tape);
void  arix_tape_checkpoint_end(ArixTape* tape);

ArixVariable* arix_add(ArixTape* tape, ArixVariable* a, ArixVariable* b);
ArixVariable* arix_sub(ArixTape* tape, ArixVariable* a, ArixVariable* b);
ArixVariable* arix_mul(ArixTape* tape, ArixVariable* a, ArixVariable* b);
ArixVariable* arix_div(ArixTape* tape, ArixVariable* a, ArixVariable* b);
ArixVariable* arix_pow(ArixTape* tape, ArixVariable* a, ArixVariable* b);
ArixVariable* arix_neg(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_matmul(ArixTape* tape, ArixVariable* a, ArixVariable* b);
ArixVariable* arix_mse_loss(ArixTape* tape, ArixVariable* pred, ArixVariable* target);
ArixVariable* arix_relu(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_gelu(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_silu(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_sigmoid(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_tanh(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_softmax(ArixTape* tape, ArixVariable* a, size_t dim);
ArixVariable* arix_exp(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_log(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_sum(ArixTape* tape, ArixVariable* a, size_t dim);
ArixVariable* arix_mean(ArixTape* tape, ArixVariable* a, size_t dim);
ArixVariable* arix_transpose(ArixTape* tape, ArixVariable* a, size_t dim1, size_t dim2);
ArixVariable* arix_reshape(ArixTape* tape, ArixVariable* a, const size_t* shape, size_t ndim);
ArixVariable* arix_rope(ArixTape* tape, ArixVariable* a, ArixTensor* cos_table);
ArixVariable* arix_dropout(ArixTape* tape, ArixVariable* a, float rate, unsigned int seed);
ArixVariable* arix_layer_norm(ArixTape* tape, ArixVariable* a, ArixVariable* gamma, ArixVariable* beta, float eps);
ArixVariable* arix_conv2d(ArixTape* tape, ArixVariable* input, ArixVariable* kernel, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w);
ArixVariable* arix_concat(ArixTape* tape, ArixVariable** vars, size_t num_vars, size_t dim);

ArixVariable* arix_minimum(ArixTape* tape, ArixVariable* a, ArixVariable* b);
ArixVariable* arix_maximum(ArixTape* tape, ArixVariable* a, ArixVariable* b);
ArixVariable* arix_sqrt(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_abs(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_sin(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_cos(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_tan(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_asin(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_acos(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_atan(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_sinh(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_cosh(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_var(ArixTape* tape, ArixVariable* a, size_t dim);
ArixVariable* arix_std(ArixTape* tape, ArixVariable* a, size_t dim);
ArixVariable* arix_cross_entropy(ArixTape* tape, ArixVariable* pred, ArixVariable* target);
ArixVariable* arix_nll_loss(ArixTape* tape, ArixVariable* pred, ArixVariable* target);
ArixVariable* arix_bce_loss(ArixTape* tape, ArixVariable* pred, ArixVariable* target);
ArixVariable* arix_embedding(ArixTape* tape, ArixVariable* weight, ArixVariable* indices);

ArixVariable* arix_log_softmax(ArixTape* tape, ArixVariable* a, size_t dim);
ArixVariable* arix_sign(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_floor(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_ceil(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_round(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_trunc(ArixTape* tape, ArixVariable* a);
ArixVariable* arix_batch_norm(ArixTape* tape, ArixVariable* a, ArixVariable* gamma, ArixVariable* beta, ArixVariable* running_mean, ArixVariable* running_var, float eps);

#endif
