#ifndef SNEPPX_AUTODIFF_H
#define SNEPPX_AUTODIFF_H

#include "multidimensional_tensor_engine.h"
#include <stddef.h>

typedef void (*BackwardFn)(void* ctx, SNEPPXTensor* grad_output);

typedef struct SNEPPXVariable SNEPPXVariable;

typedef void* (*RecomputeCtxFn)(SNEPPXVariable* var, size_t* params, size_t param_count);

typedef struct SNEPPXVariable {
    SNEPPXTensor* data;
    SNEPPXTensor* grad;
    int requires_grad;
    int checkpointed;
    BackwardFn backward_fn;
    void* backward_ctx;
    void (*free_ctx)(void*);
    RecomputeCtxFn recompute_ctx;
    int ref_count;
    struct SNEPPXVariable** parents;
    size_t num_parents;
    size_t params[8];
    size_t param_count;
} SNEPPXVariable;

typedef struct {
    SNEPPXVariable** vars;
    size_t num_vars;
    size_t capacity;
    int checkpointing;
} SNEPPXTape;

SNEPPXVariable* SNEPPX_variable_create(SNEPPXTensor* data, int requires_grad);
void          SNEPPX_variable_destroy(SNEPPXVariable* var);
void          SNEPPX_variable_set_requires_grad(SNEPPXVariable* var, int requires_grad);
SNEPPXVariable* SNEPPX_variable_detach(SNEPPXVariable* var);
SNEPPXVariable* SNEPPX_variable_copy(SNEPPXVariable* var);
void          SNEPPX_variable_zero_grad(SNEPPXVariable* var);
float         SNEPPX_variable_item(SNEPPXVariable* var);
size_t        SNEPPX_variable_numel(SNEPPXVariable* var);

SNEPPXTape* SNEPPX_tape_create(void);
void      SNEPPX_tape_destroy(SNEPPXTape* tape);
void      SNEPPX_tape_record(SNEPPXTape* tape, SNEPPXVariable* var);
void      SNEPPX_tape_backward(SNEPPXTape* tape, SNEPPXVariable* loss);
void      SNEPPX_tape_zero_grad(SNEPPXTape* tape);
float     SNEPPX_tape_global_norm(SNEPPXTape* tape);
void      SNEPPX_tape_clip_grad_norm(SNEPPXTape* tape, float max_norm);

void  SNEPPX_no_grad_enter(void);
void  SNEPPX_no_grad_exit(void);
int   SNEPPX_no_grad_is_active(void);

void  SNEPPX_tape_checkpoint_begin(SNEPPXTape* tape);
void  SNEPPX_tape_checkpoint_end(SNEPPXTape* tape);

SNEPPXVariable* SNEPPX_add(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
SNEPPXVariable* SNEPPX_sub(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
SNEPPXVariable* SNEPPX_mul(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
SNEPPXVariable* SNEPPX_div(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
SNEPPXVariable* SNEPPX_pow(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
SNEPPXVariable* SNEPPX_neg(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_matmul(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
SNEPPXVariable* SNEPPX_mse_loss(SNEPPXTape* tape, SNEPPXVariable* pred, SNEPPXVariable* target);
SNEPPXVariable* SNEPPX_relu(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_gelu(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_silu(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_sigmoid(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_tanh(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_softmax(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim);
SNEPPXVariable* SNEPPX_exp(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_log(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_sum(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim);
SNEPPXVariable* SNEPPX_mean(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim);
SNEPPXVariable* SNEPPX_transpose(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim1, size_t dim2);
SNEPPXVariable* SNEPPX_reshape(SNEPPXTape* tape, SNEPPXVariable* a, const size_t* shape, size_t ndim);
SNEPPXVariable* SNEPPX_rope(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXTensor* cos_table);
SNEPPXVariable* SNEPPX_dropout(SNEPPXTape* tape, SNEPPXVariable* a, float rate, unsigned int seed);
SNEPPXVariable* SNEPPX_layer_norm(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* gamma, SNEPPXVariable* beta, float eps);
SNEPPXVariable* SNEPPX_conv2d(SNEPPXTape* tape, SNEPPXVariable* input, SNEPPXVariable* kernel, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w);
SNEPPXVariable* SNEPPX_concat(SNEPPXTape* tape, SNEPPXVariable** vars, size_t num_vars, size_t dim);

SNEPPXVariable* SNEPPX_minimum(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
SNEPPXVariable* SNEPPX_maximum(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
SNEPPXVariable* SNEPPX_sqrt(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_abs(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_sin(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_cos(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_tan(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_asin(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_acos(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_atan(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_sinh(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_cosh(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_var(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim);
SNEPPXVariable* SNEPPX_std(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim);
SNEPPXVariable* SNEPPX_cross_entropy(SNEPPXTape* tape, SNEPPXVariable* pred, SNEPPXVariable* target);
SNEPPXVariable* SNEPPX_nll_loss(SNEPPXTape* tape, SNEPPXVariable* pred, SNEPPXVariable* target);
SNEPPXVariable* SNEPPX_bce_loss(SNEPPXTape* tape, SNEPPXVariable* pred, SNEPPXVariable* target);
SNEPPXVariable* SNEPPX_embedding(SNEPPXTape* tape, SNEPPXVariable* weight, SNEPPXVariable* indices);

SNEPPXVariable* SNEPPX_log_softmax(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim);
SNEPPXVariable* SNEPPX_sign(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_floor(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_ceil(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_round(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_trunc(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_batch_norm(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* gamma, SNEPPXVariable* beta, SNEPPXVariable* running_mean, SNEPPXVariable* running_var, float eps);

#endif
