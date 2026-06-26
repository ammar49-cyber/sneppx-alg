#ifndef ARIX_AUTODIFF_H
#define ARIX_AUTODIFF_H

#include "arix_tensor.h"
#include <stddef.h>

typedef void (*BackwardFn)(void* ctx, ArixTensor* grad_output);

typedef struct ArixVariable {
    ArixTensor* data;
    ArixTensor* grad;
    int requires_grad;
    BackwardFn backward_fn;
    void* backward_ctx;
    struct ArixVariable** parents;
    size_t num_parents;
} ArixVariable;

typedef struct {
    ArixVariable** vars;
    size_t num_vars;
    size_t capacity;
} ArixTape;

ArixVariable* arix_variable_create(ArixTensor* data, int requires_grad);
void          arix_variable_destroy(ArixVariable* var);
void          arix_variable_set_requires_grad(ArixVariable* var, int requires_grad);

ArixTape* arix_tape_create(void);
void      arix_tape_destroy(ArixTape* tape);
void      arix_tape_record(ArixTape* tape, ArixVariable* var);
void      arix_tape_backward(ArixTape* tape, ArixVariable* loss);

void  arix_no_grad_enter(void);
void  arix_no_grad_exit(void);
int   arix_no_grad_is_active(void);

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
ArixVariable* arix_dropout(ArixTape* tape, ArixVariable* a, float rate, unsigned int seed);
ArixVariable* arix_layer_norm(ArixTape* tape, ArixVariable* a, ArixVariable* gamma, ArixVariable* beta, float eps);
ArixVariable* arix_conv2d(ArixTape* tape, ArixVariable* input, ArixVariable* kernel, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w);
ArixVariable* arix_concat(ArixTape* tape, ArixVariable** vars, size_t num_vars, size_t dim);

#endif
