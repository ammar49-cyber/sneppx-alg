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
void arix_variable_destroy(ArixVariable* var);
ArixTape* arix_tape_create(void);
void arix_tape_destroy(ArixTape* tape);
void arix_tape_record(ArixTape* tape, ArixVariable* var);
void arix_tape_backward(ArixTape* tape, ArixVariable* loss);

ArixVariable* arix_add(ArixTape* tape, ArixVariable* a, ArixVariable* b);
ArixVariable* arix_mul(ArixTape* tape, ArixVariable* a, ArixVariable* b);
ArixVariable* arix_matmul(ArixTape* tape, ArixVariable* a, ArixVariable* b);
ArixVariable* arix_mse_loss(ArixTape* tape, ArixVariable* pred, ArixVariable* target);

#endif /* ARIX_AUTODIFF_H */
