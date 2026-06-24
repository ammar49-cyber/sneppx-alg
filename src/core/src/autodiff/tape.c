#include "arix_autodiff.h"
#include "arix_memory.h"
#include <string.h>
#include <stdlib.h>

ArixTape* arix_tape_create(void) {
    ArixTape* tape = (ArixTape*)arix_malloc(sizeof(ArixTape), 64);
    if (!tape) return NULL;
    memset(tape, 0, sizeof(ArixTape));
    tape->capacity = 64;
    tape->num_vars = 0;
    tape->vars = (ArixVariable**)arix_malloc(tape->capacity * sizeof(ArixVariable*), 64);
    if (!tape->vars) { arix_free(tape, sizeof(ArixTape)); return NULL; }
    memset(tape->vars, 0, tape->capacity * sizeof(ArixVariable*));
    return tape;
}

void arix_tape_destroy(ArixTape* tape) {
    if (!tape) return;
    for (size_t i = 0; i < tape->num_vars; i++) {
        if (tape->vars[i]) arix_variable_destroy(tape->vars[i]);
    }
    if (tape->vars) arix_free(tape->vars, tape->capacity * sizeof(ArixVariable*));
    arix_free(tape, sizeof(ArixTape));
}

void arix_tape_record(ArixTape* tape, ArixVariable* var) {
    if (!tape || !var) return;
    if (tape->num_vars >= tape->capacity) {
        tape->capacity *= 2;
        ArixVariable** new_vars = (ArixVariable**)arix_malloc(tape->capacity * sizeof(ArixVariable*), 64);
        if (!new_vars) return;
        memcpy(new_vars, tape->vars, tape->num_vars * sizeof(ArixVariable*));
        arix_free(tape->vars, tape->capacity / 2 * sizeof(ArixVariable*));
        tape->vars = new_vars;
    }
    tape->vars[tape->num_vars++] = var;
}

void arix_tape_backward(ArixTape* tape, ArixVariable* loss) {
    if (!tape || !loss) return;
    if (loss->grad) arix_tensor_destroy(loss->grad);
    loss->grad = arix_tensor_ones(loss->data->shape, loss->data->ndim, ARIX_FLOAT32);

    for (size_t i = tape->num_vars; i > 0; i--) {
        ArixVariable* var = tape->vars[i - 1];
        if (var->requires_grad && var->grad == NULL) {
            size_t shape[] = {1};
            var->grad = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
        }
    }
}
