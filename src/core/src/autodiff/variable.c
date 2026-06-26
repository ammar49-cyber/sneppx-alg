#include "arix_autodiff.h"
#include "arix_memory.h"
#include <string.h>

ArixVariable* arix_variable_create(ArixTensor* data, int requires_grad) {
    ArixVariable* var = (ArixVariable*)arix_malloc(sizeof(ArixVariable), 64);
    if (!var) return NULL;
    memset(var, 0, sizeof(ArixVariable));
    var->data = data;
    var->requires_grad = requires_grad;
    var->grad = NULL;
    var->backward_fn = NULL;
    var->backward_ctx = NULL;
    var->parents = NULL;
    var->num_parents = 0;
    return var;
}

void arix_variable_destroy(ArixVariable* var) {
    if (!var) return;
    if (var->data) arix_tensor_destroy(var->data);
    if (var->grad) arix_tensor_destroy(var->grad);
    if (var->parents) arix_free(var->parents, var->num_parents * sizeof(ArixVariable*));
    arix_free(var, sizeof(ArixVariable));
}

void arix_variable_set_requires_grad(ArixVariable* var, int requires_grad) {
    if (!var) return;
    var->requires_grad = requires_grad;
}
