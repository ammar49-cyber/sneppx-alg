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

ArixVariable* arix_variable_detach(ArixVariable* var) {
    if (!var || !var->data) return NULL;
    ArixVariable* out = arix_variable_create(
        arix_tensor_create(var->data->shape, var->data->ndim, ARIX_FLOAT32), 0);
    if (out && out->data) {
        memcpy(out->data->data, var->data->data, var->data->size * sizeof(float));
    }
    return out;
}

ArixVariable* arix_variable_copy(ArixVariable* var) {
    if (!var || !var->data) return NULL;
    ArixVariable* out = arix_variable_create(
        arix_tensor_create(var->data->shape, var->data->ndim, ARIX_FLOAT32), var->requires_grad);
    if (out && out->data) {
        memcpy(out->data->data, var->data->data, var->data->size * sizeof(float));
        if (var->grad) {
            out->grad = arix_tensor_create(var->grad->shape, var->grad->ndim, ARIX_FLOAT32);
            if (out->grad) memcpy(out->grad->data, var->grad->data, var->grad->size * sizeof(float));
        }
    }
    return out;
}

void arix_variable_zero_grad(ArixVariable* var) {
    if (!var) return;
    if (var->grad) { arix_tensor_destroy(var->grad); var->grad = NULL; }
}

float arix_variable_item(ArixVariable* var) {
    if (!var || !var->data || var->data->size == 0) return 0.0f;
    return ((float*)var->data->data)[0];
}

size_t arix_variable_numel(ArixVariable* var) {
    if (!var || !var->data) return 0;
    return var->data->size;
}
