#include "arix_autodiff.h"
#include "arix_memory.h"
#include <string.h>
#include <math.h>

static void backward_nop(void* ctx, ArixTensor* grad_output) {
    (void)ctx; (void)grad_output;
}

ArixVariable* arix_add(ArixTape* tape, ArixVariable* a, ArixVariable* b) {
    if (!a || !b || !a->data || !b->data) return NULL;
    size_t sz = a->data->size < b->data->size ? a->data->size : b->data->size;
    ArixTensor* result = arix_tensor_create(a->data->shape, a->data->ndim, ARIX_FLOAT32);
    if (!result) return NULL;
    float* rd = (float*)result->data;
    float* ad = (float*)a->data;
    float* bd = (float*)b->data;
    if (a->data->size == b->data->size) {
        for (size_t i = 0; i < sz; i++) rd[i] = ad[i] + bd[i];
    } else {
        size_t last = a->data->shape[a->data->ndim - 1];
        for (size_t i = 0; i < a->data->size; i++)
            rd[i] = ad[i] + bd[i % last];
    }
    ArixVariable* var = arix_variable_create(result, a->requires_grad || b->requires_grad);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    var->backward_fn = backward_nop;
    if (tape) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_mul(ArixTape* tape, ArixVariable* a, ArixVariable* b) {
    if (!a || !b || !a->data || !b->data) return NULL;
    size_t sz = a->data->size < b->data->size ? a->data->size : b->data->size;
    ArixTensor* result = arix_tensor_create(a->data->shape, a->data->ndim, ARIX_FLOAT32);
    if (!result) return NULL;
    float* rd = (float*)result->data;
    float* ad = (float*)a->data;
    float* bd = (float*)b->data;
    for (size_t i = 0; i < sz; i++) rd[i] = ad[i] * bd[i];
    ArixVariable* var = arix_variable_create(result, a->requires_grad || b->requires_grad);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    var->backward_fn = backward_nop;
    if (tape) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_matmul(ArixTape* tape, ArixVariable* a, ArixVariable* b) {
    if (!a || !b || !a->data || !b->data) return NULL;
    size_t m = a->data->shape[0], k = a->data->shape[1], n = b->data->shape[1];
    size_t shape_c[] = {m, n};
    ArixTensor* result = arix_tensor_zeros(shape_c, 2, ARIX_FLOAT32);
    if (!result) return NULL;
    float* ad = (float*)a->data;
    float* bd = (float*)b->data;
    float* rd = (float*)result->data;
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++)
            for (size_t l = 0; l < k; l++)
                rd[i * n + j] += ad[i * k + l] * bd[l * n + j];
    ArixVariable* var = arix_variable_create(result, a->requires_grad || b->requires_grad);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    var->backward_fn = backward_nop;
    if (tape) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_mse_loss(ArixTape* tape, ArixVariable* pred, ArixVariable* target) {
    if (!pred || !target || !pred->data || !target->data) return NULL;
    size_t sz = pred->data->size < target->data->size ? pred->data->size : target->data->size;
    float* pd = (float*)pred->data;
    float* td = (float*)target->data;
    float sum = 0.0f;
    for (size_t i = 0; i < sz; i++) { float d = pd[i] - td[i]; sum += d * d; }
    float loss_val = sum / (float)sz;
    size_t shape[] = {1};
    ArixTensor* loss_t = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    if (!loss_t) return NULL;
    ((float*)loss_t->data)[0] = loss_val;
    ArixVariable* var = arix_variable_create(loss_t, 0);
    if (!var) { arix_tensor_destroy(loss_t); return NULL; }
    var->backward_fn = backward_nop;
    if (tape) arix_tape_record(tape, var);
    return var;
}
