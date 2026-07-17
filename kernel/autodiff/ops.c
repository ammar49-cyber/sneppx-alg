/* ===== Zero-gradient helper for comparison ops (no gradient through comparisons) ===== */
#include "automatic_differentiation_framework.h"
#include "multidimensional_tensor_engine.h"
#include "polymorphic_memory_allocator.h"
#include <stddef.h>
#include <string.h>

/* ---------- internal helpers ---------- */
static int requires_grad(SNEPPXVariable* a, SNEPPXVariable* b) {
    return (a && a->requires_grad) || (b && b->requires_grad);
}
static int requires_grad1(SNEPPXVariable* a) {
    return a && a->requires_grad;
}
static void grad_accum(SNEPPXTensor** grad, SNEPPXTensor* new_grad) {
    if (!new_grad) return;
    if (*grad) {
        float* gd = (float*)(*grad)->data;
        float* ng = (float*)new_grad->data;
        size_t n = (*grad)->size < new_grad->size ? (*grad)->size : new_grad->size;
        for (size_t i = 0; i < n; i++) gd[i] += ng[i];
        SNEPPX_tensor_destroy(new_grad);
    } else {
        *grad = new_grad;
    }
}
static void set_parents(SNEPPXVariable* var, SNEPPXVariable** parents, size_t n) {
    if (!var || !parents || n == 0) return;
    var->parents = (SNEPPXVariable**)SNEPPX_malloc(n * sizeof(SNEPPXVariable*), 64);
    if (!var->parents) return;
    memcpy(var->parents, parents, n * sizeof(SNEPPXVariable*));
    var->num_parents = n;
}

typedef SNEPPXTensor* (*sneppx_tensor_bin_fn_t)(const SNEPPXTensor*, const SNEPPXTensor*);

static SNEPPXVariable* op_zero_grad(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b, sneppx_tensor_bin_fn_t tensor_fn) {
    if (!a || !a->data || !b || !b->data) return NULL;
    int rg = 0;
    SNEPPXTensor* result = tensor_fn(a->data, b->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (tape) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== Comparison ops backward (zero gradient - no gradient through comparisons) ===== */
SNEPPXVariable* SNEPPX_eq(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    return op_zero_grad(tape, a, b, SNEPPX_tensor_eq);
}
SNEPPXVariable* SNEPPX_ne(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    return op_zero_grad(tape, a, b, SNEPPX_tensor_ne);
}
SNEPPXVariable* SNEPPX_lt(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    return op_zero_grad(tape, a, b, SNEPPX_tensor_lt);
}
SNEPPXVariable* SNEPPX_le(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    return op_zero_grad(tape, a, b, SNEPPX_tensor_le);
}
SNEPPXVariable* SNEPPX_gt(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    return op_zero_grad(tape, a, b, SNEPPX_tensor_gt);
}
SNEPPXVariable* SNEPPX_ge(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    return op_zero_grad(tape, a, b, SNEPPX_tensor_ge);
}

/* ===== conv1d backward ===== */
typedef struct { SNEPPXVariable *input, *kernel; size_t stride, padding; } Conv1DCtx;
static void free_ctx_Conv1DCtx(void* p) { SNEPPX_free(p, sizeof(Conv1DCtx)); }
static void* recompute_Conv1DCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    Conv1DCtx* ctx = (Conv1DCtx*)SNEPPX_malloc(sizeof(Conv1DCtx), 64);
    if (ctx) {
        ctx->input = var->parents[0]; ctx->kernel = var->parents[1];
        ctx->stride = n > 0 ? params[0] : 1;
        ctx->padding = n > 1 ? params[1] : 0;
        var->free_ctx = free_ctx_Conv1DCtx;
    }
    return ctx;
}
static void backward_conv1d(void* ctx, SNEPPXTensor* grad_output) {
    Conv1DCtx* c = (Conv1DCtx*)ctx;
    float* go = (float*)grad_output->data;
    size_t N = c->input->data->shape[0], C = c->input->data->shape[1], L = c->input->data->shape[2];
    size_t K = c->kernel->data->shape[0];
    size_t KH = c->kernel->data->shape[2];
    size_t OH = (L + 2 * c->padding - KH) / c->stride + 1;

    if (c->input->requires_grad) {
        SNEPPXTensor* gi = SNEPPX_tensor_zeros(c->input->data->shape, c->input->data->ndim, SNEPPX_FLOAT32);
        float* gid = (float*)gi->data;
        float* kd = (float*)c->kernel->data->data;
        for (size_t n = 0; n < N; n++)
            for (size_t k = 0; k < K; k++)
                for (size_t oh = 0; oh < OH; oh++) {
                    float grad_val = go[n * K * OH + k * OH + oh];
                    for (size_t kh = 0; kh < KH; kh++) {
                        size_t ih = oh * c->stride + kh - c->padding;
                        if (ih < L)
                            for (size_t cc = 0; cc < C; cc++)
                                gid[n * C * L + cc * L + ih] += kd[k * C * KH + cc * KH + kh] * grad_val;
                    }
                }
        grad_accum(&c->input->grad, gi);
    }
    if (c->kernel->requires_grad) {
        SNEPPXTensor* gk = SNEPPX_tensor_zeros(c->kernel->data->shape, c->kernel->data->ndim, SNEPPX_FLOAT32);
        float* gkd = (float*)gk->data;
        float* id = (float*)c->input->data->data;
        for (size_t n = 0; n < N; n++)
            for (size_t k = 0; k < K; k++)
                for (size_t oh = 0; oh < OH; oh++) {
                    float grad_val = go[n * K * OH + k * OH + oh];
                    for (size_t kh = 0; kh < KH; kh++) {
                        size_t ih = oh * c->stride + kh - c->padding;
                        if (ih < L)
                            for (size_t cc = 0; cc < C; cc++)
                                gkd[k * C * KH + cc * KH + kh] += id[n * C * L + cc * L + ih] * grad_val;
                    }
                }
        grad_accum(&c->kernel->grad, gk);
    }
}

SNEPPXVariable* SNEPPX_conv1d(SNEPPXTape* tape, SNEPPXVariable* input, SNEPPXVariable* kernel, size_t stride, size_t padding) {
    int rg = requires_grad(input, kernel);
    if (!input || !kernel || !input->data || !kernel->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_conv1d(input->data, kernel->data, stride, padding);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        Conv1DCtx* ctx = (Conv1DCtx*)SNEPPX_malloc(sizeof(Conv1DCtx), 64);
        if (ctx) {
            ctx->input = input; ctx->kernel = kernel; ctx->stride = stride; ctx->padding = padding;
            var->backward_fn = backward_conv1d; var->backward_ctx = ctx;
            var->free_ctx = free_ctx_Conv1DCtx; var->recompute_ctx = recompute_Conv1DCtx;
            var->params[0] = stride; var->params[1] = padding; var->param_count = 2;
        }
        SNEPPXVariable* pars[2]; pars[0] = input; pars[1] = kernel;
        set_parents(var, pars, 2);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== cumsum backward ===== */
typedef struct { SNEPPXVariable* a; size_t dim; } CumsumCtx;
static void free_ctx_CumsumCtx(void* p) { SNEPPX_free(p, sizeof(CumsumCtx)); }
static void* recompute_CumsumCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    CumsumCtx* ctx = (CumsumCtx*)SNEPPX_malloc(sizeof(CumsumCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->dim = n > 0 ? params[0] : 0; var->free_ctx = free_ctx_CumsumCtx; }
    return ctx;
}
static void backward_cumsum(void* ctx, SNEPPXTensor* grad_output) {
    CumsumCtx* c = (CumsumCtx*)ctx;
    if (!c->a->requires_grad) return;
    float* go = (float*)grad_output->data;
    size_t outer = 1, inner = 1;
    for (size_t i = 0; i < c->dim; i++) outer *= c->a->data->shape[i];
    size_t dim_size = c->a->data->shape[c->dim];
    for (size_t i = c->dim + 1; i < c->a->data->ndim; i++) inner *= c->a->data->shape[i];

    SNEPPXTensor* g = SNEPPX_tensor_zeros(c->a->data->shape, c->a->data->ndim, SNEPPX_FLOAT32);
    float* gd = (float*)g->data;

    for (size_t o = 0; o < outer; o++) {
        for (size_t i = 0; i < inner; i++) {
            float cum = 0.0f;
            for (size_t d = dim_size; d > 0; d--) {
                size_t idx = o * dim_size * inner + (d - 1) * inner + i;
                cum += go[o * dim_size * inner + (d - 1) * inner + i];
                gd[idx] = cum;
            }
        }
    }
    grad_accum(&c->a->grad, g);
}
SNEPPXVariable* SNEPPX_cumsum(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_cumsum(a->data, dim);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        CumsumCtx* ctx = (CumsumCtx*)SNEPPX_malloc(sizeof(CumsumCtx), 64);
        if (ctx) { ctx->a = a; ctx->dim = dim; var->backward_fn = backward_cumsum; var->backward_ctx = ctx; var->free_ctx = free_ctx_CumsumCtx; var->recompute_ctx = recompute_CumsumCtx; var->params[0] = dim; var->param_count = 1; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== cumprod backward ===== */
typedef struct { SNEPPXVariable* a; size_t dim; } CumprodCtx;
static void free_ctx_CumprodCtx(void* p) { SNEPPX_free(p, sizeof(CumprodCtx)); }
static void* recompute_CumprodCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    CumprodCtx* ctx = (CumprodCtx*)SNEPPX_malloc(sizeof(CumprodCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->dim = n > 0 ? params[0] : 0; var->free_ctx = free_ctx_CumprodCtx; }
    return ctx;
}
static void backward_cumprod(void* ctx, SNEPPXTensor* grad_output) {
    CumprodCtx* c = (CumprodCtx*)ctx;
    if (!c->a->requires_grad) return;
    size_t outer = 1, inner = 1;
    for (size_t i = 0; i < c->dim; i++) outer *= c->a->data->shape[i];
    size_t dim_size = c->a->data->shape[c->dim];
    for (size_t i = c->dim + 1; i < c->a->data->ndim; i++) inner *= c->a->data->shape[i];

    float* xd = (float*)c->a->data->data;
    float* go = (float*)grad_output->data;
    SNEPPXTensor* g = SNEPPX_tensor_zeros(c->a->data->shape, c->a->data->ndim, SNEPPX_FLOAT32);
    float* gd = (float*)g->data;

    for (size_t o = 0; o < outer; o++) {
        for (size_t i = 0; i < inner; i++) {
            float cumprod = 1.0f;
            for (size_t d = dim_size; d > 0; d--) {
                size_t idx = o * dim_size * inner + (d - 1) * inner + i;
                cumprod *= xd[idx];
                gd[idx] = go[o * dim_size * inner + (d - 1) * inner + i] * (cumprod / (xd[idx] + 1e-12f));
            }
        }
    }
    grad_accum(&c->a->grad, g);
}
SNEPPXVariable* SNEPPX_cumprod(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_cumprod(a->data, dim);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        CumprodCtx* ctx = (CumprodCtx*)SNEPPX_malloc(sizeof(CumprodCtx), 64);
        if (ctx) { ctx->a = a; ctx->dim = dim; var->backward_fn = backward_cumprod; var->backward_ctx = ctx; var->free_ctx = free_ctx_CumprodCtx; var->recompute_ctx = recompute_CumprodCtx; var->params[0] = dim; var->param_count = 1; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== add forward/backward ===== */
typedef struct { SNEPPXVariable* a; SNEPPXVariable* b; } AddCtx;
static void free_ctx_AddCtx(void* p) { SNEPPX_free(p, sizeof(AddCtx)); }
static void* recompute_AddCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    AddCtx* ctx = (AddCtx*)SNEPPX_malloc(sizeof(AddCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->b = var->parents[1]; var->free_ctx = free_ctx_AddCtx; }
    return ctx;
}
static void backward_add(void* ctx, SNEPPXTensor* grad_output) {
    AddCtx* c = (AddCtx*)ctx;
    if (c->a->requires_grad) grad_accum(&c->a->grad, SNEPPX_tensor_copy(grad_output));
    if (c->b->requires_grad) grad_accum(&c->b->grad, SNEPPX_tensor_copy(grad_output));
}
SNEPPXVariable* SNEPPX_add(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    int rg = requires_grad(a, b);
    if (!a || !b || !a->data || !b->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_add(a->data, b->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        AddCtx* ctx = (AddCtx*)SNEPPX_malloc(sizeof(AddCtx), 64);
        if (ctx) { ctx->a = a; ctx->b = b; var->backward_fn = backward_add; var->backward_ctx = ctx; var->free_ctx = free_ctx_AddCtx; var->recompute_ctx = recompute_AddCtx; }
        SNEPPXVariable* pars[2]; pars[0] = a; pars[1] = b;
        set_parents(var, pars, 2);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== sub forward/backward ===== */
typedef struct { SNEPPXVariable* a; SNEPPXVariable* b; } SubCtx;
static void free_ctx_SubCtx(void* p) { SNEPPX_free(p, sizeof(SubCtx)); }
static void* recompute_SubCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    SubCtx* ctx = (SubCtx*)SNEPPX_malloc(sizeof(SubCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->b = var->parents[1]; var->free_ctx = free_ctx_SubCtx; }
    return ctx;
}
static void backward_sub(void* ctx, SNEPPXTensor* grad_output) {
    SubCtx* c = (SubCtx*)ctx;
    if (c->a->requires_grad) grad_accum(&c->a->grad, SNEPPX_tensor_copy(grad_output));
    if (c->b->requires_grad) grad_accum(&c->b->grad, SNEPPX_tensor_neg(grad_output));
}
SNEPPXVariable* SNEPPX_sub(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    int rg = requires_grad(a, b);
    if (!a || !b || !a->data || !b->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_sub(a->data, b->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        SubCtx* ctx = (SubCtx*)SNEPPX_malloc(sizeof(SubCtx), 64);
        if (ctx) { ctx->a = a; ctx->b = b; var->backward_fn = backward_sub; var->backward_ctx = ctx; var->free_ctx = free_ctx_SubCtx; var->recompute_ctx = recompute_SubCtx; }
        SNEPPXVariable* pars[2]; pars[0] = a; pars[1] = b;
        set_parents(var, pars, 2);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== mul forward/backward ===== */
typedef struct { SNEPPXVariable* a; SNEPPXVariable* b; } MulCtx;
static void free_ctx_MulCtx(void* p) { SNEPPX_free(p, sizeof(MulCtx)); }
static void* recompute_MulCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    MulCtx* ctx = (MulCtx*)SNEPPX_malloc(sizeof(MulCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->b = var->parents[1]; var->free_ctx = free_ctx_MulCtx; }
    return ctx;
}
static void backward_mul(void* ctx, SNEPPXTensor* grad_output) {
    MulCtx* c = (MulCtx*)ctx;
    if (c->a->requires_grad) {
        SNEPPXTensor* ga = SNEPPX_tensor_mul(c->b->data, grad_output);
        grad_accum(&c->a->grad, ga);
    }
    if (c->b->requires_grad) {
        SNEPPXTensor* gb = SNEPPX_tensor_mul(c->a->data, grad_output);
        grad_accum(&c->b->grad, gb);
    }
}
SNEPPXVariable* SNEPPX_mul(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    int rg = requires_grad(a, b);
    if (!a || !b || !a->data || !b->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_mul(a->data, b->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        MulCtx* ctx = (MulCtx*)SNEPPX_malloc(sizeof(MulCtx), 64);
        if (ctx) { ctx->a = a; ctx->b = b; var->backward_fn = backward_mul; var->backward_ctx = ctx; var->free_ctx = free_ctx_MulCtx; var->recompute_ctx = recompute_MulCtx; }
        SNEPPXVariable* pars[2]; pars[0] = a; pars[1] = b;
        set_parents(var, pars, 2);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== div forward/backward ===== */
typedef struct { SNEPPXVariable* a; SNEPPXVariable* b; } DivCtx;
static void free_ctx_DivCtx(void* p) { SNEPPX_free(p, sizeof(DivCtx)); }
static void* recompute_DivCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    DivCtx* ctx = (DivCtx*)SNEPPX_malloc(sizeof(DivCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->b = var->parents[1]; var->free_ctx = free_ctx_DivCtx; }
    return ctx;
}
static void backward_div(void* ctx, SNEPPXTensor* grad_output) {
    DivCtx* c = (DivCtx*)ctx;
    if (c->a->requires_grad) {
        SNEPPXTensor* ga = SNEPPX_tensor_div(grad_output, c->b->data);
        grad_accum(&c->a->grad, ga);
    }
    if (c->b->requires_grad) {
        SNEPPXTensor* b2 = SNEPPX_tensor_mul(c->b->data, c->b->data);
        SNEPPXTensor* a_neg = SNEPPX_tensor_neg(c->a->data);
        SNEPPXTensor* num = SNEPPX_tensor_mul(a_neg, grad_output);
        SNEPPXTensor* gb = SNEPPX_tensor_div(num, b2);
        grad_accum(&c->b->grad, gb);
        SNEPPX_tensor_destroy(b2);
        SNEPPX_tensor_destroy(a_neg);
        SNEPPX_tensor_destroy(num);
    }
}
SNEPPXVariable* SNEPPX_div(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    int rg = requires_grad(a, b);
    if (!a || !b || !a->data || !b->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_div(a->data, b->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        DivCtx* ctx = (DivCtx*)SNEPPX_malloc(sizeof(DivCtx), 64);
        if (ctx) { ctx->a = a; ctx->b = b; var->backward_fn = backward_div; var->backward_ctx = ctx; var->free_ctx = free_ctx_DivCtx; var->recompute_ctx = recompute_DivCtx; }
        SNEPPXVariable* pars[2]; pars[0] = a; pars[1] = b;
        set_parents(var, pars, 2);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== pow forward/backward ===== */
typedef struct { SNEPPXVariable* a; SNEPPXVariable* b; } PowCtx;
static void free_ctx_PowCtx(void* p) { SNEPPX_free(p, sizeof(PowCtx)); }
static void* recompute_PowCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    PowCtx* ctx = (PowCtx*)SNEPPX_malloc(sizeof(PowCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->b = var->parents[1]; var->free_ctx = free_ctx_PowCtx; }
    return ctx;
}
static void backward_pow(void* ctx, SNEPPXTensor* grad_output) {
    PowCtx* c = (PowCtx*)ctx;
    if (c->a->requires_grad) {
        size_t one_shape[] = {1};
        SNEPPXTensor* one = SNEPPX_tensor_ones(one_shape, 1, SNEPPX_FLOAT32);
        SNEPPXTensor* b_minus_1 = SNEPPX_tensor_sub(c->b->data, one);
        SNEPPXTensor* a_pow = SNEPPX_tensor_pow(c->a->data, b_minus_1);
        SNEPPXTensor* b_mul = SNEPPX_tensor_mul(c->b->data, a_pow);
        SNEPPXTensor* ga = SNEPPX_tensor_mul(b_mul, grad_output);
        grad_accum(&c->a->grad, ga);
        SNEPPX_tensor_destroy(one);
        SNEPPX_tensor_destroy(b_minus_1);
        SNEPPX_tensor_destroy(a_pow);
        SNEPPX_tensor_destroy(b_mul);
    }
    if (c->b->requires_grad) {
        SNEPPXTensor* a_pow_b = SNEPPX_tensor_pow(c->a->data, c->b->data);
        SNEPPXTensor* log_a = SNEPPX_tensor_log(c->a->data);
        SNEPPXTensor* ln_term = SNEPPX_tensor_mul(a_pow_b, log_a);
        SNEPPXTensor* gb = SNEPPX_tensor_mul(ln_term, grad_output);
        grad_accum(&c->b->grad, gb);
        SNEPPX_tensor_destroy(a_pow_b);
        SNEPPX_tensor_destroy(log_a);
        SNEPPX_tensor_destroy(ln_term);
    }
}
SNEPPXVariable* SNEPPX_pow(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    int rg = requires_grad(a, b);
    if (!a || !b || !a->data || !b->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_pow(a->data, b->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        PowCtx* ctx = (PowCtx*)SNEPPX_malloc(sizeof(PowCtx), 64);
        if (ctx) { ctx->a = a; ctx->b = b; var->backward_fn = backward_pow; var->backward_ctx = ctx; var->free_ctx = free_ctx_PowCtx; var->recompute_ctx = recompute_PowCtx; }
        SNEPPXVariable* pars[2]; pars[0] = a; pars[1] = b;
        set_parents(var, pars, 2);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== neg forward/backward ===== */
typedef struct { SNEPPXVariable* a; } NegCtx;
static void free_ctx_NegCtx(void* p) { SNEPPX_free(p, sizeof(NegCtx)); }
static void* recompute_NegCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    NegCtx* ctx = (NegCtx*)SNEPPX_malloc(sizeof(NegCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_NegCtx; }
    return ctx;
}
static void backward_neg(void* ctx, SNEPPXTensor* grad_output) {
    NegCtx* c = (NegCtx*)ctx;
    if (c->a->requires_grad) grad_accum(&c->a->grad, SNEPPX_tensor_neg(grad_output));
}
SNEPPXVariable* SNEPPX_neg(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_neg(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        NegCtx* ctx = (NegCtx*)SNEPPX_malloc(sizeof(NegCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_neg; var->backward_ctx = ctx; var->free_ctx = free_ctx_NegCtx; var->recompute_ctx = recompute_NegCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== matmul forward/backward ===== */
typedef struct { SNEPPXVariable* a; SNEPPXVariable* b; } MatmulCtx;
static void free_ctx_MatmulCtx(void* p) { SNEPPX_free(p, sizeof(MatmulCtx)); }
static void* recompute_MatmulCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    MatmulCtx* ctx = (MatmulCtx*)SNEPPX_malloc(sizeof(MatmulCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->b = var->parents[1]; var->free_ctx = free_ctx_MatmulCtx; }
    return ctx;
}
static void backward_matmul(void* ctx, SNEPPXTensor* grad_output) {
    MatmulCtx* c = (MatmulCtx*)ctx;
    if (c->a->requires_grad) {
        SNEPPXTensor* b_t = SNEPPX_tensor_transpose(c->b->data, c->b->data->ndim - 1, c->b->data->ndim - 2);
        SNEPPXTensor* ga = SNEPPX_tensor_matmul(grad_output, b_t);
        grad_accum(&c->a->grad, ga);
        SNEPPX_tensor_destroy(b_t);
    }
    if (c->b->requires_grad) {
        SNEPPXTensor* a_t = SNEPPX_tensor_transpose(c->a->data, c->a->data->ndim - 1, c->a->data->ndim - 2);
        SNEPPXTensor* gb = SNEPPX_tensor_matmul(a_t, grad_output);
        grad_accum(&c->b->grad, gb);
        SNEPPX_tensor_destroy(a_t);
    }
}
SNEPPXVariable* SNEPPX_matmul(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    int rg = requires_grad(a, b);
    if (!a || !b || !a->data || !b->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_matmul(a->data, b->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        MatmulCtx* ctx = (MatmulCtx*)SNEPPX_malloc(sizeof(MatmulCtx), 64);
        if (ctx) { ctx->a = a; ctx->b = b; var->backward_fn = backward_matmul; var->backward_ctx = ctx; var->free_ctx = free_ctx_MatmulCtx; var->recompute_ctx = recompute_MatmulCtx; }
        SNEPPXVariable* pars[2]; pars[0] = a; pars[1] = b;
        set_parents(var, pars, 2);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== mse_loss forward/backward ===== */
typedef struct { SNEPPXVariable* pred; SNEPPXVariable* target; } MseLossCtx;
static void free_ctx_MseLossCtx(void* p) { SNEPPX_free(p, sizeof(MseLossCtx)); }
static void* recompute_MseLossCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    MseLossCtx* ctx = (MseLossCtx*)SNEPPX_malloc(sizeof(MseLossCtx), 64);
    if (ctx) { ctx->pred = var->parents[0]; ctx->target = var->parents[1]; var->free_ctx = free_ctx_MseLossCtx; }
    return ctx;
}
static void backward_mse_loss(void* ctx, SNEPPXTensor* grad_output) {
    MseLossCtx* c = (MseLossCtx*)ctx;
    if (!c->pred->requires_grad) return;
    SNEPPXTensor* diff = SNEPPX_tensor_sub(c->pred->data, c->target->data);
    float* dd = (float*)diff->data;
    float* go = (float*)grad_output->data;
    float n = (float)(diff->size);
    float scale = 2.0f / n;
    for (size_t i = 0; i < diff->size; i++) dd[i] *= scale * go[i % grad_output->size];
    grad_accum(&c->pred->grad, diff);
}
SNEPPXVariable* SNEPPX_mse_loss(SNEPPXTape* tape, SNEPPXVariable* pred, SNEPPXVariable* target) {
    int rg = requires_grad1(pred);
    if (!pred || !target || !pred->data || !target->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_mse_loss(pred->data, target->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        MseLossCtx* ctx = (MseLossCtx*)SNEPPX_malloc(sizeof(MseLossCtx), 64);
        if (ctx) { ctx->pred = pred; ctx->target = target; var->backward_fn = backward_mse_loss; var->backward_ctx = ctx; var->free_ctx = free_ctx_MseLossCtx; var->recompute_ctx = recompute_MseLossCtx; }
        SNEPPXVariable* pars[2]; pars[0] = pred; pars[1] = target;
        set_parents(var, pars, 2);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== relu forward/backward ===== */
typedef struct { SNEPPXVariable* a; } ReluCtx;
static void free_ctx_ReluCtx(void* p) { SNEPPX_free(p, sizeof(ReluCtx)); }
static void* recompute_ReluCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    ReluCtx* ctx = (ReluCtx*)SNEPPX_malloc(sizeof(ReluCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_ReluCtx; }
    return ctx;
}
static void backward_relu(void* ctx, SNEPPXTensor* grad_output) {
    ReluCtx* c = (ReluCtx*)ctx;
    if (!c->a->requires_grad) return;
    SNEPPXTensor* ga = SNEPPX_tensor_copy(grad_output);
    float* gad = (float*)ga->data;
    float* ad = (float*)c->a->data->data;
    for (size_t i = 0; i < ga->size; i++)
        if (ad[i] <= 0.0f) gad[i] = 0.0f;
    grad_accum(&c->a->grad, ga);
}
SNEPPXVariable* SNEPPX_relu(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_relu(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        ReluCtx* ctx = (ReluCtx*)SNEPPX_malloc(sizeof(ReluCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_relu; var->backward_ctx = ctx; var->free_ctx = free_ctx_ReluCtx; var->recompute_ctx = recompute_ReluCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== gelu forward/backward ===== */
typedef struct { SNEPPXVariable* a; } GeluCtx;
static void free_ctx_GeluCtx(void* p) { SNEPPX_free(p, sizeof(GeluCtx)); }
static void* recompute_GeluCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    GeluCtx* ctx = (GeluCtx*)SNEPPX_malloc(sizeof(GeluCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_GeluCtx; }
    return ctx;
}
static void backward_gelu(void* ctx, SNEPPXTensor* grad_output) {
    GeluCtx* c = (GeluCtx*)ctx;
    if (!c->a->requires_grad) return;
    float* ad = (float*)c->a->data->data;
    SNEPPXTensor* ga = SNEPPX_tensor_copy(grad_output);
    float* gad = (float*)ga->data;
    float sqrt_2_over_pi = 0.7978845608028654f;
    for (size_t i = 0; i < ga->size; i++) {
        float x = ad[i];
        float x3 = x * x * x;
        float tanh_in = sqrt_2_over_pi * (x + 0.044715f * x3);
        float th = tanhf(tanh_in);
        float sech2 = 1.0f - th * th;
        float d_gelu = 0.5f * (1.0f + th) + 0.5f * x * sech2 * sqrt_2_over_pi * (1.0f + 3.0f * 0.044715f * x * x);
        gad[i] *= d_gelu;
    }
    grad_accum(&c->a->grad, ga);
}
SNEPPXVariable* SNEPPX_gelu(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_gelu(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        GeluCtx* ctx = (GeluCtx*)SNEPPX_malloc(sizeof(GeluCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_gelu; var->backward_ctx = ctx; var->free_ctx = free_ctx_GeluCtx; var->recompute_ctx = recompute_GeluCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== silu forward/backward ===== */
typedef struct { SNEPPXVariable* a; } SiluCtx;
static void free_ctx_SiluCtx(void* p) { SNEPPX_free(p, sizeof(SiluCtx)); }
static void* recompute_SiluCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    SiluCtx* ctx = (SiluCtx*)SNEPPX_malloc(sizeof(SiluCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_SiluCtx; }
    return ctx;
}
static void backward_silu(void* ctx, SNEPPXTensor* grad_output) {
    SiluCtx* c = (SiluCtx*)ctx;
    if (!c->a->requires_grad) return;
    float* ad = (float*)c->a->data->data;
    SNEPPXTensor* ga = SNEPPX_tensor_copy(grad_output);
    float* gad = (float*)ga->data;
    for (size_t i = 0; i < ga->size; i++) {
        float x = ad[i];
        float s = 1.0f / (1.0f + expf(-x));
        float ds = s * (1.0f + x * (1.0f - s));
        gad[i] *= ds;
    }
    grad_accum(&c->a->grad, ga);
}
SNEPPXVariable* SNEPPX_silu(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_silu(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        SiluCtx* ctx = (SiluCtx*)SNEPPX_malloc(sizeof(SiluCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_silu; var->backward_ctx = ctx; var->free_ctx = free_ctx_SiluCtx; var->recompute_ctx = recompute_SiluCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== sigmoid forward/backward ===== */
typedef struct { SNEPPXVariable* a; } SigmoidCtx;
static void free_ctx_SigmoidCtx(void* p) { SNEPPX_free(p, sizeof(SigmoidCtx)); }
static void* recompute_SigmoidCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    SigmoidCtx* ctx = (SigmoidCtx*)SNEPPX_malloc(sizeof(SigmoidCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_SigmoidCtx; }
    return ctx;
}
static void backward_sigmoid(void* ctx, SNEPPXTensor* grad_output) {
    SigmoidCtx* c = (SigmoidCtx*)ctx;
    if (!c->a->requires_grad) return;
    float* ad = (float*)c->a->data->data;
    SNEPPXTensor* ga = SNEPPX_tensor_copy(grad_output);
    float* gad = (float*)ga->data;
    for (size_t i = 0; i < ga->size; i++) {
        float s = 1.0f / (1.0f + expf(-ad[i]));
        gad[i] *= s * (1.0f - s);
    }
    grad_accum(&c->a->grad, ga);
}
SNEPPXVariable* SNEPPX_sigmoid(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_sigmoid(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        SigmoidCtx* ctx = (SigmoidCtx*)SNEPPX_malloc(sizeof(SigmoidCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_sigmoid; var->backward_ctx = ctx; var->free_ctx = free_ctx_SigmoidCtx; var->recompute_ctx = recompute_SigmoidCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== tanh forward/backward ===== */
typedef struct { SNEPPXVariable* a; } TanhCtx;
static void free_ctx_TanhCtx(void* p) { SNEPPX_free(p, sizeof(TanhCtx)); }
static void* recompute_TanhCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    TanhCtx* ctx = (TanhCtx*)SNEPPX_malloc(sizeof(TanhCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_TanhCtx; }
    return ctx;
}
static void backward_tanh(void* ctx, SNEPPXTensor* grad_output) {
    TanhCtx* c = (TanhCtx*)ctx;
    if (!c->a->requires_grad) return;
    float* ad = (float*)c->a->data->data;
    SNEPPXTensor* ga = SNEPPX_tensor_copy(grad_output);
    float* gad = (float*)ga->data;
    for (size_t i = 0; i < ga->size; i++) {
        float t = tanhf(ad[i]);
        gad[i] *= 1.0f - t * t;
    }
    grad_accum(&c->a->grad, ga);
}
SNEPPXVariable* SNEPPX_tanh(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_tanh(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        TanhCtx* ctx = (TanhCtx*)SNEPPX_malloc(sizeof(TanhCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_tanh; var->backward_ctx = ctx; var->free_ctx = free_ctx_TanhCtx; var->recompute_ctx = recompute_TanhCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== softmax forward/backward ===== */
typedef struct { SNEPPXVariable* a; size_t dim; } SoftmaxCtx;
static void free_ctx_SoftmaxCtx(void* p) { SNEPPX_free(p, sizeof(SoftmaxCtx)); }
static void* recompute_SoftmaxCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    SoftmaxCtx* ctx = (SoftmaxCtx*)SNEPPX_malloc(sizeof(SoftmaxCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->dim = n > 0 ? params[0] : 0; var->free_ctx = free_ctx_SoftmaxCtx; }
    return ctx;
}
static void backward_softmax(void* ctx, SNEPPXTensor* grad_output) {
    SoftmaxCtx* c = (SoftmaxCtx*)ctx;
    if (!c->a->requires_grad) return;
    SNEPPXTensor* s = SNEPPX_tensor_softmax(c->a->data, c->dim);
    SNEPPXTensor* s_grad = SNEPPX_tensor_mul(s, grad_output);
    SNEPPXTensor* sum_s_grad = SNEPPX_tensor_sum(s_grad, c->dim);

    size_t* expand_shape = (size_t*)SNEPPX_malloc(c->a->data->ndim * sizeof(size_t), 64);
    if (!expand_shape) { SNEPPX_tensor_destroy(s); SNEPPX_tensor_destroy(s_grad); SNEPPX_tensor_destroy(sum_s_grad); return; }
    for (size_t i = 0; i < c->a->data->ndim; i++)
        expand_shape[i] = i == c->dim ? c->a->data->shape[c->dim] : 1;
    SNEPPXTensor* expanded = SNEPPX_tensor_expand(sum_s_grad, expand_shape, c->a->data->ndim);
    SNEPPX_free(expand_shape, c->a->data->ndim * sizeof(size_t));
    if (!expanded) { SNEPPX_tensor_destroy(s); SNEPPX_tensor_destroy(s_grad); SNEPPX_tensor_destroy(sum_s_grad); return; }

    SNEPPXTensor* sub = SNEPPX_tensor_sub(grad_output, expanded);
    SNEPPXTensor* ga = SNEPPX_tensor_mul(s, sub);
    grad_accum(&c->a->grad, ga);
    SNEPPX_tensor_destroy(s);
    SNEPPX_tensor_destroy(s_grad);
    SNEPPX_tensor_destroy(sum_s_grad);
    SNEPPX_tensor_destroy(expanded);
    SNEPPX_tensor_destroy(sub);
}
SNEPPXVariable* SNEPPX_softmax(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_softmax(a->data, dim);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        SoftmaxCtx* ctx = (SoftmaxCtx*)SNEPPX_malloc(sizeof(SoftmaxCtx), 64);
        if (ctx) { ctx->a = a; ctx->dim = dim; var->backward_fn = backward_softmax; var->backward_ctx = ctx; var->free_ctx = free_ctx_SoftmaxCtx; var->recompute_ctx = recompute_SoftmaxCtx; var->params[0] = dim; var->param_count = 1; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== exp forward/backward ===== */
typedef struct { SNEPPXVariable* a; } ExpCtx;
static void free_ctx_ExpCtx(void* p) { SNEPPX_free(p, sizeof(ExpCtx)); }
static void* recompute_ExpCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    ExpCtx* ctx = (ExpCtx*)SNEPPX_malloc(sizeof(ExpCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_ExpCtx; }
    return ctx;
}
static void backward_exp(void* ctx, SNEPPXTensor* grad_output) {
    ExpCtx* c = (ExpCtx*)ctx;
    if (!c->a->requires_grad) return;
    SNEPPXTensor* e = SNEPPX_tensor_exp(c->a->data);
    SNEPPXTensor* ga = SNEPPX_tensor_mul(e, grad_output);
    grad_accum(&c->a->grad, ga);
    SNEPPX_tensor_destroy(e);
}
SNEPPXVariable* SNEPPX_exp(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_exp(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        ExpCtx* ctx = (ExpCtx*)SNEPPX_malloc(sizeof(ExpCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_exp; var->backward_ctx = ctx; var->free_ctx = free_ctx_ExpCtx; var->recompute_ctx = recompute_ExpCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== log forward/backward ===== */
typedef struct { SNEPPXVariable* a; } LogCtx;
static void free_ctx_LogCtx(void* p) { SNEPPX_free(p, sizeof(LogCtx)); }
static void* recompute_LogCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    LogCtx* ctx = (LogCtx*)SNEPPX_malloc(sizeof(LogCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_LogCtx; }
    return ctx;
}
static void backward_log(void* ctx, SNEPPXTensor* grad_output) {
    LogCtx* c = (LogCtx*)ctx;
    if (!c->a->requires_grad) return;
    SNEPPXTensor* ga = SNEPPX_tensor_div(grad_output, c->a->data);
    grad_accum(&c->a->grad, ga);
}
SNEPPXVariable* SNEPPX_log(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_log(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        LogCtx* ctx = (LogCtx*)SNEPPX_malloc(sizeof(LogCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_log; var->backward_ctx = ctx; var->free_ctx = free_ctx_LogCtx; var->recompute_ctx = recompute_LogCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== sum forward/backward ===== */
typedef struct { SNEPPXVariable* a; size_t dim; } SumCtx;
static void free_ctx_SumCtx(void* p) { SNEPPX_free(p, sizeof(SumCtx)); }
static void* recompute_SumCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    SumCtx* ctx = (SumCtx*)SNEPPX_malloc(sizeof(SumCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->dim = n > 0 ? params[0] : 0; var->free_ctx = free_ctx_SumCtx; }
    return ctx;
}
static void backward_sum(void* ctx, SNEPPXTensor* grad_output) {
    SumCtx* c = (SumCtx*)ctx;
    if (!c->a->requires_grad) return;
    size_t* expand_shape = (size_t*)SNEPPX_malloc(c->a->data->ndim * sizeof(size_t), 64);
    if (!expand_shape) return;
    for (size_t i = 0; i < c->a->data->ndim; i++)
        expand_shape[i] = i == c->dim ? c->a->data->shape[c->dim] : 1;
    SNEPPXTensor* ga = SNEPPX_tensor_expand(grad_output, expand_shape, c->a->data->ndim);
    SNEPPX_free(expand_shape, c->a->data->ndim * sizeof(size_t));
    if (ga) {
        grad_accum(&c->a->grad, ga);
    }
}
SNEPPXVariable* SNEPPX_sum(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_sum(a->data, dim);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        SumCtx* ctx = (SumCtx*)SNEPPX_malloc(sizeof(SumCtx), 64);
        if (ctx) { ctx->a = a; ctx->dim = dim; var->backward_fn = backward_sum; var->backward_ctx = ctx; var->free_ctx = free_ctx_SumCtx; var->recompute_ctx = recompute_SumCtx; var->params[0] = dim; var->param_count = 1; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== mean forward/backward ===== */
typedef struct { SNEPPXVariable* a; size_t dim; } MeanCtx;
static void free_ctx_MeanCtx(void* p) { SNEPPX_free(p, sizeof(MeanCtx)); }
static void* recompute_MeanCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    MeanCtx* ctx = (MeanCtx*)SNEPPX_malloc(sizeof(MeanCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->dim = n > 0 ? params[0] : 0; var->free_ctx = free_ctx_MeanCtx; }
    return ctx;
}
static void backward_mean(void* ctx, SNEPPXTensor* grad_output) {
    MeanCtx* c = (MeanCtx*)ctx;
    if (!c->a->requires_grad) return;
    size_t dim_size = c->a->data->shape[c->dim];
    float inv_dim = 1.0f / (float)dim_size;
    SNEPPXTensor* go = SNEPPX_tensor_copy(grad_output);
    float* god = (float*)go->data;
    for (size_t i = 0; i < go->size; i++) god[i] *= inv_dim;
    size_t* expand_shape = (size_t*)SNEPPX_malloc(c->a->data->ndim * sizeof(size_t), 64);
    if (!expand_shape) { SNEPPX_tensor_destroy(go); return; }
    for (size_t i = 0; i < c->a->data->ndim; i++)
        expand_shape[i] = i == c->dim ? c->a->data->shape[c->dim] : 1;
    SNEPPXTensor* expanded = SNEPPX_tensor_expand(go, expand_shape, c->a->data->ndim);
    SNEPPX_free(expand_shape, c->a->data->ndim * sizeof(size_t));
    if (expanded) {
        grad_accum(&c->a->grad, expanded);
    }
    SNEPPX_tensor_destroy(go);
}
SNEPPXVariable* SNEPPX_mean(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_mean(a->data, dim);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        MeanCtx* ctx = (MeanCtx*)SNEPPX_malloc(sizeof(MeanCtx), 64);
        if (ctx) { ctx->a = a; ctx->dim = dim; var->backward_fn = backward_mean; var->backward_ctx = ctx; var->free_ctx = free_ctx_MeanCtx; var->recompute_ctx = recompute_MeanCtx; var->params[0] = dim; var->param_count = 1; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== transpose forward/backward ===== */
typedef struct { SNEPPXVariable* a; size_t dim1; size_t dim2; } TransposeCtx;
static void free_ctx_TransposeCtx(void* p) { SNEPPX_free(p, sizeof(TransposeCtx)); }
static void* recompute_TransposeCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    TransposeCtx* ctx = (TransposeCtx*)SNEPPX_malloc(sizeof(TransposeCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->dim1 = n > 0 ? params[0] : 0; ctx->dim2 = n > 1 ? params[1] : 1; var->free_ctx = free_ctx_TransposeCtx; }
    return ctx;
}
static void backward_transpose(void* ctx, SNEPPXTensor* grad_output) {
    TransposeCtx* c = (TransposeCtx*)ctx;
    if (!c->a->requires_grad) return;
    SNEPPXTensor* ga = SNEPPX_tensor_transpose(grad_output, c->dim1, c->dim2);
    grad_accum(&c->a->grad, ga);
}
SNEPPXVariable* SNEPPX_transpose(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim1, size_t dim2) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_transpose(a->data, dim1, dim2);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        TransposeCtx* ctx = (TransposeCtx*)SNEPPX_malloc(sizeof(TransposeCtx), 64);
        if (ctx) { ctx->a = a; ctx->dim1 = dim1; ctx->dim2 = dim2; var->backward_fn = backward_transpose; var->backward_ctx = ctx; var->free_ctx = free_ctx_TransposeCtx; var->recompute_ctx = recompute_TransposeCtx; var->params[0] = dim1; var->params[1] = dim2; var->param_count = 2; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== reshape forward/backward ===== */
typedef struct { SNEPPXVariable* a; } ReshapeCtx;
static void free_ctx_ReshapeCtx(void* p) { SNEPPX_free(p, sizeof(ReshapeCtx)); }
static void* recompute_ReshapeCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    ReshapeCtx* ctx = (ReshapeCtx*)SNEPPX_malloc(sizeof(ReshapeCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_ReshapeCtx; }
    return ctx;
}
static void backward_reshape(void* ctx, SNEPPXTensor* grad_output) {
    ReshapeCtx* c = (ReshapeCtx*)ctx;
    if (!c->a->requires_grad) return;
    SNEPPXTensor* ga = SNEPPX_tensor_reshape(grad_output, c->a->data->shape, c->a->data->ndim);
    grad_accum(&c->a->grad, ga);
}
SNEPPXVariable* SNEPPX_reshape(SNEPPXTape* tape, SNEPPXVariable* a, const size_t* shape, size_t ndim) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_reshape(a->data, shape, ndim);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        ReshapeCtx* ctx = (ReshapeCtx*)SNEPPX_malloc(sizeof(ReshapeCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_reshape; var->backward_ctx = ctx; var->free_ctx = free_ctx_ReshapeCtx; var->recompute_ctx = recompute_ReshapeCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== dropout forward/backward ===== */
typedef struct { SNEPPXVariable* a; float rate; unsigned int seed; SNEPPXTensor* mask; } DropoutCtx;
static void free_ctx_DropoutCtx(void* p) {
    DropoutCtx* c = (DropoutCtx*)p;
    if (c->mask) SNEPPX_tensor_destroy(c->mask);
    SNEPPX_free(p, sizeof(DropoutCtx));
}
static void* recompute_DropoutCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    DropoutCtx* ctx = (DropoutCtx*)SNEPPX_malloc(sizeof(DropoutCtx), 64);
    if (ctx) {
        ctx->a = var->parents[0];
        ctx->rate = n > 0 ? *(float*)&params[0] : 0.5f;
        ctx->seed = n > 1 ? (unsigned int)params[1] : 0;
        ctx->mask = NULL;
        var->free_ctx = free_ctx_DropoutCtx;
    }
    return ctx;
}
static void backward_dropout(void* ctx, SNEPPXTensor* grad_output) {
    DropoutCtx* c = (DropoutCtx*)ctx;
    if (!c->a->requires_grad) return;
    float scale = 1.0f / (1.0f - c->rate);
    if (!c->mask) {
        unsigned long s = c->seed ? (unsigned long)c->seed : 123456789;
        c->mask = SNEPPX_tensor_zeros(c->a->data->shape, c->a->data->ndim, SNEPPX_FLOAT32);
        if (!c->mask) return;
        float* md = (float*)c->mask->data;
        for (size_t i = 0; i < c->mask->size; i++) {
            s = s * 1103515245UL + 12345UL;
            float r = (float)((s >> 16) & 0x7FFF) / 32767.0f;
            md[i] = (r >= c->rate) ? scale : 0.0f;
        }
    }
    SNEPPXTensor* ga = SNEPPX_tensor_mul(c->mask, grad_output);
    grad_accum(&c->a->grad, ga);
}
SNEPPXVariable* SNEPPX_dropout(SNEPPXTape* tape, SNEPPXVariable* a, float rate, unsigned int seed) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_dropout(a->data, rate, seed);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        DropoutCtx* ctx = (DropoutCtx*)SNEPPX_malloc(sizeof(DropoutCtx), 64);
        if (ctx) {
            ctx->a = a; ctx->rate = rate; ctx->seed = seed; ctx->mask = NULL;
            var->backward_fn = backward_dropout; var->backward_ctx = ctx;
            var->free_ctx = free_ctx_DropoutCtx; var->recompute_ctx = recompute_DropoutCtx;
            *(float*)&var->params[0] = rate; var->params[1] = (size_t)seed; var->param_count = 2;
        }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== layer_norm forward/backward ===== */
typedef struct { SNEPPXVariable* a; SNEPPXVariable* gamma; SNEPPXVariable* beta; float eps; } LayerNormCtx;
static void free_ctx_LayerNormCtx(void* p) { SNEPPX_free(p, sizeof(LayerNormCtx)); }
static void* recompute_LayerNormCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    LayerNormCtx* ctx = (LayerNormCtx*)SNEPPX_malloc(sizeof(LayerNormCtx), 64);
    if (ctx) {
        ctx->a = var->parents[0]; ctx->gamma = var->parents[1]; ctx->beta = var->parents[2];
        ctx->eps = n > 0 ? *(float*)&params[0] : 1e-5f;
        var->free_ctx = free_ctx_LayerNormCtx;
    }
    return ctx;
}
static void backward_layer_norm(void* ctx, SNEPPXTensor* grad_output) {
    LayerNormCtx* c = (LayerNormCtx*)ctx;
    if (!c->a->requires_grad && !(c->gamma && c->gamma->requires_grad) && !(c->beta && c->beta->requires_grad)) return;
    size_t d = c->a->data->shape[c->a->data->ndim - 1];
    size_t n = c->a->data->size / d;
    float* xd = (float*)c->a->data->data;
    float* gd = (float*)c->gamma->data;
    float* bd = c->beta ? (float*)c->beta->data : NULL;
    float* go = (float*)grad_output->data;
    float eps = c->eps;
    if (c->a->requires_grad) {
        SNEPPXTensor* gx = SNEPPX_tensor_zeros(c->a->data->shape, c->a->data->ndim, SNEPPX_FLOAT32);
        if (gx) {
            float* gxd = (float*)gx->data;
            for (size_t i = 0; i < n; i++) {
                double sum = 0.0, sum2 = 0.0;
                for (size_t j = 0; j < d; j++) { float v = xd[i * d + j]; sum += v; sum2 += (double)v * v; }
                double mean = sum / (double)d;
                double var = sum2 / (double)d - mean * mean;
                double inv_std = 1.0 / sqrt(var + (double)eps);
                for (size_t j = 0; j < d; j++) {
                    double x_hat = (xd[i * d + j] - mean) * inv_std;
                    double g = gd ? gd[j] : 1.0;
                    gxd[i * d + j] = (float)(g * go[i * d + j]);
                }
                double dx_norm_sum = 0.0, dx_norm_x_hat_sum = 0.0;
                for (size_t j = 0; j < d; j++) {
                    dx_norm_sum += gxd[i * d + j];
                    dx_norm_x_hat_sum += gxd[i * d + j] * (xd[i * d + j] - mean) * inv_std;
                }
                for (size_t j = 0; j < d; j++) {
                    double x_hat = (xd[i * d + j] - mean) * inv_std;
                    double grad = gxd[i * d + j] - dx_norm_sum / (double)d - x_hat * dx_norm_x_hat_sum / (double)d;
                    gxd[i * d + j] = (float)(grad * inv_std);
                }
            }
            grad_accum(&c->a->grad, gx);
        }
    }
    if (c->gamma && c->gamma->requires_grad) {
        SNEPPXTensor* gg = SNEPPX_tensor_zeros(c->gamma->data->shape, c->gamma->data->ndim, SNEPPX_FLOAT32);
        if (gg) {
            float* ggd = (float*)gg->data;
            for (size_t i = 0; i < n; i++) {
                double sum = 0.0, sum2 = 0.0;
                for (size_t j = 0; j < d; j++) { float v = xd[i * d + j]; sum += v; sum2 += (double)v * v; }
                double mean = sum / (double)d;
                double var = sum2 / (double)d - mean * mean;
                double inv_std = 1.0 / sqrt(var + (double)eps);
                for (size_t j = 0; j < d; j++) {
                    double x_hat = (xd[i * d + j] - mean) * inv_std;
                    ggd[j] += (float)(x_hat * go[i * d + j]);
                }
            }
            grad_accum(&c->gamma->grad, gg);
        }
    }
    if (c->beta && c->beta->requires_grad) {
        SNEPPXTensor* gb = SNEPPX_tensor_zeros(c->beta->data->shape, c->beta->data->ndim, SNEPPX_FLOAT32);
        if (gb) {
            float* gbd = (float*)gb->data;
            for (size_t i = 0; i < n; i++)
                for (size_t j = 0; j < d; j++)
                    gbd[j] += go[i * d + j];
            grad_accum(&c->beta->grad, gb);
        }
    }
}
SNEPPXVariable* SNEPPX_layer_norm(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* gamma, SNEPPXVariable* beta, float eps) {
    int rg = requires_grad(a, gamma) || (beta && beta->requires_grad);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_layer_norm(a->data, gamma ? gamma->data : NULL, beta ? beta->data : NULL, eps);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        LayerNormCtx* ctx = (LayerNormCtx*)SNEPPX_malloc(sizeof(LayerNormCtx), 64);
        if (ctx) {
            ctx->a = a; ctx->gamma = gamma; ctx->beta = beta; ctx->eps = eps;
            var->backward_fn = backward_layer_norm; var->backward_ctx = ctx;
            var->free_ctx = free_ctx_LayerNormCtx; var->recompute_ctx = recompute_LayerNormCtx;
            *(float*)&var->params[0] = eps; var->param_count = 1;
        }
        SNEPPXVariable* pars[3]; pars[0] = a; pars[1] = gamma; pars[2] = beta;
        set_parents(var, pars, 3);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== conv2d forward/backward ===== */
typedef struct { SNEPPXVariable *input, *kernel; size_t stride_h, stride_w, pad_h, pad_w; } Conv2DCtx;
static void free_ctx_Conv2DCtx(void* p) { SNEPPX_free(p, sizeof(Conv2DCtx)); }
static void* recompute_Conv2DCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    Conv2DCtx* ctx = (Conv2DCtx*)SNEPPX_malloc(sizeof(Conv2DCtx), 64);
    if (ctx) {
        ctx->input = var->parents[0]; ctx->kernel = var->parents[1];
        ctx->stride_h = n > 0 ? params[0] : 1; ctx->stride_w = n > 1 ? params[1] : 1;
        ctx->pad_h = n > 2 ? params[2] : 0; ctx->pad_w = n > 3 ? params[3] : 0;
        var->free_ctx = free_ctx_Conv2DCtx;
    }
    return ctx;
}
static void backward_conv2d(void* ctx, SNEPPXTensor* grad_output) {
    Conv2DCtx* c = (Conv2DCtx*)ctx;
    float* go = (float*)grad_output->data;
    size_t N = c->input->data->shape[0], C = c->input->data->shape[1];
    size_t H = c->input->data->shape[2], W = c->input->data->shape[3];
    size_t K = c->kernel->data->shape[0];
    size_t KH = c->kernel->data->shape[2], KW = c->kernel->data->shape[3];
    size_t OH = (H + 2 * c->pad_h - KH) / c->stride_h + 1;
    size_t OW = (W + 2 * c->pad_w - KW) / c->stride_w + 1;

    if (c->input->requires_grad) {
        SNEPPXTensor* gi = SNEPPX_tensor_zeros(c->input->data->shape, c->input->data->ndim, SNEPPX_FLOAT32);
        float* gid = (float*)gi->data;
        float* kd = (float*)c->kernel->data->data;
        for (size_t n = 0; n < N; n++)
            for (size_t k = 0; k < K; k++)
                for (size_t oh = 0; oh < OH; oh++)
                    for (size_t ow = 0; ow < OW; ow++) {
                        float grad_val = go[n * K * OH * OW + k * OH * OW + oh * OW + ow];
                        for (size_t kh = 0; kh < KH; kh++) {
                            size_t ih = oh * c->stride_h + kh - c->pad_h;
                            if (ih >= H) continue;
                            for (size_t kw = 0; kw < KW; kw++) {
                                size_t iw = ow * c->stride_w + kw - c->pad_w;
                                if (iw >= W) continue;
                                for (size_t cc = 0; cc < C; cc++)
                                    gid[n * C * H * W + cc * H * W + ih * W + iw] +=
                                        kd[k * C * KH * KW + cc * KH * KW + kh * KW + kw] * grad_val;
                            }
                        }
                    }
        grad_accum(&c->input->grad, gi);
    }
    if (c->kernel->requires_grad) {
        SNEPPXTensor* gk = SNEPPX_tensor_zeros(c->kernel->data->shape, c->kernel->data->ndim, SNEPPX_FLOAT32);
        float* gkd = (float*)gk->data;
        float* id = (float*)c->input->data->data;
        for (size_t n = 0; n < N; n++)
            for (size_t k = 0; k < K; k++)
                for (size_t oh = 0; oh < OH; oh++)
                    for (size_t ow = 0; ow < OW; ow++) {
                        float grad_val = go[n * K * OH * OW + k * OH * OW + oh * OW + ow];
                        for (size_t kh = 0; kh < KH; kh++) {
                            size_t ih = oh * c->stride_h + kh - c->pad_h;
                            if (ih >= H) continue;
                            for (size_t kw = 0; kw < KW; kw++) {
                                size_t iw = ow * c->stride_w + kw - c->pad_w;
                                if (iw >= W) continue;
                                for (size_t cc = 0; cc < C; cc++)
                                    gkd[k * C * KH * KW + cc * KH * KW + kh * KW + kw] +=
                                        id[n * C * H * W + cc * H * W + ih * W + iw] * grad_val;
                            }
                        }
                    }
        grad_accum(&c->kernel->grad, gk);
    }
}
SNEPPXVariable* SNEPPX_conv2d(SNEPPXTape* tape, SNEPPXVariable* input, SNEPPXVariable* kernel, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w) {
    int rg = requires_grad(input, kernel);
    if (!input || !kernel || !input->data || !kernel->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_conv2d(input->data, kernel->data, stride_h, stride_w, pad_h, pad_w);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        Conv2DCtx* ctx = (Conv2DCtx*)SNEPPX_malloc(sizeof(Conv2DCtx), 64);
        if (ctx) {
            ctx->input = input; ctx->kernel = kernel;
            ctx->stride_h = stride_h; ctx->stride_w = stride_w;
            ctx->pad_h = pad_h; ctx->pad_w = pad_w;
            var->backward_fn = backward_conv2d; var->backward_ctx = ctx;
            var->free_ctx = free_ctx_Conv2DCtx; var->recompute_ctx = recompute_Conv2DCtx;
            var->params[0] = stride_h; var->params[1] = stride_w;
            var->params[2] = pad_h; var->params[3] = pad_w; var->param_count = 4;
        }
        SNEPPXVariable* pars[2]; pars[0] = input; pars[1] = kernel;
        set_parents(var, pars, 2);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== concat forward/backward ===== */
typedef struct { SNEPPXVariable** vars; size_t num_vars; size_t dim; size_t* sizes; } ConcatCtx;
static void free_ctx_ConcatCtx(void* p) {
    ConcatCtx* c = (ConcatCtx*)p;
    if (c->sizes) SNEPPX_free(c->sizes, c->num_vars * sizeof(size_t));
    SNEPPX_free(p, sizeof(ConcatCtx));
}
static void* recompute_ConcatCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    ConcatCtx* ctx = (ConcatCtx*)SNEPPX_malloc(sizeof(ConcatCtx), 64);
    if (ctx) {
        ctx->num_vars = var->num_parents;
        ctx->vars = var->parents;
        ctx->dim = n > 0 ? params[0] : 0;
        ctx->sizes = NULL;
        var->free_ctx = free_ctx_ConcatCtx;
    }
    return ctx;
}
static void backward_concat(void* ctx, SNEPPXTensor* grad_output) {
    ConcatCtx* c = (ConcatCtx*)ctx;
    size_t offset = 0;
    for (size_t i = 0; i < c->num_vars; i++) {
        if (!c->vars[i]->requires_grad) { offset += c->sizes ? c->sizes[i] : c->vars[i]->data->shape[c->dim]; continue; }
        size_t sz = c->sizes ? c->sizes[i] : c->vars[i]->data->shape[c->dim];
        SNEPPXTensor* slice = SNEPPX_tensor_narrow(grad_output, c->dim, offset, sz);
        if (slice) {
            grad_accum(&c->vars[i]->grad, slice);
        }
        offset += sz;
    }
}
SNEPPXVariable* SNEPPX_concat(SNEPPXTape* tape, SNEPPXVariable** vars, size_t num_vars, size_t dim) {
    if (!vars || num_vars == 0) return NULL;
    int rg = 0;
    for (size_t i = 0; i < num_vars; i++) {
        if (!vars[i] || !vars[i]->data) return NULL;
        if (vars[i]->requires_grad) rg = 1;
    }
    SNEPPXTensor** t_vars = (SNEPPXTensor**)SNEPPX_malloc(num_vars * sizeof(SNEPPXTensor*), 64);
    if (!t_vars) return NULL;
    for (size_t i = 0; i < num_vars; i++) t_vars[i] = vars[i]->data;
    SNEPPXTensor* result = SNEPPX_tensor_concat((const SNEPPXTensor**)t_vars, num_vars, dim);
    SNEPPX_free(t_vars, num_vars * sizeof(SNEPPXTensor*));
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        ConcatCtx* ctx = (ConcatCtx*)SNEPPX_malloc(sizeof(ConcatCtx), 64);
        if (ctx) {
            ctx->vars = vars; ctx->num_vars = num_vars; ctx->dim = dim;
            ctx->sizes = (size_t*)SNEPPX_malloc(num_vars * sizeof(size_t), 64);
            if (ctx->sizes)
                for (size_t i = 0; i < num_vars; i++) ctx->sizes[i] = vars[i]->data->shape[dim];
            var->backward_fn = backward_concat; var->backward_ctx = ctx;
            var->free_ctx = free_ctx_ConcatCtx; var->recompute_ctx = recompute_ConcatCtx;
            var->params[0] = dim; var->param_count = 1;
        }
        set_parents(var, vars, num_vars);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== embedding forward/backward ===== */
typedef struct { SNEPPXVariable* weight; SNEPPXVariable* indices; } EmbeddingCtx;
static void free_ctx_EmbeddingCtx(void* p) { SNEPPX_free(p, sizeof(EmbeddingCtx)); }
static void* recompute_EmbeddingCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    EmbeddingCtx* ctx = (EmbeddingCtx*)SNEPPX_malloc(sizeof(EmbeddingCtx), 64);
    if (ctx) { ctx->weight = var->parents[0]; ctx->indices = var->parents[1]; var->free_ctx = free_ctx_EmbeddingCtx; }
    return ctx;
}
static void backward_embedding(void* ctx, SNEPPXTensor* grad_output) {
    EmbeddingCtx* c = (EmbeddingCtx*)ctx;
    if (!c->weight->requires_grad) return;
    int* idx = (int*)c->indices->data->data;
    size_t embed_dim = c->weight->data->shape[1];
    size_t num_indices = c->indices->data->size;
    float* go = (float*)grad_output->data;
    SNEPPXTensor* gw_t = SNEPPX_tensor_zeros(c->weight->data->shape, c->weight->data->ndim, SNEPPX_FLOAT32);
    if (!gw_t) return;
    float* gwd = (float*)gw_t->data;
    size_t vocab_size = c->weight->data->shape[0];
    for (size_t i = 0; i < num_indices; i++) {
        int ix = idx[i];
        if (ix >= 0 && (size_t)ix < vocab_size) {
            for (size_t j = 0; j < embed_dim; j++)
                gwd[(size_t)ix * embed_dim + j] += go[i * embed_dim + j];
        }
    }
    grad_accum(&c->weight->grad, gw_t);
}
SNEPPXVariable* SNEPPX_embedding(SNEPPXTape* tape, SNEPPXVariable* weight, SNEPPXVariable* indices) {
    int rg = requires_grad1(weight);
    if (!weight || !indices || !weight->data || !indices->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_embedding(weight->data, indices->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        EmbeddingCtx* ctx = (EmbeddingCtx*)SNEPPX_malloc(sizeof(EmbeddingCtx), 64);
        if (ctx) { ctx->weight = weight; ctx->indices = indices; var->backward_fn = backward_embedding; var->backward_ctx = ctx; var->free_ctx = free_ctx_EmbeddingCtx; var->recompute_ctx = recompute_EmbeddingCtx; }
        SNEPPXVariable* pars[2]; pars[0] = weight; pars[1] = indices;
        set_parents(var, pars, 2);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}
