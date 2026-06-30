#include "automatic_differentiation_framework.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

static int g_grad_enabled = 1;

void arix_no_grad_enter(void) { g_grad_enabled = 0; }
void arix_no_grad_exit(void)  { g_grad_enabled = 1; }
int  arix_no_grad_is_active(void) { return !g_grad_enabled; }

static int requires_grad(ArixVariable* a, ArixVariable* b) {
    return g_grad_enabled && ((a && a->requires_grad) || (b && b->requires_grad));
}
static int requires_grad1(ArixVariable* a) {
    return g_grad_enabled && a && a->requires_grad;
}

static void grad_accum(ArixTensor** grad, ArixTensor* contrib) {
    if (!contrib) return;
    if (!*grad) {
        *grad = contrib;
        return;
    }
    float* gd = (float*)(*grad)->data;
    float* cd = (float*)contrib->data;
    size_t n = contrib->size < (*grad)->size ? contrib->size : (*grad)->size;
    for (size_t i = 0; i < n; i++) gd[i] += cd[i];
    arix_tensor_destroy(contrib);
}

static void grad_accum_scalar(ArixTensor** grad, ArixTensor* contrib) {
    if (!contrib) return;
    if (!*grad) {
        *grad = contrib;
        return;
    }
    float* gd = (float*)(*grad)->data;
    float* cd = (float*)contrib->data;
    if (contrib->size == 1 && (*grad)->size > 1) {
        float s = cd[0];
        for (size_t i = 0; i < (*grad)->size; i++) gd[i] += s;
    } else if ((*grad)->size == 1 && contrib->size > 1) {
        float sum = 0;
        for (size_t i = 0; i < contrib->size; i++) sum += cd[i];
        gd[0] += sum;
    } else {
        size_t n = contrib->size < (*grad)->size ? contrib->size : (*grad)->size;
        for (size_t i = 0; i < n; i++) gd[i] += cd[i];
    }
    arix_tensor_destroy(contrib);
}

static void reduce_grad_to_shape(ArixTensor** grad_ptr, const ArixTensor* target) {
    ArixTensor* g = *grad_ptr;
    if (!g) return;
    if (g->ndim == target->ndim && g->size == target->size) return;

    size_t max_ndim = g->ndim > target->ndim ? g->ndim : target->ndim;
    size_t* g_shape = (size_t*)calloc(max_ndim, sizeof(size_t));
    size_t* t_shape = (size_t*)calloc(max_ndim, sizeof(size_t));
    for (size_t i = 0; i < g->ndim; i++) g_shape[max_ndim - 1 - i] = g->shape[g->ndim - 1 - i];
    for (size_t i = 0; i < target->ndim; i++) t_shape[max_ndim - 1 - i] = target->shape[target->ndim - 1 - i];

    float* data = (float*)g->data;
    int broadcast = 0;
    for (size_t i = 0; i < max_ndim; i++) {
        if (g_shape[i] != t_shape[i]) { broadcast = 1; break; }
    }

    if (!broadcast) { free(g_shape); free(t_shape); return; }

    size_t flat_size = 1;
    for (size_t i = 0; i < target->ndim; i++) flat_size *= target->shape[i];
    float* reduced = (float*)calloc(flat_size, sizeof(float));
    if (!reduced) { free(g_shape); free(t_shape); return; }

    size_t* strides_g = (size_t*)calloc(max_ndim, sizeof(size_t));
    size_t* strides_t = (size_t*)calloc(target->ndim, sizeof(size_t));
    strides_t[target->ndim - 1] = 1;
    for (size_t i = target->ndim; i > 1; i--) strides_t[i - 2] = strides_t[i - 1] * target->shape[i - 1];
    strides_g[max_ndim - 1] = 1;
    for (size_t i = max_ndim; i > 1; i--) strides_g[i - 2] = strides_g[i - 1] * g_shape[i - 1];

    for (size_t idx = 0; idx < g->size; idx++) {
        size_t tmp = idx;
        size_t t_idx = 0;
        for (size_t d = max_ndim; d > 0; d--) {
            size_t coord = tmp / strides_g[d - 1];
            tmp = tmp % strides_g[d - 1];
            size_t t_dim = d - 1;
            if (t_dim < target->ndim) {
                size_t tc = g_shape[d - 1] == t_shape[d - 1] ? coord : 0;
                t_idx += tc * strides_t[t_dim];
            }
        }
        reduced[t_idx] += data[idx];
    }

    size_t new_shape[32];
    for (size_t i = 0; i < target->ndim; i++) new_shape[i] = target->shape[i];
    ArixTensor* rt = arix_tensor_zeros(new_shape, target->ndim, ARIX_FLOAT32);
    if (rt) {
        memcpy(rt->data, reduced, flat_size * sizeof(float));
        arix_tensor_destroy(g);
        *grad_ptr = rt;
    }

    free(g_shape); free(t_shape); free(strides_g); free(strides_t); free(reduced);
}

static void backward_nop(void* ctx, ArixTensor* grad_output) {
    (void)ctx; (void)grad_output;
}

typedef struct { ArixVariable *a, *b; } BinopCtx;
static void free_ctx_BinopCtx(void* p) { arix_free(p, sizeof(BinopCtx)); }
static void* recompute_BinopCtx(ArixVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    BinopCtx* ctx = (BinopCtx*)arix_malloc(sizeof(BinopCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->b = var->parents[1]; var->free_ctx = free_ctx_BinopCtx; }
    return ctx;
}

typedef struct { ArixVariable *a, *b; ArixTensor* result; } PowCtx;
static void free_ctx_PowCtx(void* p) {
    PowCtx* c = (PowCtx*)p;
    if (c->result) arix_tensor_destroy(c->result);
    arix_free(p, sizeof(PowCtx));
}
static void* recompute_PowCtx(ArixVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    PowCtx* ctx = (PowCtx*)arix_malloc(sizeof(PowCtx), 64);
    if (ctx) {
        ctx->a = var->parents[0]; ctx->b = var->parents[1];
        ctx->result = arix_tensor_pow(ctx->a->data, ctx->b->data);
        var->free_ctx = free_ctx_PowCtx;
    }
    return ctx;
}

typedef struct { ArixVariable* a; } UnaryCtx;
static void free_ctx_UnaryCtx(void* p) { arix_free(p, sizeof(UnaryCtx)); }
static void* recompute_UnaryCtx(ArixVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    UnaryCtx* ctx = (UnaryCtx*)arix_malloc(sizeof(UnaryCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_UnaryCtx; }
    return ctx;
}

static void backward_add(void* ctx, ArixTensor* grad_output) {
    BinopCtx* c = (BinopCtx*)ctx;
    if (c->a->requires_grad) {
        ArixTensor* g = arix_tensor_copy(grad_output);
        reduce_grad_to_shape(&g, c->a->data);
        grad_accum(&c->a->grad, g);
    }
    if (c->b->requires_grad) {
        ArixTensor* g = arix_tensor_copy(grad_output);
        reduce_grad_to_shape(&g, c->b->data);
        grad_accum(&c->b->grad, g);
    }
}

static void backward_mul(void* ctx, ArixTensor* grad_output) {
    BinopCtx* c = (BinopCtx*)ctx;
    if (c->a->requires_grad) {
        ArixTensor* g = arix_tensor_mul(grad_output, c->b->data);
        reduce_grad_to_shape(&g, c->a->data);
        grad_accum(&c->a->grad, g);
    }
    if (c->b->requires_grad) {
        ArixTensor* g = arix_tensor_mul(grad_output, c->a->data);
        reduce_grad_to_shape(&g, c->b->data);
        grad_accum(&c->b->grad, g);
    }
}

static void backward_sub(void* ctx, ArixTensor* grad_output) {
    BinopCtx* c = (BinopCtx*)ctx;
    if (c->a->requires_grad) {
        ArixTensor* g = arix_tensor_copy(grad_output);
        reduce_grad_to_shape(&g, c->a->data);
        grad_accum(&c->a->grad, g);
    }
    if (c->b->requires_grad) {
        ArixTensor* g = arix_tensor_neg(grad_output);
        reduce_grad_to_shape(&g, c->b->data);
        grad_accum(&c->b->grad, g);
    }
}

static void backward_div(void* ctx, ArixTensor* grad_output) {
    BinopCtx* c = (BinopCtx*)ctx;
    if (!c->b->data) return;
    if (c->a->requires_grad) {
        ArixTensor* g = arix_tensor_div(grad_output, c->b->data);
        reduce_grad_to_shape(&g, c->a->data);
        grad_accum(&c->a->grad, g);
    }
    if (c->b->requires_grad) {
        ArixTensor* b_sq = arix_tensor_mul(c->b->data, c->b->data);
        ArixTensor* a_div_b_sq = arix_tensor_div(c->a->data, b_sq);
        ArixTensor* neg_adbsq = arix_tensor_neg(a_div_b_sq);
        ArixTensor* g = arix_tensor_mul(grad_output, neg_adbsq);
        arix_tensor_destroy(b_sq);
        arix_tensor_destroy(a_div_b_sq);
        arix_tensor_destroy(neg_adbsq);
        reduce_grad_to_shape(&g, c->b->data);
        grad_accum(&c->b->grad, g);
    }
}

static void backward_neg(void* ctx, ArixTensor* grad_output) {
    UnaryCtx* c = (UnaryCtx*)ctx;
    if (c->a->requires_grad) {
        ArixTensor* g = arix_tensor_neg(grad_output);
        grad_accum(&c->a->grad, g);
    }
}

static void backward_pow(void* ctx, ArixTensor* grad_output) {
    PowCtx* c = (PowCtx*)ctx;
    if (!c->a->data || !c->b->data) return;
    if (c->a->requires_grad) {
        ArixTensor* b_minus_1 = NULL;
        if (c->b->data->size == 1) {
            b_minus_1 = arix_tensor_copy(c->b->data);
            float bv = ((float*)b_minus_1->data)[0];
            ((float*)b_minus_1->data)[0] = bv - 1.0f;
        } else {
            ArixTensor* ones = arix_tensor_ones(c->b->data->shape, c->b->data->ndim, ARIX_FLOAT32);
            b_minus_1 = arix_tensor_sub(c->b->data, ones);
            arix_tensor_destroy(ones);
        }
        ArixTensor* a_pow_bm1 = arix_tensor_pow(c->a->data, b_minus_1);
        ArixTensor* da = arix_tensor_mul(c->b->data, a_pow_bm1);
        ArixTensor* ga = arix_tensor_mul(grad_output, da);
        arix_tensor_destroy(b_minus_1);
        arix_tensor_destroy(a_pow_bm1);
        arix_tensor_destroy(da);
        reduce_grad_to_shape(&ga, c->a->data);
        grad_accum(&c->a->grad, ga);
    }
    if (c->b->requires_grad && c->result) {
        ArixTensor* log_a = arix_tensor_log(c->a->data);
        ArixTensor* db = arix_tensor_mul(c->result, log_a);
        ArixTensor* gb = arix_tensor_mul(grad_output, db);
        arix_tensor_destroy(log_a);
        arix_tensor_destroy(db);
        reduce_grad_to_shape(&gb, c->b->data);
        grad_accum(&c->b->grad, gb);
    }
}

static void backward_matmul(void* ctx, ArixTensor* grad_output) {
    BinopCtx* c = (BinopCtx*)ctx;
    size_t m = grad_output->shape[0], n = grad_output->shape[1];
    size_t k = c->b->data->shape[0];
    float* go = (float*)grad_output->data;

    if (c->a->requires_grad) {
        float* bd = (float*)c->b->data->data;
        ArixTensor* ga = arix_tensor_zeros(c->a->data->shape, c->a->data->ndim, ARIX_FLOAT32);
        float* gad = (float*)ga->data;
        for (size_t i = 0; i < m; i++)
            for (size_t ki = 0; ki < k; ki++)
                for (size_t j = 0; j < n; j++)
                    gad[i * k + ki] += go[i * n + j] * bd[ki * n + j];
        grad_accum(&c->a->grad, ga);
    }
    if (c->b->requires_grad) {
        float* ad = (float*)c->a->data->data;
        size_t ak = c->a->data->shape[1];
        size_t bk0 = c->b->data->shape[0], bk1 = c->b->data->shape[1];
        ArixTensor* gb = arix_tensor_zeros(c->b->data->shape, c->b->data->ndim, ARIX_FLOAT32);
        float* gbd = (float*)gb->data;
        for (size_t ki = 0; ki < bk0; ki++)
            for (size_t j = 0; j < bk1; j++)
                for (size_t i = 0; i < m; i++)
                    gbd[ki * bk1 + j] += ad[i * ak + ki] * go[i * n + j];
        grad_accum(&c->b->grad, gb);
    }
}

typedef struct { ArixVariable* pred; ArixVariable* target; size_t N; } MSECtx;
static void free_ctx_MSECtx(void* p) { arix_free(p, sizeof(MSECtx)); }
static void* recompute_MSECtx(ArixVariable* var, size_t* params, size_t n) {
    MSECtx* ctx = (MSECtx*)arix_malloc(sizeof(MSECtx), 64);
    if (ctx) { ctx->pred = var->parents[0]; ctx->target = NULL; ctx->N = n > 0 ? params[0] : 0; var->free_ctx = free_ctx_MSECtx; }
    return ctx;
}

static void backward_mse(void* ctx, ArixTensor* grad_output) {
    MSECtx* c = (MSECtx*)ctx;
    (void)grad_output;
    if (c->pred->requires_grad) {
        ArixTensor* diff = arix_tensor_sub(c->pred->data, c->target->data);
        float* dd = (float*)diff->data;
        float scale = 2.0f / (float)c->N;
        for (size_t i = 0; i < diff->size; i++) dd[i] *= scale;
        grad_accum(&c->pred->grad, diff);
    }
}

static void backward_relu(void* ctx, ArixTensor* grad_output) {
    UnaryCtx* c = (UnaryCtx*)ctx;
    if (!c->a->requires_grad) return;
    ArixTensor* mask = arix_tensor_copy(c->a->data);
    float* md = (float*)mask->data;
    for (size_t i = 0; i < mask->size; i++) md[i] = md[i] > 0.0f ? 1.0f : 0.0f;
    ArixTensor* g = arix_tensor_mul(grad_output, mask);
    grad_accum(&c->a->grad, g);
    arix_tensor_destroy(mask);
}

static void backward_gelu(void* ctx, ArixTensor* grad_output) {
    UnaryCtx* c = (UnaryCtx*)ctx;
    if (!c->a->requires_grad) return;
    float* ad = (float*)c->a->data->data;
    size_t n = c->a->data->size;
    ArixTensor* g = arix_tensor_zeros(c->a->data->shape, c->a->data->ndim, ARIX_FLOAT32);
    float* gd = (float*)g->data;
    const float sqrt2 = 1.41421356237f;
    const float sqrt2pi = 2.50662827463f;
    for (size_t i = 0; i < n; i++) {
        float x = ad[i];
        float x_over_sqrt2 = x / sqrt2;
        float erf_val = erff(x_over_sqrt2);
        float grad_factor = 0.5f * (1.0f + erf_val) + x * expf(-x * x / 2.0f) / sqrt2pi;
        gd[i] = grad_factor * ((float*)grad_output->data)[i];
    }
    grad_accum(&c->a->grad, g);
}

static void backward_silu(void* ctx, ArixTensor* grad_output) {
    UnaryCtx* c = (UnaryCtx*)ctx;
    if (!c->a->requires_grad) return;
    float* ad = (float*)c->a->data->data;
    size_t n = c->a->data->size;
    ArixTensor* g = arix_tensor_zeros(c->a->data->shape, c->a->data->ndim, ARIX_FLOAT32);
    float* gd = (float*)g->data;
    float* go = (float*)grad_output->data;
    for (size_t i = 0; i < n; i++) {
        float x = ad[i];
        float s = 1.0f / (1.0f + expf(-x));
        float ds = s * (1.0f - s);
        gd[i] = go[i] * (s + x * ds);
    }
    grad_accum(&c->a->grad, g);
}

static void backward_sigmoid(void* ctx, ArixTensor* grad_output) {
    UnaryCtx* c = (UnaryCtx*)ctx;
    if (!c->a->requires_grad) return;
    size_t sz = c->a->data->size;
    float* sd = (float*)c->a->data->data;
    float* go = (float*)grad_output->data;
    ArixTensor* g = arix_tensor_zeros(c->a->data->shape, c->a->data->ndim, ARIX_FLOAT32);
    float* gd = (float*)g->data;
    for (size_t i = 0; i < sz; i++) {
        float s = 1.0f / (1.0f + expf(-sd[i]));
        gd[i] = go[i] * s * (1.0f - s);
    }
    grad_accum(&c->a->grad, g);
}

typedef struct { ArixVariable* a; size_t dim; } SoftmaxCtx;
static void free_ctx_SoftmaxCtx(void* p) { arix_free(p, sizeof(SoftmaxCtx)); }
static void* recompute_SoftmaxCtx(ArixVariable* var, size_t* params, size_t n) {
    SoftmaxCtx* ctx = (SoftmaxCtx*)arix_malloc(sizeof(SoftmaxCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->dim = n > 0 ? params[0] : 0; var->free_ctx = free_ctx_SoftmaxCtx; }
    return ctx;
}

static void backward_softmax(void* ctx, ArixTensor* grad_output) {
    SoftmaxCtx* c = (SoftmaxCtx*)ctx;
    if (!c->a->requires_grad) return;
    ArixTensor* sm = arix_tensor_softmax(c->a->data, c->dim);
    float* sd = (float*)sm->data;
    float* go = (float*)grad_output->data;
    size_t outer = 1, inner = 1;
    for (size_t i = 0; i < c->dim; i++) outer *= c->a->data->shape[i];
    size_t dim_size = c->a->data->shape[c->dim];
    for (size_t i = c->dim + 1; i < c->a->data->ndim; i++) inner *= c->a->data->shape[i];
    ArixTensor* g = arix_tensor_zeros(c->a->data->shape, c->a->data->ndim, ARIX_FLOAT32);
    float* gd = (float*)g->data;
    for (size_t o = 0; o < outer; o++) {
        for (size_t d = 0; d < dim_size; d++) {
            for (size_t i = 0; i < inner; i++) {
                size_t idx = o * dim_size * inner + d * inner + i;
                float si = sd[idx];
                float sum = 0.0f;
                for (size_t j = 0; j < dim_size; j++) {
                    size_t jdx = o * dim_size * inner + j * inner + i;
                    float sj = sd[jdx];
                    sum += go[jdx] * sj;
                }
                gd[idx] = si * (go[idx] - sum);
            }
        }
    }
    grad_accum(&c->a->grad, g);
    arix_tensor_destroy(sm);
}

static void backward_exp(void* ctx, ArixTensor* grad_output) {
    UnaryCtx* c = (UnaryCtx*)ctx;
    if (!c->a->requires_grad) return;
    ArixTensor* e = arix_tensor_exp(c->a->data);
    ArixTensor* g = arix_tensor_mul(grad_output, e);
    grad_accum(&c->a->grad, g);
    arix_tensor_destroy(e);
}

static void backward_log(void* ctx, ArixTensor* grad_output) {
    UnaryCtx* c = (UnaryCtx*)ctx;
    if (!c->a->requires_grad) return;
    float* ad = (float*)c->a->data->data;
    size_t n = c->a->data->size;
    float* go = (float*)grad_output->data;
    ArixTensor* g = arix_tensor_zeros(c->a->data->shape, c->a->data->ndim, ARIX_FLOAT32);
    float* gd = (float*)g->data;
    for (size_t i = 0; i < n; i++) gd[i] = go[i] / ad[i];
    grad_accum(&c->a->grad, g);
}

static void backward_tanh(void* ctx, ArixTensor* grad_output) {
    UnaryCtx* c = (UnaryCtx*)ctx;
    if (!c->a->requires_grad) return;
    ArixTensor* t = arix_tensor_tanh(c->a->data);
    float* td = (float*)t->data;
    size_t n = t->size;
    ArixTensor* g = arix_tensor_zeros(c->a->data->shape, c->a->data->ndim, ARIX_FLOAT32);
    float* gd = (float*)g->data;
    float* go = (float*)grad_output->data;
    for (size_t i = 0; i < n; i++) gd[i] = go[i] * (1.0f - td[i] * td[i]);
    grad_accum(&c->a->grad, g);
    arix_tensor_destroy(t);
}

typedef struct { ArixVariable* a; size_t dim; } DimCtx;
static void free_ctx_DimCtx(void* p) { arix_free(p, sizeof(DimCtx)); }
static void* recompute_DimCtx(ArixVariable* var, size_t* params, size_t n) {
    DimCtx* ctx = (DimCtx*)arix_malloc(sizeof(DimCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->dim = n > 0 ? params[0] : 0; var->free_ctx = free_ctx_DimCtx; }
    return ctx;
}

static void backward_sum(void* ctx, ArixTensor* grad_output) {
    DimCtx* c = (DimCtx*)ctx;
    if (!c->a->requires_grad) return;
    ArixTensor* g = arix_tensor_zeros(c->a->data->shape, c->a->data->ndim, ARIX_FLOAT32);
    float* gd = (float*)g->data;
    float* go = (float*)grad_output->data;
    size_t outer = 1, inner = 1;
    for (size_t i = 0; i < c->dim; i++) outer *= c->a->data->shape[i];
    size_t dim_size = c->a->data->shape[c->dim];
    for (size_t i = c->dim + 1; i < c->a->data->ndim; i++) inner *= c->a->data->shape[i];
    for (size_t o = 0; o < outer; o++)
        for (size_t d = 0; d < dim_size; d++)
            for (size_t i = 0; i < inner; i++)
                gd[o * dim_size * inner + d * inner + i] = go[o * inner + i];
    grad_accum(&c->a->grad, g);
}

static void backward_mean(void* ctx, ArixTensor* grad_output) {
    DimCtx* c = (DimCtx*)ctx;
    if (!c->a->requires_grad) return;
    float scale = 1.0f / (float)c->a->data->shape[c->dim];
    ArixTensor* g = arix_tensor_zeros(c->a->data->shape, c->a->data->ndim, ARIX_FLOAT32);
    float* gd = (float*)g->data;
    float* go = (float*)grad_output->data;
    size_t outer = 1, inner = 1;
    for (size_t i = 0; i < c->dim; i++) outer *= c->a->data->shape[i];
    size_t dim_size = c->a->data->shape[c->dim];
    for (size_t i = c->dim + 1; i < c->a->data->ndim; i++) inner *= c->a->data->shape[i];
    for (size_t o = 0; o < outer; o++)
        for (size_t d = 0; d < dim_size; d++)
            for (size_t i = 0; i < inner; i++)
                gd[o * dim_size * inner + d * inner + i] = go[o * inner + i] * scale;
    grad_accum(&c->a->grad, g);
}

typedef struct { ArixVariable* a; size_t dim1, dim2; } TransposeCtx;
static void free_ctx_TransposeCtx(void* p) { arix_free(p, sizeof(TransposeCtx)); }
static void* recompute_TransposeCtx(ArixVariable* var, size_t* params, size_t n) {
    TransposeCtx* ctx = (TransposeCtx*)arix_malloc(sizeof(TransposeCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->dim1 = n > 0 ? params[0] : 0; ctx->dim2 = n > 1 ? params[1] : 1; var->free_ctx = free_ctx_TransposeCtx; }
    return ctx;
}

static void backward_transpose(void* ctx, ArixTensor* grad_output) {
    TransposeCtx* c = (TransposeCtx*)ctx;
    if (!c->a->requires_grad) return;
    ArixTensor* g = arix_tensor_transpose(grad_output, c->dim1, c->dim2);
    grad_accum(&c->a->grad, g);
}

typedef struct { ArixVariable* a; ArixTensor* mask; float rate; } DropoutCtx;
static void free_ctx_DropoutCtx(void* p) {
    DropoutCtx* c = (DropoutCtx*)p;
    if (c->mask) arix_tensor_destroy(c->mask);
    arix_free(p, sizeof(DropoutCtx));
}

static void backward_dropout(void* ctx, ArixTensor* grad_output) {
    DropoutCtx* c = (DropoutCtx*)ctx;
    if (!c->a->requires_grad) return;
    float scale = 1.0f / (1.0f - c->rate);
    ArixTensor* g = arix_tensor_mul(grad_output, c->mask);
    float* gd = (float*)g->data;
    for (size_t i = 0; i < g->size; i++) gd[i] *= scale;
    grad_accum(&c->a->grad, g);
}

typedef struct { ArixVariable *a, *gamma, *beta; float eps; } LayerNormCtx;
static void free_ctx_LayerNormCtx(void* p) { arix_free(p, sizeof(LayerNormCtx)); }
static void* recompute_LayerNormCtx(ArixVariable* var, size_t* params, size_t n) {
    LayerNormCtx* ctx = (LayerNormCtx*)arix_malloc(sizeof(LayerNormCtx), 64);
    if (ctx) {
        ctx->a = var->parents[0];
        ctx->gamma = var->num_parents > 1 ? var->parents[1] : NULL;
        ctx->beta = var->num_parents > 2 ? var->parents[2] : NULL;
        ctx->eps = n > 0 ? *(float*)&params[0] : 1e-5f;
        var->free_ctx = free_ctx_LayerNormCtx;
    }
    return ctx;
}

static void backward_layer_norm(void* ctx, ArixTensor* grad_output) {
    LayerNormCtx* c = (LayerNormCtx*)ctx;
    float* xd = (float*)c->a->data->data;
    float* gd = (float*)grad_output->data;
    size_t n = c->a->data->size;
    size_t last_dim = c->a->data->shape[c->a->data->ndim - 1];
    size_t outer = n / last_dim;

    if (c->a->requires_grad) {
        ArixTensor* gx = arix_tensor_zeros(c->a->data->shape, c->a->data->ndim, ARIX_FLOAT32);
        float* gxd = (float*)gx->data;
        for (size_t o = 0; o < outer; o++) {
            float mean = 0, var = 0;
            for (size_t i = 0; i < last_dim; i++) mean += xd[o * last_dim + i];
            mean /= (float)last_dim;
            for (size_t i = 0; i < last_dim; i++) { float d = xd[o * last_dim + i] - mean; var += d * d; }
            var /= (float)last_dim;
            float inv_std = 1.0f / sqrtf(var + c->eps);
            float gamma_val = c->gamma && c->gamma->data ? ((float*)c->gamma->data->data)[0] : 1.0f;
            float dnorm_sum = 0, dnorm_dot = 0;
            for (size_t i = 0; i < last_dim; i++) {
                float x_hat = (xd[o * last_dim + i] - mean) * inv_std;
                float dnorm = gd[o * last_dim + i] * gamma_val;
                dnorm_sum += dnorm;
                dnorm_dot += dnorm * x_hat;
            }
            for (size_t i = 0; i < last_dim; i++) {
                float x_hat = (xd[o * last_dim + i] - mean) * inv_std;
                float dnorm = gd[o * last_dim + i] * gamma_val;
                gxd[o * last_dim + i] = (dnorm - dnorm_sum / (float)last_dim - x_hat * dnorm_dot / (float)last_dim) * inv_std;
            }
        }
        grad_accum(&c->a->grad, gx);
    }
    if (c->gamma && c->gamma->requires_grad) {
        ArixTensor* gg = arix_tensor_zeros(c->gamma->data->shape, c->gamma->data->ndim, ARIX_FLOAT32);
        float* ggd = (float*)gg->data;
        for (size_t o = 0; o < outer; o++) {
            float mean = 0;
            for (size_t i = 0; i < last_dim; i++) mean += xd[o * last_dim + i];
            mean /= (float)last_dim;
            float var = 0;
            for (size_t i = 0; i < last_dim; i++) { float d = xd[o * last_dim + i] - mean; var += d * d; }
            var /= (float)last_dim;
            float inv_std = 1.0f / sqrtf(var + c->eps);
            for (size_t i = 0; i < last_dim; i++) {
                float x_hat = (xd[o * last_dim + i] - mean) * inv_std;
                ggd[i] += gd[o * last_dim + i] * x_hat;
            }
        }
        grad_accum(&c->gamma->grad, gg);
    }
    if (c->beta && c->beta->requires_grad) {
        ArixTensor* gb = arix_tensor_zeros(c->beta->data->shape, c->beta->data->ndim, ARIX_FLOAT32);
        float* gbd = (float*)gb->data;
        for (size_t i = 0; i < n; i++) gbd[i % last_dim] += gd[i];
        grad_accum(&c->beta->grad, gb);
    }
}

typedef struct { ArixVariable *input, *kernel; size_t stride_h, stride_w, pad_h, pad_w; } Conv2DCtx;
static void free_ctx_Conv2DCtx(void* p) { arix_free(p, sizeof(Conv2DCtx)); }
static void* recompute_Conv2DCtx(ArixVariable* var, size_t* params, size_t n) {
    Conv2DCtx* ctx = (Conv2DCtx*)arix_malloc(sizeof(Conv2DCtx), 64);
    if (ctx) {
        ctx->input = var->parents[0]; ctx->kernel = var->parents[1];
        ctx->stride_h = n > 0 ? params[0] : 1;
        ctx->stride_w = n > 1 ? params[1] : 1;
        ctx->pad_h = n > 2 ? params[2] : 0;
        ctx->pad_w = n > 3 ? params[3] : 0;
        var->free_ctx = free_ctx_Conv2DCtx;
    }
    return ctx;
}

static void backward_conv2d(void* ctx, ArixTensor* grad_output) {
    Conv2DCtx* c = (Conv2DCtx*)ctx;
    float* go = (float*)grad_output->data;
    size_t N = c->input->data->shape[0], C = c->input->data->shape[1], H = c->input->data->shape[2], W = c->input->data->shape[3];
    size_t K = c->kernel->data->shape[0];
    size_t KH = c->kernel->data->shape[2], KW = c->kernel->data->shape[3];
    size_t OH = (H + 2 * c->pad_h - KH) / c->stride_h + 1;
    size_t OW = (W + 2 * c->pad_w - KW) / c->stride_w + 1;

    if (c->input->requires_grad) {
        ArixTensor* gi = arix_tensor_zeros(c->input->data->shape, c->input->data->ndim, ARIX_FLOAT32);
        float* gid = (float*)gi->data;
        float* kd = (float*)c->kernel->data->data;
        for (size_t n = 0; n < N; n++)
            for (size_t k = 0; k < K; k++)
                for (size_t oh = 0; oh < OH; oh++)
                    for (size_t ow = 0; ow < OW; ow++) {
                        float grad_val = go[n * K * OH * OW + k * OH * OW + oh * OW + ow];
                        for (size_t kh = 0; kh < KH; kh++)
                            for (size_t kw = 0; kw < KW; kw++) {
                                size_t ih = oh * c->stride_h + kh - c->pad_h;
                                size_t iw = ow * c->stride_w + kw - c->pad_w;
                                if (ih < H && iw < W)
                                    for (size_t cc = 0; cc < C; cc++)
                                        gid[n * C * H * W + cc * H * W + ih * W + iw] +=
                                            kd[k * C * KH * KW + cc * KH * KW + kh * KW + kw] * grad_val;
                            }
                    }
        grad_accum(&c->input->grad, gi);
    }
    if (c->kernel->requires_grad) {
        ArixTensor* gk = arix_tensor_zeros(c->kernel->data->shape, c->kernel->data->ndim, ARIX_FLOAT32);
        float* gkd = (float*)gk->data;
        float* id = (float*)c->input->data->data;
        for (size_t n = 0; n < N; n++)
            for (size_t k = 0; k < K; k++)
                for (size_t oh = 0; oh < OH; oh++)
                    for (size_t ow = 0; ow < OW; ow++) {
                        float grad_val = go[n * K * OH * OW + k * OH * OW + oh * OW + ow];
                        for (size_t kh = 0; kh < KH; kh++)
                            for (size_t kw = 0; kw < KW; kw++) {
                                size_t ih = oh * c->stride_h + kh - c->pad_h;
                                size_t iw = ow * c->stride_w + kw - c->pad_w;
                                if (ih < H && iw < W)
                                    for (size_t cc = 0; cc < C; cc++)
                                        gkd[k * C * KH * KW + cc * KH * KW + kh * KW + kw] +=
                                            id[n * C * H * W + cc * H * W + ih * W + iw] * grad_val;
                            }
                    }
        grad_accum(&c->kernel->grad, gk);
    }
}

static void set_parents(ArixVariable* var, ArixVariable** par, size_t n) {
    var->parents = (ArixVariable**)arix_malloc(n * sizeof(ArixVariable*), 64);
    if (var->parents) {
        memcpy(var->parents, par, n * sizeof(ArixVariable*));
        var->num_parents = n;
        for (size_t i = 0; i < n; i++)
            if (par[i]) par[i]->ref_count++;
    }
}

static ArixVariable* op_binary(ArixTape* tape, ArixVariable* a, ArixVariable* b,
                               ArixTensor* (*tensor_fn)(const ArixTensor*, const ArixTensor*),
                               BackwardFn bw, ArixVariable** parents, size_t np) {
    if (!a || !b || !a->data || !b->data) return NULL;
    int rg = requires_grad(a, b);
    ArixTensor* result = tensor_fn(a->data, b->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    var->backward_fn = bw;
    if (bw && parents) set_parents(var, parents, np);
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

static ArixVariable* op_unary(ArixTape* tape, ArixVariable* a,
                              ArixTensor* (*tensor_fn)(const ArixTensor*),
                              BackwardFn bw) {
    if (!a || !a->data) return NULL;
    int rg = requires_grad1(a);
    ArixTensor* result = tensor_fn(a->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    var->backward_fn = bw;
    if (bw) set_parents(var, &a, 1);
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_add(ArixTape* tape, ArixVariable* a, ArixVariable* b) {
    int rg = requires_grad(a, b);
    if (!a || !b || !a->data || !b->data) return NULL;
    ArixTensor* result = arix_tensor_add(a->data, b->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        BinopCtx* ctx = (BinopCtx*)arix_malloc(sizeof(BinopCtx), 64);
        if (ctx) { ctx->a = a; ctx->b = b; var->backward_fn = backward_add; var->backward_ctx = ctx; var->free_ctx = free_ctx_BinopCtx; var->recompute_ctx = recompute_BinopCtx; }
        ArixVariable* pars[2]; pars[0] = a; pars[1] = b;
        set_parents(var, pars, 2);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_sub(ArixTape* tape, ArixVariable* a, ArixVariable* b) {
    int rg = requires_grad(a, b);
    if (!a || !b || !a->data || !b->data) return NULL;
    ArixTensor* result = arix_tensor_sub(a->data, b->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        BinopCtx* ctx = (BinopCtx*)arix_malloc(sizeof(BinopCtx), 64);
        if (ctx) { ctx->a = a; ctx->b = b; var->backward_fn = backward_sub; var->backward_ctx = ctx; var->free_ctx = free_ctx_BinopCtx; var->recompute_ctx = recompute_BinopCtx; }
        ArixVariable* pars[2]; pars[0] = a; pars[1] = b;
        set_parents(var, pars, 2);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_mul(ArixTape* tape, ArixVariable* a, ArixVariable* b) {
    int rg = requires_grad(a, b);
    if (!a || !b || !a->data || !b->data) return NULL;
    ArixTensor* result = arix_tensor_mul(a->data, b->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        BinopCtx* ctx = (BinopCtx*)arix_malloc(sizeof(BinopCtx), 64);
        if (ctx) { ctx->a = a; ctx->b = b; var->backward_fn = backward_mul; var->backward_ctx = ctx; var->free_ctx = free_ctx_BinopCtx; var->recompute_ctx = recompute_BinopCtx; }
        ArixVariable* pars[2]; pars[0] = a; pars[1] = b;
        set_parents(var, pars, 2);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_div(ArixTape* tape, ArixVariable* a, ArixVariable* b) {
    int rg = requires_grad(a, b);
    if (!a || !b || !a->data || !b->data) return NULL;
    ArixTensor* result = arix_tensor_div(a->data, b->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        BinopCtx* ctx = (BinopCtx*)arix_malloc(sizeof(BinopCtx), 64);
        if (ctx) { ctx->a = a; ctx->b = b; var->backward_fn = backward_div; var->backward_ctx = ctx; var->free_ctx = free_ctx_BinopCtx; var->recompute_ctx = recompute_BinopCtx; }
        ArixVariable* pars[2]; pars[0] = a; pars[1] = b;
        set_parents(var, pars, 2);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_pow(ArixTape* tape, ArixVariable* a, ArixVariable* b) {
    int rg = requires_grad(a, b);
    if (!a || !b || !a->data || !b->data) return NULL;
    ArixTensor* result = arix_tensor_pow(a->data, b->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        PowCtx* ctx = (PowCtx*)arix_malloc(sizeof(PowCtx), 64);
        if (ctx) {
            ctx->a = a; ctx->b = b;
            ctx->result = arix_tensor_copy(result);
            var->backward_fn = backward_pow;
            var->backward_ctx = ctx;
            var->free_ctx = free_ctx_PowCtx;
            var->recompute_ctx = recompute_PowCtx;
        }
        ArixVariable* pars[2]; pars[0] = a; pars[1] = b;
        set_parents(var, pars, 2);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_neg(ArixTape* tape, ArixVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_neg(a->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        UnaryCtx* ctx = (UnaryCtx*)arix_malloc(sizeof(UnaryCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_neg; var->backward_ctx = ctx; var->free_ctx = free_ctx_UnaryCtx; var->recompute_ctx = recompute_UnaryCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

static void backward_minimum(void* ctx, ArixTensor* grad_output) {
    BinopCtx* c = (BinopCtx*)ctx;
    if (c->a->requires_grad) {
        ArixTensor* g = arix_tensor_zeros(c->a->data->shape, c->a->data->ndim, ARIX_FLOAT32);
        float* gd = (float*)g->data;
        float* ad = (float*)c->a->data->data;
        float* bd = (float*)c->b->data->data;
        float* go = (float*)grad_output->data;
        size_t sz = c->a->data->size < c->b->data->size ? c->a->data->size : c->b->data->size;
        for (size_t i = 0; i < sz; i++)
            gd[i] = (ad[i] <= bd[i]) ? go[i % grad_output->size] : 0.0f;
        reduce_grad_to_shape(&g, c->a->data);
        grad_accum(&c->a->grad, g);
    }
    if (c->b->requires_grad) {
        ArixTensor* g = arix_tensor_zeros(c->b->data->shape, c->b->data->ndim, ARIX_FLOAT32);
        float* gd = (float*)g->data;
        float* ad = (float*)c->a->data->data;
        float* bd = (float*)c->b->data->data;
        float* go = (float*)grad_output->data;
        size_t sz = c->a->data->size < c->b->data->size ? c->a->data->size : c->b->data->size;
        for (size_t i = 0; i < sz; i++)
            gd[i] = (ad[i] > bd[i]) ? go[i % grad_output->size] : 0.0f;
        reduce_grad_to_shape(&g, c->b->data);
        grad_accum(&c->b->grad, g);
    }
}

static void backward_maximum(void* ctx, ArixTensor* grad_output) {
    BinopCtx* c = (BinopCtx*)ctx;
    if (c->a->requires_grad) {
        ArixTensor* g = arix_tensor_zeros(c->a->data->shape, c->a->data->ndim, ARIX_FLOAT32);
        float* gd = (float*)g->data;
        float* ad = (float*)c->a->data->data;
        float* bd = (float*)c->b->data->data;
        float* go = (float*)grad_output->data;
        size_t sz = c->a->data->size < c->b->data->size ? c->a->data->size : c->b->data->size;
        for (size_t i = 0; i < sz; i++)
            gd[i] = (ad[i] >= bd[i]) ? go[i % grad_output->size] : 0.0f;
        reduce_grad_to_shape(&g, c->a->data);
        grad_accum(&c->a->grad, g);
    }
    if (c->b->requires_grad) {
        ArixTensor* g = arix_tensor_zeros(c->b->data->shape, c->b->data->ndim, ARIX_FLOAT32);
        float* gd = (float*)g->data;
        float* ad = (float*)c->a->data->data;
        float* bd = (float*)c->b->data->data;
        float* go = (float*)grad_output->data;
        size_t sz = c->a->data->size < c->b->data->size ? c->a->data->size : c->b->data->size;
        for (size_t i = 0; i < sz; i++)
            gd[i] = (ad[i] < bd[i]) ? go[i % grad_output->size] : 0.0f;
        reduce_grad_to_shape(&g, c->b->data);
        grad_accum(&c->b->grad, g);
    }
}

ArixVariable* arix_minimum(ArixTape* tape, ArixVariable* a, ArixVariable* b) {
    int rg = requires_grad(a, b);
    if (!a || !b || !a->data || !b->data) return NULL;
    ArixTensor* result = arix_tensor_minimum(a->data, b->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        BinopCtx* ctx = (BinopCtx*)arix_malloc(sizeof(BinopCtx), 64);
        if (ctx) { ctx->a = a; ctx->b = b; var->backward_fn = backward_minimum; var->backward_ctx = ctx; var->free_ctx = free_ctx_BinopCtx; var->recompute_ctx = recompute_BinopCtx; }
        ArixVariable* pars[2]; pars[0] = a; pars[1] = b;
        set_parents(var, pars, 2);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_maximum(ArixTape* tape, ArixVariable* a, ArixVariable* b) {
    int rg = requires_grad(a, b);
    if (!a || !b || !a->data || !b->data) return NULL;
    ArixTensor* result = arix_tensor_maximum(a->data, b->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        BinopCtx* ctx = (BinopCtx*)arix_malloc(sizeof(BinopCtx), 64);
        if (ctx) { ctx->a = a; ctx->b = b; var->backward_fn = backward_maximum; var->backward_ctx = ctx; var->free_ctx = free_ctx_BinopCtx; var->recompute_ctx = recompute_BinopCtx; }
        ArixVariable* pars[2]; pars[0] = a; pars[1] = b;
        set_parents(var, pars, 2);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_matmul(ArixTape* tape, ArixVariable* a, ArixVariable* b) {
    int rg = requires_grad(a, b);
    if (!a || !b || !a->data || !b->data) return NULL;
    ArixTensor* result = arix_tensor_matmul(a->data, b->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        BinopCtx* ctx = (BinopCtx*)arix_malloc(sizeof(BinopCtx), 64);
        if (ctx) { ctx->a = a; ctx->b = b; var->backward_fn = backward_matmul; var->backward_ctx = ctx; var->free_ctx = free_ctx_BinopCtx; var->recompute_ctx = recompute_BinopCtx; }
        ArixVariable* pars[2]; pars[0] = a; pars[1] = b;
        set_parents(var, pars, 2);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_mse_loss(ArixTape* tape, ArixVariable* pred, ArixVariable* target) {
    if (!pred || !target || !pred->data || !target->data) return NULL;
    ArixTensor* lt = arix_tensor_mse_loss(pred->data, target->data);
    if (!lt) return NULL;
    ArixVariable* var = arix_variable_create(lt, 0);
    if (!var) { arix_tensor_destroy(lt); return NULL; }
    int rg = requires_grad(pred, target);
    if (rg && g_grad_enabled) {
        MSECtx* ctx = (MSECtx*)arix_malloc(sizeof(MSECtx), 64);
        if (ctx) { ctx->pred = pred; ctx->target = target; ctx->N = pred->data->size; var->backward_fn = backward_mse; var->backward_ctx = ctx; var->free_ctx = free_ctx_MSECtx; }
        set_parents(var, &pred, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_relu(ArixTape* tape, ArixVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_relu(a->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        UnaryCtx* ctx = (UnaryCtx*)arix_malloc(sizeof(UnaryCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_relu; var->backward_ctx = ctx; var->free_ctx = free_ctx_UnaryCtx; var->recompute_ctx = recompute_UnaryCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_gelu(ArixTape* tape, ArixVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_gelu(a->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        UnaryCtx* ctx = (UnaryCtx*)arix_malloc(sizeof(UnaryCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_gelu; var->backward_ctx = ctx; var->free_ctx = free_ctx_UnaryCtx; var->recompute_ctx = recompute_UnaryCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_silu(ArixTape* tape, ArixVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_silu(a->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        UnaryCtx* ctx = (UnaryCtx*)arix_malloc(sizeof(UnaryCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_silu; var->backward_ctx = ctx; var->free_ctx = free_ctx_UnaryCtx; var->recompute_ctx = recompute_UnaryCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_sigmoid(ArixTape* tape, ArixVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_sigmoid(a->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        UnaryCtx* ctx = (UnaryCtx*)arix_malloc(sizeof(UnaryCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_sigmoid; var->backward_ctx = ctx; var->free_ctx = free_ctx_UnaryCtx; var->recompute_ctx = recompute_UnaryCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_tanh(ArixTape* tape, ArixVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_tanh(a->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        UnaryCtx* ctx = (UnaryCtx*)arix_malloc(sizeof(UnaryCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_tanh; var->backward_ctx = ctx; var->free_ctx = free_ctx_UnaryCtx; var->recompute_ctx = recompute_UnaryCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_softmax(ArixTape* tape, ArixVariable* a, size_t dim) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_softmax(a->data, dim);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        SoftmaxCtx* ctx = (SoftmaxCtx*)arix_malloc(sizeof(SoftmaxCtx), 64);
        if (ctx) { ctx->a = a; ctx->dim = dim; var->backward_fn = backward_softmax; var->backward_ctx = ctx; var->free_ctx = free_ctx_SoftmaxCtx; var->recompute_ctx = recompute_SoftmaxCtx; var->params[0] = dim; var->param_count = 1; }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_exp(ArixTape* tape, ArixVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_exp(a->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        UnaryCtx* ctx = (UnaryCtx*)arix_malloc(sizeof(UnaryCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_exp; var->backward_ctx = ctx; var->free_ctx = free_ctx_UnaryCtx; var->recompute_ctx = recompute_UnaryCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_log(ArixTape* tape, ArixVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_log(a->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        UnaryCtx* ctx = (UnaryCtx*)arix_malloc(sizeof(UnaryCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_log; var->backward_ctx = ctx; var->free_ctx = free_ctx_UnaryCtx; var->recompute_ctx = recompute_UnaryCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_sum(ArixTape* tape, ArixVariable* a, size_t dim) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_sum(a->data, dim);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        DimCtx* ctx = (DimCtx*)arix_malloc(sizeof(DimCtx), 64);
        if (ctx) { ctx->a = a; ctx->dim = dim; var->backward_fn = backward_sum; var->backward_ctx = ctx; var->free_ctx = free_ctx_DimCtx; var->recompute_ctx = recompute_DimCtx; var->params[0] = dim; var->param_count = 1; }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_mean(ArixTape* tape, ArixVariable* a, size_t dim) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_mean(a->data, dim);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        DimCtx* ctx = (DimCtx*)arix_malloc(sizeof(DimCtx), 64);
        if (ctx) { ctx->a = a; ctx->dim = dim; var->backward_fn = backward_mean; var->backward_ctx = ctx; var->free_ctx = free_ctx_DimCtx; var->recompute_ctx = recompute_DimCtx; var->params[0] = dim; var->param_count = 1; }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_transpose(ArixTape* tape, ArixVariable* a, size_t dim1, size_t dim2) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_transpose(a->data, dim1, dim2);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        TransposeCtx* ctx = (TransposeCtx*)arix_malloc(sizeof(TransposeCtx), 64);
        if (ctx) { ctx->a = a; ctx->dim1 = dim1; ctx->dim2 = dim2; var->backward_fn = backward_transpose; var->backward_ctx = ctx; var->free_ctx = free_ctx_TransposeCtx; var->recompute_ctx = recompute_TransposeCtx; var->params[0] = dim1; var->params[1] = dim2; var->param_count = 2; }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_dropout(ArixTape* tape, ArixVariable* a, float rate, unsigned int seed) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_dropout(a->data, rate, seed);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        DropoutCtx* ctx = (DropoutCtx*)arix_malloc(sizeof(DropoutCtx), 64);
        if (ctx) {
            ctx->a = a;
            ctx->rate = rate;
            ctx->mask = arix_tensor_copy(result);
            float* md = (float*)ctx->mask->data;
            for (size_t i = 0; i < ctx->mask->size; i++) md[i] = ((float*)result->data)[i] != 0.0f ? 1.0f : 0.0f;
            var->backward_fn = backward_dropout;
            var->backward_ctx = ctx;
            var->free_ctx = free_ctx_DropoutCtx;
        }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_layer_norm(ArixTape* tape, ArixVariable* a, ArixVariable* gamma, ArixVariable* beta, float eps) {
    int rg = requires_grad1(a) || (gamma && gamma->requires_grad) || (beta && beta->requires_grad);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_layer_norm(a->data, gamma ? gamma->data : NULL, beta ? beta->data : NULL, eps);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        LayerNormCtx* ctx = (LayerNormCtx*)arix_malloc(sizeof(LayerNormCtx), 64);
        if (ctx) { ctx->a = a; ctx->gamma = gamma; ctx->beta = beta; ctx->eps = eps; var->backward_fn = backward_layer_norm; var->backward_ctx = ctx; var->free_ctx = free_ctx_LayerNormCtx; var->recompute_ctx = recompute_LayerNormCtx; *(float*)&var->params[0] = eps; var->param_count = 1; }
        ArixVariable** pars = (ArixVariable**)arix_malloc(3 * sizeof(ArixVariable*), 64);
        if (pars) { size_t np = 0; pars[np++] = a; if (gamma) pars[np++] = gamma; if (beta) pars[np++] = beta; set_parents(var, pars, np); }
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

typedef struct { ArixVariable** vars; size_t num_vars; size_t dim; size_t* splits; } ConcatCtx;
static void free_ctx_ConcatCtx(void* p) {
    ConcatCtx* c = (ConcatCtx*)p;
    if (c->vars) arix_free(c->vars, c->num_vars * sizeof(ArixVariable*));
    if (c->splits) arix_free(c->splits, c->num_vars * sizeof(size_t));
    arix_free(p, sizeof(ConcatCtx));
}
static void* recompute_ConcatCtx(ArixVariable* var, size_t* params, size_t n) {
    size_t nv = var->num_parents;
    size_t dim = n > 0 ? params[0] : 0;
    ConcatCtx* ctx = (ConcatCtx*)arix_malloc(sizeof(ConcatCtx), 64);
    if (ctx) {
        ctx->vars = (ArixVariable**)arix_malloc(nv * sizeof(ArixVariable*), 64);
        ctx->splits = (size_t*)arix_malloc(nv * sizeof(size_t), 64);
        if (ctx->vars && ctx->splits) {
            for (size_t i = 0; i < nv; i++) {
                ctx->vars[i] = var->parents[i];
                ctx->splits[i] = var->parents[i]->data->shape[dim];
            }
        }
        ctx->num_vars = nv;
        ctx->dim = dim;
        var->free_ctx = free_ctx_ConcatCtx;
    }
    return ctx;
}

static void backward_concat(void* ctx, ArixTensor* grad_output) {
    ConcatCtx* c = (ConcatCtx*)ctx;
    if (!c || !grad_output) return;
    size_t element_offset = 0;
    size_t slice_size = grad_output->size / grad_output->shape[c->dim];
    for (size_t i = 0; i < c->num_vars; i++) {
        size_t start = element_offset / slice_size;
        ArixTensor* slice = arix_tensor_slice(grad_output, c->dim, start, start + c->splits[i]);
        if (slice) {
            grad_accum(&c->vars[i]->grad, slice);
        }
        element_offset += c->splits[i] * slice_size;
    }
}

ArixVariable* arix_concat(ArixTape* tape, ArixVariable** vars, size_t num_vars, size_t dim) {
    if (!vars || num_vars == 0) return NULL;
    int rg = 0;
    for (size_t i = 0; i < num_vars; i++) {
        if (!vars[i] || !vars[i]->data) return NULL;
        if (vars[i]->requires_grad) rg = 1;
    }
    const ArixTensor** tensors = (const ArixTensor**)arix_malloc(num_vars * sizeof(ArixTensor*), 64);
    if (!tensors) return NULL;
    for (size_t i = 0; i < num_vars; i++) tensors[i] = vars[i]->data;
    ArixTensor* result = arix_tensor_concat(tensors, num_vars, dim);
    arix_free((void*)tensors, num_vars * sizeof(ArixTensor*));
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        ConcatCtx* ctx = (ConcatCtx*)arix_malloc(sizeof(ConcatCtx), 64);
        if (!ctx) { arix_tensor_destroy(result); return NULL; }
        ctx->vars = (ArixVariable**)arix_malloc(num_vars * sizeof(ArixVariable*), 64);
        ctx->splits = (size_t*)arix_malloc(num_vars * sizeof(size_t), 64);
        if (!ctx->vars || !ctx->splits) {
            arix_free(ctx->vars, num_vars * sizeof(ArixVariable*));
            arix_free(ctx->splits, num_vars * sizeof(size_t));
            arix_free(ctx, sizeof(ConcatCtx));
            arix_tensor_destroy(result); return NULL;
        }
        memcpy(ctx->vars, vars, num_vars * sizeof(ArixVariable*));
        for (size_t i = 0; i < num_vars; i++) ctx->splits[i] = vars[i]->data->shape[dim];
        ctx->num_vars = num_vars;
        ctx->dim = dim;
        var->backward_fn = backward_concat;
        var->backward_ctx = ctx;
        var->free_ctx = free_ctx_ConcatCtx;
        var->recompute_ctx = recompute_ConcatCtx;
        var->params[0] = dim; var->param_count = 1;
        set_parents(var, vars, num_vars);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_conv2d(ArixTape* tape, ArixVariable* input, ArixVariable* kernel, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w) {
    int rg = requires_grad(input, kernel);
    if (!input || !kernel || !input->data || !kernel->data) return NULL;
    ArixTensor* result = arix_tensor_conv2d(input->data, kernel->data, stride_h, stride_w, pad_h, pad_w);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        Conv2DCtx* ctx = (Conv2DCtx*)arix_malloc(sizeof(Conv2DCtx), 64);
        if (ctx) { ctx->input = input; ctx->kernel = kernel; ctx->stride_h = stride_h; ctx->stride_w = stride_w; ctx->pad_h = pad_h; ctx->pad_w = pad_w; var->backward_fn = backward_conv2d; var->backward_ctx = ctx; var->free_ctx = free_ctx_Conv2DCtx; var->recompute_ctx = recompute_Conv2DCtx; var->params[0] = stride_h; var->params[1] = stride_w; var->params[2] = pad_h; var->params[3] = pad_w; var->param_count = 4; }
        ArixVariable* pars[2]; pars[0] = input; pars[1] = kernel;
        set_parents(var, pars, 2);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

/* ===== sqrt ===== */
static void backward_sqrt(void* ctx, ArixTensor* grad_output) {
    UnaryCtx* c = (UnaryCtx*)ctx;
    if (!c->a->requires_grad) return;
    float* xd = (float*)c->a->data->data;
    float* go = (float*)grad_output->data;
    size_t n = c->a->data->size;
    ArixTensor* g = arix_tensor_zeros(c->a->data->shape, c->a->data->ndim, ARIX_FLOAT32);
    float* gd = (float*)g->data;
    for (size_t i = 0; i < n; i++) {
        float sx = sqrtf(fabsf(xd[i])) + 1e-8f;
        gd[i] = go[i] * (0.5f / sx);
    }
    grad_accum(&c->a->grad, g);
}

ArixVariable* arix_sqrt(ArixTape* tape, ArixVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_sqrt(a->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        UnaryCtx* ctx = (UnaryCtx*)arix_malloc(sizeof(UnaryCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_sqrt; var->backward_ctx = ctx; var->free_ctx = free_ctx_UnaryCtx; var->recompute_ctx = recompute_UnaryCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

/* ===== abs ===== */
static void backward_abs(void* ctx, ArixTensor* grad_output) {
    UnaryCtx* c = (UnaryCtx*)ctx;
    if (!c->a->requires_grad) return;
    float* xd = (float*)c->a->data->data;
    float* go = (float*)grad_output->data;
    size_t n = c->a->data->size;
    ArixTensor* g = arix_tensor_zeros(c->a->data->shape, c->a->data->ndim, ARIX_FLOAT32);
    float* gd = (float*)g->data;
    for (size_t i = 0; i < n; i++) {
        gd[i] = go[i] * (xd[i] > 0.0f ? 1.0f : (xd[i] < 0.0f ? -1.0f : 0.0f));
    }
    grad_accum(&c->a->grad, g);
}

ArixVariable* arix_abs(ArixTape* tape, ArixVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_abs(a->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        UnaryCtx* ctx = (UnaryCtx*)arix_malloc(sizeof(UnaryCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_abs; var->backward_ctx = ctx; var->free_ctx = free_ctx_UnaryCtx; var->recompute_ctx = recompute_UnaryCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

/* ===== sin ===== */
static void backward_sin(void* ctx, ArixTensor* grad_output) {
    UnaryCtx* c = (UnaryCtx*)ctx;
    if (!c->a->requires_grad) return;
    float* xd = (float*)c->a->data->data;
    float* go = (float*)grad_output->data;
    size_t n = c->a->data->size;
    ArixTensor* g = arix_tensor_zeros(c->a->data->shape, c->a->data->ndim, ARIX_FLOAT32);
    float* gd = (float*)g->data;
    for (size_t i = 0; i < n; i++) gd[i] = go[i] * cosf(xd[i]);
    grad_accum(&c->a->grad, g);
}

ArixVariable* arix_sin(ArixTape* tape, ArixVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_sin(a->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        UnaryCtx* ctx = (UnaryCtx*)arix_malloc(sizeof(UnaryCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_sin; var->backward_ctx = ctx; var->free_ctx = free_ctx_UnaryCtx; var->recompute_ctx = recompute_UnaryCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

/* ===== cos ===== */
static void backward_cos(void* ctx, ArixTensor* grad_output) {
    UnaryCtx* c = (UnaryCtx*)ctx;
    if (!c->a->requires_grad) return;
    float* xd = (float*)c->a->data->data;
    float* go = (float*)grad_output->data;
    size_t n = c->a->data->size;
    ArixTensor* g = arix_tensor_zeros(c->a->data->shape, c->a->data->ndim, ARIX_FLOAT32);
    float* gd = (float*)g->data;
    for (size_t i = 0; i < n; i++) gd[i] = -go[i] * sinf(xd[i]);
    grad_accum(&c->a->grad, g);
}

ArixVariable* arix_cos(ArixTape* tape, ArixVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_cos(a->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        UnaryCtx* ctx = (UnaryCtx*)arix_malloc(sizeof(UnaryCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_cos; var->backward_ctx = ctx; var->free_ctx = free_ctx_UnaryCtx; var->recompute_ctx = recompute_UnaryCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

/* ===== tan ===== */
static void backward_tan(void* ctx, ArixTensor* grad_output) {
    UnaryCtx* c = (UnaryCtx*)ctx;
    if (!c->a->requires_grad) return;
    float* xd = (float*)c->a->data->data;
    float* go = (float*)grad_output->data;
    size_t n = c->a->data->size;
    ArixTensor* g = arix_tensor_zeros(c->a->data->shape, c->a->data->ndim, ARIX_FLOAT32);
    float* gd = (float*)g->data;
    for (size_t i = 0; i < n; i++) {
        float t = tanf(xd[i]);
        gd[i] = go[i] * (1.0f + t * t);
    }
    grad_accum(&c->a->grad, g);
}

ArixVariable* arix_tan(ArixTape* tape, ArixVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_tan(a->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        UnaryCtx* ctx = (UnaryCtx*)arix_malloc(sizeof(UnaryCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_tan; var->backward_ctx = ctx; var->free_ctx = free_ctx_UnaryCtx; var->recompute_ctx = recompute_UnaryCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

/* ===== asin ===== */
static void backward_asin(void* ctx, ArixTensor* grad_output) {
    UnaryCtx* c = (UnaryCtx*)ctx;
    if (!c->a->requires_grad) return;
    float* xd = (float*)c->a->data->data;
    float* go = (float*)grad_output->data;
    size_t n = c->a->data->size;
    ArixTensor* g = arix_tensor_zeros(c->a->data->shape, c->a->data->ndim, ARIX_FLOAT32);
    float* gd = (float*)g->data;
    for (size_t i = 0; i < n; i++) {
        float denom = sqrtf(1.0f - xd[i] * xd[i]) + 1e-8f;
        gd[i] = go[i] / denom;
    }
    grad_accum(&c->a->grad, g);
}

ArixVariable* arix_asin(ArixTape* tape, ArixVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_asin(a->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        UnaryCtx* ctx = (UnaryCtx*)arix_malloc(sizeof(UnaryCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_asin; var->backward_ctx = ctx; var->free_ctx = free_ctx_UnaryCtx; var->recompute_ctx = recompute_UnaryCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

/* ===== acos ===== */
static void backward_acos(void* ctx, ArixTensor* grad_output) {
    UnaryCtx* c = (UnaryCtx*)ctx;
    if (!c->a->requires_grad) return;
    float* xd = (float*)c->a->data->data;
    float* go = (float*)grad_output->data;
    size_t n = c->a->data->size;
    ArixTensor* g = arix_tensor_zeros(c->a->data->shape, c->a->data->ndim, ARIX_FLOAT32);
    float* gd = (float*)g->data;
    for (size_t i = 0; i < n; i++) {
        float denom = sqrtf(1.0f - xd[i] * xd[i]) + 1e-8f;
        gd[i] = -go[i] / denom;
    }
    grad_accum(&c->a->grad, g);
}

ArixVariable* arix_acos(ArixTape* tape, ArixVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_acos(a->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        UnaryCtx* ctx = (UnaryCtx*)arix_malloc(sizeof(UnaryCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_acos; var->backward_ctx = ctx; var->free_ctx = free_ctx_UnaryCtx; var->recompute_ctx = recompute_UnaryCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

/* ===== atan ===== */
static void backward_atan(void* ctx, ArixTensor* grad_output) {
    UnaryCtx* c = (UnaryCtx*)ctx;
    if (!c->a->requires_grad) return;
    float* xd = (float*)c->a->data->data;
    float* go = (float*)grad_output->data;
    size_t n = c->a->data->size;
    ArixTensor* g = arix_tensor_zeros(c->a->data->shape, c->a->data->ndim, ARIX_FLOAT32);
    float* gd = (float*)g->data;
    for (size_t i = 0; i < n; i++) gd[i] = go[i] / (1.0f + xd[i] * xd[i]);
    grad_accum(&c->a->grad, g);
}

ArixVariable* arix_atan(ArixTape* tape, ArixVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_atan(a->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        UnaryCtx* ctx = (UnaryCtx*)arix_malloc(sizeof(UnaryCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_atan; var->backward_ctx = ctx; var->free_ctx = free_ctx_UnaryCtx; var->recompute_ctx = recompute_UnaryCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

/* ===== sinh ===== */
static void backward_sinh(void* ctx, ArixTensor* grad_output) {
    UnaryCtx* c = (UnaryCtx*)ctx;
    if (!c->a->requires_grad) return;
    float* xd = (float*)c->a->data->data;
    float* go = (float*)grad_output->data;
    size_t n = c->a->data->size;
    ArixTensor* g = arix_tensor_zeros(c->a->data->shape, c->a->data->ndim, ARIX_FLOAT32);
    float* gd = (float*)g->data;
    for (size_t i = 0; i < n; i++) gd[i] = go[i] * coshf(xd[i]);
    grad_accum(&c->a->grad, g);
}

ArixVariable* arix_sinh(ArixTape* tape, ArixVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_sinh(a->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        UnaryCtx* ctx = (UnaryCtx*)arix_malloc(sizeof(UnaryCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_sinh; var->backward_ctx = ctx; var->free_ctx = free_ctx_UnaryCtx; var->recompute_ctx = recompute_UnaryCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

/* ===== cosh ===== */
static void backward_cosh(void* ctx, ArixTensor* grad_output) {
    UnaryCtx* c = (UnaryCtx*)ctx;
    if (!c->a->requires_grad) return;
    float* xd = (float*)c->a->data->data;
    float* go = (float*)grad_output->data;
    size_t n = c->a->data->size;
    ArixTensor* g = arix_tensor_zeros(c->a->data->shape, c->a->data->ndim, ARIX_FLOAT32);
    float* gd = (float*)g->data;
    for (size_t i = 0; i < n; i++) gd[i] = go[i] * sinhf(xd[i]);
    grad_accum(&c->a->grad, g);
}

ArixVariable* arix_cosh(ArixTape* tape, ArixVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_cosh(a->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        UnaryCtx* ctx = (UnaryCtx*)arix_malloc(sizeof(UnaryCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_cosh; var->backward_ctx = ctx; var->free_ctx = free_ctx_UnaryCtx; var->recompute_ctx = recompute_UnaryCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

/* ===== var ===== */
static void backward_var(void* ctx, ArixTensor* grad_output) {
    DimCtx* c = (DimCtx*)ctx;
    if (!c->a->requires_grad) return;
    float* xd = (float*)c->a->data->data;
    float* go = (float*)grad_output->data;
    size_t outer = 1, inner = 1;
    for (size_t i = 0; i < c->dim; i++) outer *= c->a->data->shape[i];
    size_t dim_size = c->a->data->shape[c->dim];
    for (size_t i = c->dim + 1; i < c->a->data->ndim; i++) inner *= c->a->data->shape[i];
    float n_1 = (float)(dim_size > 1 ? dim_size - 1 : 1);
    ArixTensor* g = arix_tensor_zeros(c->a->data->shape, c->a->data->ndim, ARIX_FLOAT32);
    float* gd = (float*)g->data;
    for (size_t o = 0; o < outer; o++) {
        for (size_t i = 0; i < inner; i++) {
            float mean = 0.0f;
            for (size_t d = 0; d < dim_size; d++) mean += xd[o * dim_size * inner + d * inner + i];
            mean /= (float)dim_size;
            float gv = go[o * inner + i];
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = o * dim_size * inner + d * inner + i;
                gd[idx] = 2.0f * (xd[idx] - mean) * gv / n_1;
            }
        }
    }
    grad_accum(&c->a->grad, g);
}

ArixVariable* arix_var(ArixTape* tape, ArixVariable* a, size_t dim) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_var(a->data, dim);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        DimCtx* ctx = (DimCtx*)arix_malloc(sizeof(DimCtx), 64);
        if (ctx) { ctx->a = a; ctx->dim = dim; var->backward_fn = backward_var; var->backward_ctx = ctx; var->free_ctx = free_ctx_DimCtx; var->recompute_ctx = recompute_DimCtx; var->params[0] = dim; var->param_count = 1; }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

/* ===== std ===== */
static void backward_std(void* ctx, ArixTensor* grad_output) {
    DimCtx* c = (DimCtx*)ctx;
    if (!c->a->requires_grad) return;
    float* xd = (float*)c->a->data->data;
    float* go = (float*)grad_output->data;
    size_t outer = 1, inner = 1;
    for (size_t i = 0; i < c->dim; i++) outer *= c->a->data->shape[i];
    size_t dim_size = c->a->data->shape[c->dim];
    for (size_t i = c->dim + 1; i < c->a->data->ndim; i++) inner *= c->a->data->shape[i];
    float n_1 = (float)(dim_size > 1 ? dim_size - 1 : 1);
    float eps = 1e-8f;
    ArixTensor* g = arix_tensor_zeros(c->a->data->shape, c->a->data->ndim, ARIX_FLOAT32);
    float* gd = (float*)g->data;
    for (size_t o = 0; o < outer; o++) {
        for (size_t i = 0; i < inner; i++) {
            float mean = 0.0f;
            for (size_t d = 0; d < dim_size; d++) mean += xd[o * dim_size * inner + d * inner + i];
            mean /= (float)dim_size;
            float var = 0.0f;
            for (size_t d = 0; d < dim_size; d++) {
                float diff = xd[o * dim_size * inner + d * inner + i] - mean;
                var += diff * diff;
            }
            var /= (float)dim_size;
            float std_val = sqrtf(var + eps);
            float gv = go[o * inner + i];
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = o * dim_size * inner + d * inner + i;
                gd[idx] = (xd[idx] - mean) * gv / (n_1 * std_val);
            }
        }
    }
    grad_accum(&c->a->grad, g);
}

ArixVariable* arix_std(ArixTape* tape, ArixVariable* a, size_t dim) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_std(a->data, dim);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        DimCtx* ctx = (DimCtx*)arix_malloc(sizeof(DimCtx), 64);
        if (ctx) { ctx->a = a; ctx->dim = dim; var->backward_fn = backward_std; var->backward_ctx = ctx; var->free_ctx = free_ctx_DimCtx; var->recompute_ctx = recompute_DimCtx; var->params[0] = dim; var->param_count = 1; }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

/* ===== cross_entropy ===== */
typedef struct { ArixVariable* pred; ArixVariable* target; size_t N; size_t num_classes; } CrossEntropyCtx;
static void free_ctx_CrossEntropyCtx(void* p) { arix_free(p, sizeof(CrossEntropyCtx)); }

static void backward_cross_entropy(void* ctx, ArixTensor* grad_output) {
    CrossEntropyCtx* c = (CrossEntropyCtx*)ctx;
    (void)grad_output;
    if (c->pred->requires_grad) {
        ArixTensor* sm = arix_tensor_softmax(c->pred->data, 1);
        float* sd = (float*)sm->data;
        float* td = (float*)c->target->data;
        size_t n = c->target->data->size;
        size_t nc = c->num_classes;
        float scale = 1.0f / (float)c->N;
        ArixTensor* g = arix_tensor_zeros(c->pred->data->shape, c->pred->data->ndim, ARIX_FLOAT32);
        float* gd = (float*)g->data;
        for (size_t i = 0; i < n; i++) {
            int t = (int)td[i];
            for (size_t j = 0; j < nc; j++) {
                float indicator = (j == (size_t)t) ? 1.0f : 0.0f;
                gd[i * nc + j] = (sd[i * nc + j] - indicator) * scale;
            }
        }
        arix_tensor_destroy(sm);
        grad_accum(&c->pred->grad, g);
    }
}

ArixVariable* arix_cross_entropy(ArixTape* tape, ArixVariable* pred, ArixVariable* target) {
    if (!pred || !target || !pred->data || !target->data) return NULL;
    ArixTensor* result = arix_tensor_cross_entropy(pred->data, target->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, 0);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    int rg = requires_grad(pred, target);
    if (rg && g_grad_enabled) {
        CrossEntropyCtx* ctx = (CrossEntropyCtx*)arix_malloc(sizeof(CrossEntropyCtx), 64);
        if (ctx) {
            ctx->pred = pred; ctx->target = target;
            ctx->N = pred->data->size / pred->data->shape[1];
            ctx->num_classes = pred->data->shape[1];
            var->backward_fn = backward_cross_entropy; var->backward_ctx = ctx; var->free_ctx = free_ctx_CrossEntropyCtx;
        }
        set_parents(var, &pred, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

/* ===== nll_loss ===== */
typedef struct { ArixVariable* pred; ArixVariable* target; size_t N; } NLLCtx;
static void free_ctx_NLLCtx(void* p) { arix_free(p, sizeof(NLLCtx)); }

static void backward_nll(void* ctx, ArixTensor* grad_output) {
    NLLCtx* c = (NLLCtx*)ctx;
    (void)grad_output;
    if (c->pred->requires_grad) {
        float* pd = (float*)c->pred->data->data;
        float* td = (float*)c->target->data;
        size_t n = c->target->data->size;
        size_t nc = c->pred->data->size / n;
        ArixTensor* g = arix_tensor_zeros(c->pred->data->shape, c->pred->data->ndim, ARIX_FLOAT32);
        float* gd = (float*)g->data;
        for (size_t i = 0; i < n; i++) {
            int t = (int)td[i];
            size_t idx = i * nc + (size_t)t;
            gd[idx] = -1.0f / (pd[idx] + 1e-8f);
        }
        grad_accum(&c->pred->grad, g);
    }
}

ArixVariable* arix_nll_loss(ArixTape* tape, ArixVariable* pred, ArixVariable* target) {
    if (!pred || !target || !pred->data || !target->data) return NULL;
    ArixTensor* result = arix_tensor_nll_loss(pred->data, target->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, 0);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    int rg = requires_grad(pred, target);
    if (rg && g_grad_enabled) {
        NLLCtx* ctx = (NLLCtx*)arix_malloc(sizeof(NLLCtx), 64);
        if (ctx) {
            ctx->pred = pred; ctx->target = target; ctx->N = target->data->size;
            var->backward_fn = backward_nll; var->backward_ctx = ctx; var->free_ctx = free_ctx_NLLCtx;
        }
        set_parents(var, &pred, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

/* ===== bce_loss ===== */
typedef struct { ArixVariable* pred; ArixVariable* target; } BCECtx;
static void free_ctx_BCECtx(void* p) { arix_free(p, sizeof(BCECtx)); }

static void backward_bce(void* ctx, ArixTensor* grad_output) {
    BCECtx* c = (BCECtx*)ctx;
    if (c->pred->requires_grad) {
        float* pd = (float*)c->pred->data->data;
        float* td = (float*)c->target->data->data;
        float* go = (float*)grad_output->data;
        size_t n = c->pred->data->size;
        ArixTensor* g = arix_tensor_zeros(c->pred->data->shape, c->pred->data->ndim, ARIX_FLOAT32);
        float* gd = (float*)g->data;
        for (size_t i = 0; i < n; i++) {
            float p = pd[i];
            float denom = p * (1.0f - p) + 1e-8f;
            gd[i] = go[i] * (p - td[i]) / denom;
        }
        grad_accum(&c->pred->grad, g);
    }
}

ArixVariable* arix_bce_loss(ArixTape* tape, ArixVariable* pred, ArixVariable* target) {
    if (!pred || !target || !pred->data || !target->data) return NULL;
    ArixTensor* result = arix_tensor_binary_cross_entropy(pred->data, target->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, 0);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    int rg = requires_grad(pred, target);
    if (rg && g_grad_enabled) {
        BCECtx* ctx = (BCECtx*)arix_malloc(sizeof(BCECtx), 64);
        if (ctx) { ctx->pred = pred; ctx->target = target; var->backward_fn = backward_bce; var->backward_ctx = ctx; var->free_ctx = free_ctx_BCECtx; }
        set_parents(var, &pred, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

/* ===== embedding ===== */
typedef struct { ArixVariable* weight; ArixVariable* indices; } EmbeddingCtx;
static void free_ctx_EmbeddingCtx(void* p) { arix_free(p, sizeof(EmbeddingCtx)); }
static void* recompute_EmbeddingCtx(ArixVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    EmbeddingCtx* ctx = (EmbeddingCtx*)arix_malloc(sizeof(EmbeddingCtx), 64);
    if (ctx) { ctx->weight = var->parents[0]; ctx->indices = var->parents[1]; var->free_ctx = free_ctx_EmbeddingCtx; }
    return ctx;
}

static void backward_embedding(void* ctx, ArixTensor* grad_output) {
    EmbeddingCtx* c = (EmbeddingCtx*)ctx;
    if (c->weight->requires_grad) {
        size_t vocab_size = c->weight->data->shape[0];
        size_t embed_dim = c->weight->data->shape[1];
        float* go = (float*)grad_output->data;
        size_t* idx = (size_t*)c->indices->data->data;
        size_t num_indices = c->indices->data->size;
        ArixTensor* g = arix_tensor_zeros(c->weight->data->shape, c->weight->data->ndim, ARIX_FLOAT32);
        float* gd = (float*)g->data;
        for (size_t i = 0; i < num_indices; i++) {
            size_t ix = idx[i];
            if (ix >= 0 && (size_t)ix < vocab_size) {
                for (size_t d = 0; d < embed_dim; d++)
                    gd[(size_t)ix * embed_dim + d] += go[i * embed_dim + d];
            }
        }
        grad_accum(&c->weight->grad, g);
    }
}

ArixVariable* arix_embedding(ArixTape* tape, ArixVariable* weight, ArixVariable* indices) {
    if (!weight || !indices || !weight->data || !indices->data) return NULL;
    ArixTensor* result = arix_tensor_embedding(weight->data, indices->data);
    if (!result) return NULL;
    int rg = requires_grad(weight, NULL);
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        EmbeddingCtx* ctx = (EmbeddingCtx*)arix_malloc(sizeof(EmbeddingCtx), 64);
        if (ctx) { ctx->weight = weight; ctx->indices = indices; var->backward_fn = backward_embedding; var->backward_ctx = ctx; var->free_ctx = free_ctx_EmbeddingCtx; var->recompute_ctx = recompute_EmbeddingCtx; }
        set_parents(var, &weight, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

/* ===== log_softmax ===== */
static void backward_log_softmax(void* ctx, ArixTensor* grad_output) {
    DimCtx* c = (DimCtx*)ctx;
    if (!c->a->requires_grad) return;
    ArixTensor* sm = arix_tensor_softmax(c->a->data, c->dim);
    float* sd = (float*)sm->data;
    float* go = (float*)grad_output->data;
    size_t outer = 1, inner = 1;
    for (size_t i = 0; i < c->dim; i++) outer *= c->a->data->shape[i];
    size_t dim_size = c->a->data->shape[c->dim];
    for (size_t i = c->dim + 1; i < c->a->data->ndim; i++) inner *= c->a->data->shape[i];
    ArixTensor* g = arix_tensor_zeros(c->a->data->shape, c->a->data->ndim, ARIX_FLOAT32);
    float* gd = (float*)g->data;
    for (size_t o = 0; o < outer; o++) {
        for (size_t i = 0; i < inner; i++) {
            float sum_grad = 0.0f;
            for (size_t d = 0; d < dim_size; d++)
                sum_grad += go[o * dim_size * inner + d * inner + i];
            for (size_t d = 0; d < dim_size; d++) {
                size_t idx = o * dim_size * inner + d * inner + i;
                gd[idx] = go[idx] - sd[idx] * sum_grad;
            }
        }
    }
    grad_accum(&c->a->grad, g);
    arix_tensor_destroy(sm);
}

ArixVariable* arix_log_softmax(ArixTape* tape, ArixVariable* a, size_t dim) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_log_softmax(a->data, dim);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        DimCtx* ctx = (DimCtx*)arix_malloc(sizeof(DimCtx), 64);
        if (ctx) { ctx->a = a; ctx->dim = dim; var->backward_fn = backward_log_softmax; var->backward_ctx = ctx; var->free_ctx = free_ctx_DimCtx; var->recompute_ctx = recompute_DimCtx; var->params[0] = dim; var->param_count = 1; }
        set_parents(var, &a, 1);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

/* ===== sign (zero grad) ===== */
static void backward_zero(void* ctx, ArixTensor* grad_output) {
    (void)ctx; (void)grad_output;
}

static ArixVariable* op_zero_grad(ArixTape* tape, ArixVariable* a,
                                   ArixTensor* (*tensor_fn)(const ArixTensor*)) {
    if (!a || !a->data) return NULL;
    ArixTensor* result = tensor_fn(a->data);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, 0);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}

ArixVariable* arix_sign(ArixTape* tape, ArixVariable* a) { return op_zero_grad(tape, a, arix_tensor_sign); }
ArixVariable* arix_floor(ArixTape* tape, ArixVariable* a) { return op_zero_grad(tape, a, arix_tensor_floor); }
ArixVariable* arix_ceil(ArixTape* tape, ArixVariable* a) { return op_zero_grad(tape, a, arix_tensor_ceil); }
ArixVariable* arix_round(ArixTape* tape, ArixVariable* a) { return op_zero_grad(tape, a, arix_tensor_round); }
ArixVariable* arix_trunc(ArixTape* tape, ArixVariable* a) { return op_zero_grad(tape, a, arix_tensor_trunc); }

/* ===== batch_norm ===== */
typedef struct { ArixVariable *a, *gamma, *beta, *running_mean, *running_var; float eps; size_t C; } BatchNormCtx;
static void free_ctx_BatchNormCtx(void* p) { arix_free(p, sizeof(BatchNormCtx)); }
static void* recompute_BatchNormCtx(ArixVariable* var, size_t* params, size_t n) {
    BatchNormCtx* ctx = (BatchNormCtx*)arix_malloc(sizeof(BatchNormCtx), 64);
    if (ctx) {
        ctx->a = var->parents[0];
        ctx->gamma = var->num_parents > 1 ? var->parents[1] : NULL;
        ctx->beta = var->num_parents > 2 ? var->parents[2] : NULL;
        ctx->running_mean = NULL; ctx->running_var = NULL;
        ctx->eps = n > 0 ? *(float*)&params[0] : 1e-5f;
        ctx->C = n > 1 ? params[1] : 0;
        var->free_ctx = free_ctx_BatchNormCtx;
    }
    return ctx;
}

static void backward_batch_norm(void* ctx, ArixTensor* grad_output) {
    BatchNormCtx* c = (BatchNormCtx*)ctx;
    float* xd = (float*)c->a->data->data;
    float* gd = (float*)grad_output->data;
    size_t N = c->a->data->shape[0], C = c->C;
    size_t H = 1, W = 1;
    size_t ndim = c->a->data->ndim;
    if (ndim == 4) { H = c->a->data->shape[2]; W = c->a->data->shape[3]; }
    else if (ndim == 3) { H = c->a->data->shape[2]; W = 1; }
    else if (ndim == 2) { H = 1; W = 1; }
    size_t spatial = H * W;
    size_t n = N * spatial;
    float eps = c->eps;
    size_t total = c->a->data->size;

    if (c->a->requires_grad) {
        ArixTensor* gx = arix_tensor_zeros(c->a->data->shape, ndim, ARIX_FLOAT32);
        float* gxd = (float*)gx->data;
        for (size_t c_ = 0; c_ < C; c_++) {
            float mean = 0, var = 0;
            for (size_t i = 0; i < n; i++) mean += xd[i * C + c_];
            mean /= (float)n;
            for (size_t i = 0; i < n; i++) { float d = xd[i * C + c_] - mean; var += d * d; }
            var /= (float)n;
            float inv_std = 1.0f / sqrtf(var + eps);
            float gamma_val = c->gamma && c->gamma->data ? ((float*)c->gamma->data->data)[c_] : 1.0f;
            float dnorm_sum = 0, dnorm_dot = 0;
            for (size_t i = 0; i < n; i++) {
                size_t idx = i * C + c_;
                float x_hat = (xd[idx] - mean) * inv_std;
                float dnorm = gd[idx] * gamma_val;
                dnorm_sum += dnorm;
                dnorm_dot += dnorm * x_hat;
            }
            for (size_t i = 0; i < n; i++) {
                size_t idx = i * C + c_;
                float x_hat = (xd[idx] - mean) * inv_std;
                float dnorm = gd[idx] * gamma_val;
                gxd[idx] = (dnorm - dnorm_sum / (float)n - x_hat * dnorm_dot / (float)n) * inv_std;
            }
        }
        grad_accum(&c->a->grad, gx);
    }
    if (c->gamma && c->gamma->requires_grad) {
        ArixTensor* gg = arix_tensor_zeros(c->gamma->data->shape, c->gamma->data->ndim, ARIX_FLOAT32);
        float* ggd = (float*)gg->data;
        for (size_t c_ = 0; c_ < C; c_++) {
            float mean = 0;
            for (size_t i = 0; i < n; i++) mean += xd[i * C + c_];
            mean /= (float)n;
            float var = 0;
            for (size_t i = 0; i < n; i++) { float d = xd[i * C + c_] - mean; var += d * d; }
            var /= (float)n;
            float inv_std = 1.0f / sqrtf(var + eps);
            for (size_t i = 0; i < n; i++) {
                size_t idx = i * C + c_;
                float x_hat = (xd[idx] - mean) * inv_std;
                ggd[c_] += gd[idx] * x_hat;
            }
        }
        grad_accum(&c->gamma->grad, gg);
    }
    if (c->beta && c->beta->requires_grad) {
        ArixTensor* gb = arix_tensor_zeros(c->beta->data->shape, c->beta->data->ndim, ARIX_FLOAT32);
        float* gbd = (float*)gb->data;
        for (size_t c_ = 0; c_ < C; c_++)
            for (size_t i = 0; i < n; i++)
                gbd[c_] += gd[i * C + c_];
        grad_accum(&c->beta->grad, gb);
    }
}

ArixVariable* arix_batch_norm(ArixTape* tape, ArixVariable* a, ArixVariable* gamma, ArixVariable* beta, ArixVariable* running_mean, ArixVariable* running_var, float eps) {
    int rg = requires_grad1(a) || (gamma && gamma->requires_grad) || (beta && beta->requires_grad);
    if (!a || !a->data) return NULL;
    ArixTensor* result = arix_tensor_batch_norm(a->data, gamma ? gamma->data : NULL, beta ? beta->data : NULL, running_mean ? running_mean->data : NULL, running_var ? running_var->data : NULL, eps);
    if (!result) return NULL;
    ArixVariable* var = arix_variable_create(result, rg);
    if (!var) { arix_tensor_destroy(result); return NULL; }
    if (rg && g_grad_enabled) {
        BatchNormCtx* ctx = (BatchNormCtx*)arix_malloc(sizeof(BatchNormCtx), 64);
        if (ctx) {
            ctx->a = a; ctx->gamma = gamma; ctx->beta = beta;
            ctx->running_mean = running_mean; ctx->running_var = running_var;
            ctx->eps = eps; ctx->C = a->data->shape[1];
            var->backward_fn = backward_batch_norm; var->backward_ctx = ctx;
            var->free_ctx = free_ctx_BatchNormCtx;
            var->recompute_ctx = recompute_BatchNormCtx;
            *(float*)&var->params[0] = eps; var->params[1] = a->data->shape[1]; var->param_count = 2;
        }
        size_t np = 1;
        ArixVariable* pars[5]; pars[0] = a;
        if (gamma) pars[np++] = gamma;
        if (beta) pars[np++] = beta;
        set_parents(var, pars, np);
    }
    if (tape && g_grad_enabled) arix_tape_record(tape, var);
    return var;
}
