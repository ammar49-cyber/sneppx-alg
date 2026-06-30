#include "automatic_differentiation_framework.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_passed = 0, tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } \
} while(0)

#define FLOAT_CLOSE(a,b) (fabsf((a)-(b)) < 5e-2f)

static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout);
    fn(); printf("PASS\n"); tests_passed++;
}

static void test_grad_add(void) {
    size_t sh[] = {3};
    float ad[] = {1.0f, 2.0f, 3.0f};
    float bd[] = {4.0f, 5.0f, 6.0f};
    ArixTensor* ta = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    memcpy(ta->data, ad, 12); memcpy(tb->data, bd, 12);
    ArixVariable *a = arix_variable_create(ta, 1), *b = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* c = arix_add(tape, a, b);
    arix_tape_backward(tape, c);
    for (size_t i = 0; i < 3; i++) {
        ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[i], 1.0f), "add da = 1");
        ASSERT(FLOAT_CLOSE(((float*)b->grad->data)[i], 1.0f), "add db = 1");
    }
    arix_tape_destroy(tape);
}

static void test_grad_mul(void) {
    size_t sh[] = {2};
    float ad[] = {3.0f, 4.0f};
    float bd[] = {5.0f, 6.0f};
    ArixTensor* ta = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    memcpy(ta->data, ad, 8); memcpy(tb->data, bd, 8);
    ArixVariable *a = arix_variable_create(ta, 1), *b = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* c = arix_mul(tape, a, b);
    arix_tape_backward(tape, c);
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[0], 5.0f) && FLOAT_CLOSE(((float*)a->grad->data)[1], 6.0f), "mul da = b");
    ASSERT(FLOAT_CLOSE(((float*)b->grad->data)[0], 3.0f) && FLOAT_CLOSE(((float*)b->grad->data)[1], 4.0f), "mul db = a");
    arix_tape_destroy(tape);
}

static void test_grad_matmul(void) {
    size_t sha[] = {2,3}, shb[] = {3,2};
    ArixTensor* ta = arix_tensor_zeros(sha, 2, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_zeros(shb, 2, ARIX_FLOAT32);
    for (size_t i = 0; i < 6; i++) { ((float*)ta->data)[i] = (float)(i+1); ((float*)tb->data)[i] = (float)(i+7); }
    ArixVariable *a = arix_variable_create(ta, 1), *b = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* c = arix_matmul(tape, a, b);
    arix_tape_backward(tape, c);
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[0], 15), "matmul da[0]=15");
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[5], 23), "matmul da[5]=23");
    ASSERT(FLOAT_CLOSE(((float*)b->grad->data)[0], 5), "matmul db[0]=5");
    ASSERT(FLOAT_CLOSE(((float*)b->grad->data)[5], 9), "matmul db[5]=9");
    arix_tape_destroy(tape);
}

static void test_grad_div(void) {
    size_t sh[] = {2};
    float ad[] = {10.0f, 20.0f}, bd[] = {2.0f, 5.0f};
    ArixTensor* ta = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    memcpy(ta->data, ad, 8); memcpy(tb->data, bd, 8);
    ArixVariable *a = arix_variable_create(ta, 1), *b = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* c = arix_div(tape, a, b);
    arix_tape_backward(tape, c);
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[0], 0.5f), "div da = 1/b");
    ASSERT(FLOAT_CLOSE(((float*)b->grad->data)[0], -2.5f), "div db = -a/b^2");
    arix_tape_destroy(tape);
}

static void test_grad_relu(void) {
    size_t sh[] = {3};
    float ad[] = {-2.0f, 0.0f, 5.0f};
    ArixTensor* ta = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    memcpy(ta->data, ad, 12);
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* r = arix_relu(tape, a);
    arix_tape_backward(tape, r);
    float* ag = (float*)a->grad->data;
    ASSERT(FLOAT_CLOSE(ag[0], 0.0f) && FLOAT_CLOSE(ag[1], 0.0f) && FLOAT_CLOSE(ag[2], 1.0f), "relu grad");
    arix_tape_destroy(tape);
}

static void test_grad_sigmoid(void) {
    size_t sh[] = {1};
    ArixTensor* ta = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    ((float*)ta->data)[0] = 0.0f;
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* s = arix_sigmoid(tape, a);
    arix_tape_backward(tape, s);
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[0], 0.25f), "sigmoid grad at 0 = 0.25");
    arix_tape_destroy(tape);
}

static void test_grad_tanh(void) {
    size_t sh[] = {1};
    ArixTensor* ta = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    ((float*)ta->data)[0] = 0.0f;
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* h = arix_tanh(tape, a);
    arix_tape_backward(tape, h);
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[0], 1.0f), "tanh grad at 0 = 1");
    arix_tape_destroy(tape);
}

static void test_grad_exp_log(void) {
    size_t sh[] = {2};
    float ad[] = {0.0f, 1.0f};
    ArixTensor* ta = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    memcpy(ta->data, ad, 8);
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* e = arix_exp(tape, a);
    arix_tape_backward(tape, e);
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[0], 1.0f) && FLOAT_CLOSE(((float*)a->grad->data)[1], 2.71828f), "exp grad");
    arix_tape_destroy(tape);
}

static void test_grad_pow(void) {
    size_t sh[] = {2};
    float ad[] = {2.0f, 3.0f};
    ArixTensor* ta = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    memcpy(ta->data, ad, 8);
    ArixTensor* tb = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    ((float*)tb->data)[0] = 2.0f; ((float*)tb->data)[1] = 3.0f;
    ArixVariable *a = arix_variable_create(ta, 1), *b = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* p = arix_pow(tape, a, b);
    arix_tape_backward(tape, p);
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[0], 4.0f), "pow da = b*a^(b-1): 4");
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[1], 27.0f), "pow da: 3*3^2=27");
    arix_tape_destroy(tape);
}

static void test_grad_sin_cos(void) {
    size_t sh[] = {2};
    float ad[] = {0.0f, 1.570796f};
    ArixTensor* ta = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    memcpy(ta->data, ad, 8);
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* s = arix_sin(tape, a);
    arix_tape_backward(tape, s);
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[0], 1.0f), "sin grad at 0 = cos(0) = 1");
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[1], 0.0f), "sin grad at pi/2 = cos(pi/2) = 0");
    arix_tape_destroy(tape);
}

static void test_grad_chain_shared_input(void) {
    size_t sh[] = {2};
    float ad[] = {2.0f, 3.0f};
    ArixTensor* ta = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    memcpy(ta->data, ad, 8);
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixVariable* b = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* c = arix_add(tape, a, b);
    ArixVariable* d = arix_mul(tape, c, a);
    ArixVariable* loss = arix_sum(tape, d, 0);
    arix_tape_backward(tape, loss);
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[0], 4.0f), "chain shared a[0]=4");
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[1], 6.0f), "chain shared a[1]=6");
    arix_tape_destroy(tape);
}

static void test_grad_mse(void) {
    size_t sh[] = {3};
    ArixTensor* tp = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    ArixTensor* tt = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    ((float*)tp->data)[0] = 1; ((float*)tp->data)[1] = 2; ((float*)tp->data)[2] = 3;
    ArixVariable* pred = arix_variable_create(tp, 1);
    ArixVariable* target = arix_variable_create(tt, 0);
    ArixTape* tape = arix_tape_create();
    ArixVariable* loss = arix_mse_loss(tape, pred, target);
    arix_tape_backward(tape, loss);
    float* pg = (float*)pred->grad->data;
    ASSERT(FLOAT_CLOSE(pg[0], 2.0f/3.0f) && FLOAT_CLOSE(pg[1], 4.0f/3.0f) && FLOAT_CLOSE(pg[2], 2.0f), "mse grad = 2*(pred-target)/N");
    arix_tape_destroy(tape);
}

static void test_grad_softmax(void) {
    size_t sh[] = {3};
    float ad[] = {1.0f, 2.0f, 3.0f};
    ArixTensor* ta = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    memcpy(ta->data, ad, 12);
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* s = arix_softmax(tape, a, 0);
    arix_tape_backward(tape, s);
    float* ag = (float*)a->grad->data;
    float sum = ag[0] + ag[1] + ag[2];
    ASSERT(FLOAT_CLOSE(sum, 0.0f), "softmax grads sum to 0");
    arix_tape_destroy(tape);
}

static void test_grad_cross_entropy(void) {
    size_t pred_sh[] = {2, 3}, tgt_sh[] = {2};
    float pd[] = {1.0f, 2.0f, 3.0f, 1.0f, 0.5f, 0.1f};
    float td[] = {1.0f, 0.0f};
    ArixTensor* tp = arix_tensor_zeros(pred_sh, 2, ARIX_FLOAT32);
    ArixTensor* tt = arix_tensor_zeros(tgt_sh, 1, ARIX_FLOAT32);
    memcpy(tp->data, pd, 24);
    memcpy(tt->data, td, 8);
    ArixVariable* pred = arix_variable_create(tp, 1);
    ArixVariable* target = arix_variable_create(tt, 0);
    ArixTape* tape = arix_tape_create();
    ArixVariable* loss = arix_cross_entropy(tape, pred, target);
    arix_tape_backward(tape, loss);
    ASSERT(pred->grad != NULL, "cross_entropy grad exists");
    ASSERT(pred->grad->size == 6, "cross_entropy grad size matches pred");
    arix_tape_destroy(tape);
}

static void test_grad_embedding(void) {
    size_t wsh[] = {4, 2};
    ArixTensor* tw = arix_tensor_zeros(wsh, 2, ARIX_FLOAT32);
    for (size_t i = 0; i < 8; i++) ((float*)tw->data)[i] = (float)(i + 1);
    size_t ish[] = {3};
    ArixTensor* ti = arix_tensor_zeros(ish, 1, ARIX_FLOAT32);
    size_t idx_vals[] = {0, 2, 1};
    memcpy(ti->data, idx_vals, 3 * sizeof(size_t));
    ArixVariable* w = arix_variable_create(tw, 1);
    ArixVariable* idx = arix_variable_create(ti, 0);
    ArixTape* tape = arix_tape_create();
    ArixVariable* e = arix_embedding(tape, w, idx);
    arix_tape_backward(tape, e);
    ASSERT(w->grad != NULL, "embedding grad exists");
    float* wg = (float*)w->grad->data;
    ASSERT(FLOAT_CLOSE(wg[0], 1.0f) && FLOAT_CLOSE(wg[1], 1.0f) && FLOAT_CLOSE(wg[2], 1.0f), "embedding grad at used indices");
    arix_tape_destroy(tape);
}

static void test_grad_layer_norm(void) {
    size_t sh[] = {2, 3};
    ArixTensor* ta = arix_tensor_ones(sh, 2, ARIX_FLOAT32);
    ArixTensor* tg = arix_tensor_ones(sh, 2, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_zeros(sh, 2, ARIX_FLOAT32);
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixVariable* g = arix_variable_create(tg, 1);
    ArixVariable* beta = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* ln = arix_layer_norm(tape, a, g, beta, 1e-5f);
    arix_tape_backward(tape, ln);
    ASSERT(a->grad != NULL, "ln a grad");
    ASSERT(g->grad != NULL, "ln gamma grad");
    ASSERT(beta->grad != NULL, "ln beta grad");
    arix_tape_destroy(tape);
}

static void test_grad_multiple_consumers(void) {
    size_t sh[] = {2};
    float ad[] = {2.0f, 3.0f};
    ArixTensor* ta = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    memcpy(ta->data, ad, 8);
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* b = arix_mul(tape, a, a);
    ArixVariable* c = arix_exp(tape, a);
    ArixVariable* d = arix_add(tape, b, c);
    ArixVariable* loss = arix_sum(tape, d, 0);
    arix_tape_backward(tape, loss);
    float* ag = (float*)a->grad->data;
    ASSERT(FLOAT_CLOSE(ag[0], 4.0f + 7.389f), "multi-consumer grad da = 2*a + exp(a)");
    ASSERT(FLOAT_CLOSE(ag[1], 6.0f + 20.085f), "multi-consumer grad da = 2*a + exp(a)");
    arix_tape_destroy(tape);
}

static void test_grad_no_grad_leaves_zero(void) {
    size_t sh[] = {2};
    ArixTensor* ta = arix_tensor_ones(sh, 1, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_ones(sh, 1, ARIX_FLOAT32);
    ArixVariable* a = arix_variable_create(ta, 0);
    ArixVariable* b = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* c = arix_add(tape, a, b);
    arix_tape_backward(tape, c);
    ASSERT(a->grad == NULL, "no_grad a has no grad");
    ASSERT(b->grad != NULL, "no_grad b has grad");
    arix_tape_destroy(tape);
}

static void test_grad_transpose(void) {
    size_t sh[] = {2, 3};
    ArixTensor* ta = arix_tensor_zeros(sh, 2, ARIX_FLOAT32);
    for (size_t i = 0; i < 6; i++) ((float*)ta->data)[i] = (float)(i + 1);
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* t = arix_transpose(tape, a, 0, 1);
    arix_tape_backward(tape, t);
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[0], 1.0f), "transpose grad preserved");
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[2], 1.0f), "transpose grad preserved");
    arix_tape_destroy(tape);
}

static void test_grad_dropout(void) {
    size_t sh[] = {10};
    ArixTensor* ta = arix_tensor_ones(sh, 1, ARIX_FLOAT32);
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* d = arix_dropout(tape, a, 0.5f, 42);
    arix_tape_backward(tape, d);
    ASSERT(a->grad != NULL, "dropout grad exists");
    arix_tape_destroy(tape);
}

static void test_grad_neg(void) {
    size_t sh[] = {2};
    ArixTensor* ta = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    ((float*)ta->data)[0] = 5.0f; ((float*)ta->data)[1] = -3.0f;
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* n = arix_neg(tape, a);
    arix_tape_backward(tape, n);
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[0], -1.0f), "neg grad = -1");
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[1], -1.0f), "neg grad = -1");
    arix_tape_destroy(tape);
}

static void test_grad_sum_mean(void) {
    size_t sh[] = {2, 3};
    ArixTensor* ta = arix_tensor_zeros(sh, 2, ARIX_FLOAT32);
    for (size_t i = 0; i < 6; i++) ((float*)ta->data)[i] = (float)(i+1);
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* m = arix_mean(tape, a, 1);
    arix_tape_backward(tape, m);
    ASSERT(a->grad != NULL, "sum+mean grad exists");
    arix_tape_destroy(tape);
}

static void test_grad_sqrt_abs(void) {
    size_t sh[] = {2};
    float ad[] = {4.0f, 9.0f};
    ArixTensor* ta = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    memcpy(ta->data, ad, 8);
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* r = arix_sqrt(tape, a);
    arix_tape_backward(tape, r);
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[0], 0.25f), "sqrt grad = 0.5/sqrt(x): 0.25");
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[1], 1.0f/6.0f), "sqrt grad = 0.5/sqrt(9) = 1/6");
    arix_tape_destroy(tape);
}

static void test_grad_var_std(void) {
    size_t sh[] = {3};
    ArixTensor* ta = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    ((float*)ta->data)[0] = 1; ((float*)ta->data)[1] = 2; ((float*)ta->data)[2] = 3;
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* v = arix_var(tape, a, 0);
    arix_tape_backward(tape, v);
    ASSERT(a->grad != NULL, "var grad exists");
    arix_tape_destroy(tape);
}

static void test_grad_log_softmax(void) {
    size_t sh[] = {3};
    float ad[] = {1.0f, 2.0f, 3.0f};
    ArixTensor* ta = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    memcpy(ta->data, ad, 12);
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* l = arix_log_softmax(tape, a, 0);
    arix_tape_backward(tape, l);
    ASSERT(a->grad != NULL, "log_softmax grad exists");
    float sum = 0;
    for (size_t i = 0; i < 3; i++) sum += ((float*)a->grad->data)[i];
    ASSERT(FLOAT_CLOSE(sum, 0.0f), "log_softmax grads sum to 0");
    arix_tape_destroy(tape);
}

static void test_grad_concat(void) {
    size_t sh1[] = {2, 2}, sh2[] = {2, 3};
    ArixTensor* ta = arix_tensor_ones(sh1, 2, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_ones(sh2, 2, ARIX_FLOAT32);
    ArixVariable *a = arix_variable_create(ta, 1), *b = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* vars[] = {a, b};
    ArixVariable* c = arix_concat(tape, vars, 2, 1);
    ArixVariable* s = arix_sum(tape, c, 0);
    arix_tape_backward(tape, s);
    ASSERT(a->grad != NULL && b->grad != NULL, "concat grad exists");
    arix_tape_destroy(tape);
}

static void test_grad_silu_gelu(void) {
    size_t sh[] = {2};
    float ad[] = {0.0f, 1.0f};
    ArixTensor* ta = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    memcpy(ta->data, ad, 8);
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* s = arix_silu(tape, a);
    arix_tape_backward(tape, s);
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[0], 0.5f), "silu grad at 0 = 0.5");
    arix_tape_destroy(tape);
}

static void test_grad_numerical_stability(void) {
    size_t sh[] = {3};
    float ad[] = {0.0f, -1.0f, 1e-10f};
    ArixTensor* ta = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    memcpy(ta->data, ad, 12);
    ((float*)tb->data)[0] = 2.0f; ((float*)tb->data)[1] = 2.0f; ((float*)tb->data)[2] = 2.0f;
    ArixVariable *a = arix_variable_create(ta, 1), *b = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* d = arix_div(tape, a, b);
    (void)d;
    ArixVariable* l = arix_log(tape, arix_abs(tape, a));
    (void)l;
    ArixVariable* s = arix_sqrt(tape, arix_abs(tape, a));
    (void)s;
    arix_tape_backward(tape, s);
    ASSERT(a->grad != NULL, "numerical stability: grad exists");
    arix_tape_destroy(tape);
}

static void test_grad_tape_clear(void) {
    size_t sh[] = {2};
    ArixTensor* ta = arix_tensor_ones(sh, 1, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_ones(sh, 1, ARIX_FLOAT32);
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixVariable* b = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    arix_tape_record(tape, a);
    arix_tape_record(tape, b);
    ArixVariable* c = arix_add(tape, a, b);
    arix_tape_backward(tape, c);
    ASSERT(a->grad != NULL, "grad exists before clear");
    arix_tape_zero_grad(tape);
    ASSERT(a->grad == NULL, "grad cleared");
    arix_tape_destroy(tape);
}

static void test_grad_global_norm(void) {
    size_t sh[] = {2};
    ArixTensor* ta = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    ((float*)ta->data)[0] = 2; ((float*)ta->data)[1] = 3;
    ((float*)tb->data)[0] = 0; ((float*)tb->data)[1] = 4;
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixVariable* b = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    arix_tape_record(tape, a);
    arix_tape_record(tape, b);
    ArixVariable* c = arix_add(tape, a, b);
    arix_tape_backward(tape, c);
    if (c->grad) { arix_tensor_destroy(c->grad); c->grad = NULL; }
    float gn = arix_tape_global_norm(tape);
    ASSERT(FLOAT_CLOSE(gn, sqrtf(4)), "global norm = 2");
    arix_tape_destroy(tape);
}

static void test_grad_clip_grad_norm(void) {
    size_t sh[] = {2};
    ArixTensor* ta = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    ((float*)ta->data)[0] = 10; ((float*)ta->data)[1] = 0;
    ((float*)tb->data)[0] = 0; ((float*)tb->data)[1] = 0;
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixVariable* b = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    arix_tape_record(tape, a);
    arix_tape_record(tape, b);
    ArixVariable* c = arix_add(tape, a, b);
    arix_tape_backward(tape, c);
    if (c->grad) { arix_tensor_destroy(c->grad); c->grad = NULL; }
    arix_tape_clip_grad_norm(tape, 1.0f);
    float gn = arix_tape_global_norm(tape);
    ASSERT(FLOAT_CLOSE(gn, 1.0f), "clipped norm = 1");
    arix_tape_destroy(tape);
}

static void test_grad_minimum(void) {
    size_t sh[] = {3};
    float ad[] = {1.0f, 5.0f, 3.0f};
    float bd[] = {4.0f, 2.0f, 6.0f};
    ArixTensor* ta = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    memcpy(ta->data, ad, 12); memcpy(tb->data, bd, 12);
    ArixVariable *a = arix_variable_create(ta, 1), *b = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* c = arix_minimum(tape, a, b);
    arix_tape_backward(tape, c);
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[0], 1.0f), "min a[0]=1 (< b[0])");
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[1], 0.0f), "min a[1]=0 (> b[1])");
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[2], 1.0f), "min a[2]=1 (< b[2])");
    ASSERT(FLOAT_CLOSE(((float*)b->grad->data)[0], 0.0f), "min b[0]=0 (> a[0])");
    ASSERT(FLOAT_CLOSE(((float*)b->grad->data)[1], 1.0f), "min b[1]=1 (< a[1])");
    ASSERT(FLOAT_CLOSE(((float*)b->grad->data)[2], 0.0f), "min b[2]=0 (> a[2])");
    arix_tape_destroy(tape);
}

static void test_grad_maximum(void) {
    size_t sh[] = {3};
    float ad[] = {1.0f, 5.0f, 3.0f};
    float bd[] = {4.0f, 2.0f, 6.0f};
    ArixTensor* ta = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_zeros(sh, 1, ARIX_FLOAT32);
    memcpy(ta->data, ad, 12); memcpy(tb->data, bd, 12);
    ArixVariable *a = arix_variable_create(ta, 1), *b = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* c = arix_maximum(tape, a, b);
    arix_tape_backward(tape, c);
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[0], 0.0f), "max a[0]=0 (< b[0])");
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[1], 1.0f), "max a[1]=1 (> b[1])");
    ASSERT(FLOAT_CLOSE(((float*)a->grad->data)[2], 0.0f), "max a[2]=0 (< b[2])");
    ASSERT(FLOAT_CLOSE(((float*)b->grad->data)[0], 1.0f), "max b[0]=1 (> a[0])");
    ASSERT(FLOAT_CLOSE(((float*)b->grad->data)[1], 0.0f), "max b[1]=0 (< a[1])");
    ASSERT(FLOAT_CLOSE(((float*)b->grad->data)[2], 1.0f), "max b[2]=1 (> a[2])");
    arix_tape_destroy(tape);
}

static void test_grad_conv2d(void) {
    size_t ishape[] = {1, 1, 4, 4};
    size_t kshape[] = {1, 1, 2, 2};
    ArixTensor* input = arix_tensor_zeros(ishape, 4, ARIX_FLOAT32);
    ArixTensor* kernel = arix_tensor_zeros(kshape, 4, ARIX_FLOAT32);
    float* xd = (float*)input->data;
    float* kd = (float*)kernel->data;
    for (size_t i = 0; i < 16; i++) xd[i] = (float)((i * 7 + 3) % 13) / 13.0f;
    for (size_t i = 0; i < 4; i++) kd[i] = (float)((i * 13 + 7) % 11) / 11.0f;

    ArixVariable *va = arix_variable_create(input, 1);
    ArixVariable *vk = arix_variable_create(kernel, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* conv = arix_conv2d(tape, va, vk, 1, 1, 0, 0);
    size_t N_conv = conv->data->size;
    ArixTensor* tgt_data = arix_tensor_zeros(conv->data->shape, conv->data->ndim, ARIX_FLOAT32);
    ArixVariable* target = arix_variable_create(tgt_data, 0);
    ArixVariable* loss = arix_mse_loss(tape, conv, target);
    arix_tape_backward(tape, loss);

    for (size_t idx = 0; idx < 16; idx++) {
        float orig = xd[idx];
        float eps = 1e-4f;
        xd[idx] = orig + eps;
        ArixTensor* out_p = arix_tensor_conv2d(input, kernel, 1, 1, 0, 0);
        float lp = 0;
        for (size_t i = 0; i < out_p->size; i++) { float v = ((float*)out_p->data)[i]; lp += v * v / (float)N_conv; }
        arix_tensor_destroy(out_p);
        xd[idx] = orig - eps;
        ArixTensor* out_m = arix_tensor_conv2d(input, kernel, 1, 1, 0, 0);
        float lm = 0;
        for (size_t i = 0; i < out_m->size; i++) { float v = ((float*)out_m->data)[i]; lm += v * v / (float)N_conv; }
        arix_tensor_destroy(out_m);
        xd[idx] = orig;
        float fd = (lp - lm) / (2.0f * eps);
        float ad = ((float*)va->grad->data)[idx];
        float ratio = fabsf(fd) > 1e-6f ? fabsf(ad - fd) / (fabsf(fd) + 1e-6f) : fabsf(ad - fd);
        ASSERT(ratio < 0.15f, "conv2d input grad finite-diff match");
    }

    for (size_t idx = 0; idx < 4; idx++) {
        float orig = kd[idx];
        float eps = 1e-4f;
        kd[idx] = orig + eps;
        ArixTensor* out_p = arix_tensor_conv2d(input, kernel, 1, 1, 0, 0);
        float lp = 0;
        for (size_t i = 0; i < out_p->size; i++) { float v = ((float*)out_p->data)[i]; lp += v * v / (float)N_conv; }
        arix_tensor_destroy(out_p);
        kd[idx] = orig - eps;
        ArixTensor* out_m = arix_tensor_conv2d(input, kernel, 1, 1, 0, 0);
        float lm = 0;
        for (size_t i = 0; i < out_m->size; i++) { float v = ((float*)out_m->data)[i]; lm += v * v / (float)N_conv; }
        arix_tensor_destroy(out_m);
        kd[idx] = orig;
        float fd = (lp - lm) / (2.0f * eps);
        float ad = ((float*)vk->grad->data)[idx];
        float ratio = fabsf(fd) > 1e-6f ? fabsf(ad - fd) / (fabsf(fd) + 1e-6f) : fabsf(ad - fd);
        ASSERT(ratio < 0.15f, "conv2d kernel grad finite-diff match");
    }

    arix_variable_destroy(target);
    arix_tape_destroy(tape);
}

int main(void) {
    run_test("add grad",              test_grad_add);
    run_test("mul grad",              test_grad_mul);
    run_test("matmul grad",           test_grad_matmul);
    run_test("div grad",              test_grad_div);
    run_test("relu grad",             test_grad_relu);
    run_test("sigmoid grad",          test_grad_sigmoid);
    run_test("tanh grad",             test_grad_tanh);
    run_test("exp/log grad",          test_grad_exp_log);
    run_test("pow grad",              test_grad_pow);
    run_test("sin/cos grad",          test_grad_sin_cos);
    run_test("chain shared input",    test_grad_chain_shared_input);
    run_test("mse loss grad",         test_grad_mse);
    run_test("softmax grad",          test_grad_softmax);
    run_test("cross_entropy grad",    test_grad_cross_entropy);
    run_test("embedding grad",        test_grad_embedding);
    run_test("layer_norm grad",       test_grad_layer_norm);
    run_test("multi-consumer grad",   test_grad_multiple_consumers);
    run_test("no_grad leaves zero",   test_grad_no_grad_leaves_zero);
    run_test("transpose grad",        test_grad_transpose);
    run_test("dropout grad",          test_grad_dropout);
    run_test("neg grad",              test_grad_neg);
    run_test("sum/mean grad",         test_grad_sum_mean);
    run_test("sqrt/abs grad",         test_grad_sqrt_abs);
    run_test("var/std grad",          test_grad_var_std);
    run_test("log_softmax grad",      test_grad_log_softmax);
    run_test("concat grad",           test_grad_concat);
    run_test("silu/gelu grad",        test_grad_silu_gelu);
    run_test("numerical stability",   test_grad_numerical_stability);
    run_test("tape zero_grad",        test_grad_tape_clear);
    run_test("global norm",           test_grad_global_norm);
    run_test("clip grad norm",        test_grad_clip_grad_norm);

    run_test("min grad",              test_grad_minimum);
    run_test("max grad",              test_grad_maximum);
    run_test("conv2d grad",           test_grad_conv2d);

    printf("\nResults: %d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
