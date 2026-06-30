#include "automatic_differentiation_framework.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_passed = 0, tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } \
} while(0)

static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout);
    fn(); printf("PASS\n"); tests_passed++;
}

static int float_close(float a, float b) { return fabsf(a - b) < 1e-3f; }

static void test_variable_create(void) {
    size_t shape[] = {2, 3};
    ArixTensor* t = arix_tensor_zeros(shape, 2, ARIX_FLOAT32);
    ASSERT(t != NULL, "tensor created");
    ArixVariable* v = arix_variable_create(t, 1);
    ASSERT(v != NULL, "var created");
    ASSERT(v->data == t, "data matches");
    ASSERT(v->requires_grad == 1, "requires_grad set");
    ASSERT(v->grad == NULL, "grad is NULL");
    ASSERT(v->backward_fn == NULL, "backward_fn NULL");
    arix_variable_destroy(v);
}

static void test_add_forward(void) {
    size_t shape[] = {3};
    ArixTensor* ta = arix_tensor_ones(shape, 1, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_ones(shape, 1, ARIX_FLOAT32);
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixVariable* b = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* c = arix_add(tape, a, b);
    ASSERT(c != NULL, "add result");
    float* cd = (float*)c->data->data;
    ASSERT(float_close(cd[0], 2.0f) && float_close(cd[1], 2.0f) && float_close(cd[2], 2.0f), "add values");
    arix_tape_destroy(tape);
}

static void test_add_backward(void) {
    size_t shape[] = {3};
    ArixTensor* ta = arix_tensor_ones(shape, 1, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_ones(shape, 1, ARIX_FLOAT32);
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixVariable* b = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* c = arix_add(tape, a, b);
    arix_tape_backward(tape, c);
    ASSERT(a->grad != NULL, "a has grad");
    ASSERT(b->grad != NULL, "b has grad");
    float* ag = (float*)a->grad->data;
    float* bg = (float*)b->grad->data;
    ASSERT(float_close(ag[0], 1.0f) && float_close(ag[1], 1.0f) && float_close(ag[2], 1.0f), "a grad = 1");
    ASSERT(float_close(bg[0], 1.0f) && float_close(bg[1], 1.0f) && float_close(bg[2], 1.0f), "b grad = 1");
    arix_tape_destroy(tape);
}

static void test_mul_backward(void) {
    size_t shape[] = {3};
    float adata[] = {2.0f, 3.0f, 4.0f};
    float bdata[] = {5.0f, 6.0f, 7.0f};
    ArixTensor* ta = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    memcpy(ta->data, adata, 3 * sizeof(float));
    memcpy(tb->data, bdata, 3 * sizeof(float));
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixVariable* b = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* c = arix_mul(tape, a, b);
    arix_tape_backward(tape, c);
    float* ag = (float*)a->grad->data;
    float* bg = (float*)b->grad->data;
    ASSERT(float_close(ag[0], 5.0f) && float_close(ag[1], 6.0f) && float_close(ag[2], 7.0f), "a grad = b");
    ASSERT(float_close(bg[0], 2.0f) && float_close(bg[1], 3.0f) && float_close(bg[2], 4.0f), "b grad = a");
    arix_tape_destroy(tape);
}

static void test_matmul_backward(void) {
    size_t sha[] = {2, 3}, shb[] = {3, 2};
    float adata[] = {1,2,3,4,5,6};
    float bdata[] = {7,8,9,10,11,12};
    ArixTensor* ta = arix_tensor_zeros(sha, 2, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_zeros(shb, 2, ARIX_FLOAT32);
    memcpy(ta->data, adata, 6 * sizeof(float));
    memcpy(tb->data, bdata, 6 * sizeof(float));
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixVariable* b = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* c = arix_matmul(tape, a, b);
    arix_tape_backward(tape, c);
    ASSERT(a->grad != NULL, "a grad exists");
    ASSERT(b->grad != NULL, "b grad exists");
    float* ag = (float*)a->grad->data;
    float* bg = (float*)b->grad->data;
    ASSERT(float_close(ag[0], 15.0f) && float_close(ag[5], 23.0f), "matmul grad a");
    ASSERT(float_close(bg[0], 5.0f) && float_close(bg[5], 9.0f), "matmul grad b");
    arix_tape_destroy(tape);
}

static void test_mse_loss_backward(void) {
    size_t shape[] = {3};
    float pdata[] = {1.0f, 2.0f, 3.0f};
    float tdata[] = {0.0f, 0.0f, 0.0f};
    ArixTensor* tp = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    ArixTensor* tt = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    memcpy(tp->data, pdata, 3 * sizeof(float));
    memcpy(tt->data, tdata, 3 * sizeof(float));
    ArixVariable* pred = arix_variable_create(tp, 1);
    ArixVariable* target = arix_variable_create(tt, 0);
    ArixTape* tape = arix_tape_create();
    ArixVariable* loss = arix_mse_loss(tape, pred, target);
    ASSERT(loss != NULL, "mse loss");
    arix_tape_backward(tape, loss);
    ASSERT(pred->grad != NULL, "pred has grad");
    float* pg = (float*)pred->grad->data;
    ASSERT(float_close(pg[0], 2.0f/3.0f), "mse grad");
    arix_tape_destroy(tape);
}

static void test_relu_backward(void) {
    size_t shape[] = {4};
    float adata[] = {-2.0f, 0.0f, 3.0f, 5.0f};
    ArixTensor* ta = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    memcpy(ta->data, adata, 4 * sizeof(float));
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* r = arix_relu(tape, a);
    arix_tape_backward(tape, r);
    float* ag = (float*)a->grad->data;
    ASSERT(float_close(ag[0], 0.0f) && float_close(ag[1], 0.0f), "relu grad neg=0");
    ASSERT(float_close(ag[2], 1.0f) && float_close(ag[3], 1.0f), "relu grad pos=1");
    arix_tape_destroy(tape);
}

static void test_sigmoid_backward(void) {
    size_t shape[] = {2};
    float adata[] = {0.0f, 1.0f};
    ArixTensor* ta = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    memcpy(ta->data, adata, 2 * sizeof(float));
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* s = arix_sigmoid(tape, a);
    arix_tape_backward(tape, s);
    ASSERT(a->grad != NULL, "sigmoid grad");
    float* ag = (float*)a->grad->data;
    ASSERT(float_close(ag[0], 0.25f), "sigmoid grad at 0 = 0.25");
    ASSERT(float_close(ag[1], 0.1966f), "sigmoid grad at 1 ~ 0.1966");
    arix_tape_destroy(tape);
}

static void test_tanh_backward(void) {
    size_t shape[] = {2};
    float adata[] = {0.0f, 1.0f};
    ArixTensor* ta = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    memcpy(ta->data, adata, 2 * sizeof(float));
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* h = arix_tanh(tape, a);
    arix_tape_backward(tape, h);
    ASSERT(a->grad != NULL, "tanh grad");
    float* ag = (float*)a->grad->data;
    ASSERT(float_close(ag[0], 1.0f), "tanh grad at 0 = 1");
    ASSERT(float_close(ag[1], 0.4199f), "tanh grad at 1 ~ 0.4199");
    arix_tape_destroy(tape);
}

static void test_softmax_backward(void) {
    size_t shape[] = {3};
    float adata[] = {1.0f, 2.0f, 3.0f};
    ArixTensor* ta = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    memcpy(ta->data, adata, 3 * sizeof(float));
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* s = arix_softmax(tape, a, 0);
    ASSERT(s != NULL, "softmax ok");
    arix_tape_backward(tape, s);
    ASSERT(a->grad != NULL, "softmax grad");
    float* ag = (float*)a->grad->data;
    float sum = 0;
    for (size_t i = 0; i < 3; i++) sum += ag[i];
    ASSERT(float_close(sum, 0.0f), "softmax grads sum to 0");
    arix_tape_destroy(tape);
}

static void test_no_grad(void) {
    size_t shape[] = {2};
    ArixTensor* ta = arix_tensor_ones(shape, 1, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_ones(shape, 1, ARIX_FLOAT32);
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixVariable* b = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    arix_no_grad_enter();
    ArixVariable* c = arix_add(tape, a, b);
    arix_no_grad_exit();
    ASSERT(c != NULL, "add in no_grad");
    ASSERT(c->backward_fn == NULL, "no backward_fn in no_grad");
    ASSERT(tape->num_vars == 0, "no tape record in no_grad");
    arix_variable_destroy(c);
    arix_tape_destroy(tape);
}

static void test_exp_backward(void) {
    size_t shape[] = {2};
    float adata[] = {0.0f, 1.0f};
    ArixTensor* ta = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    memcpy(ta->data, adata, 2 * sizeof(float));
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* e = arix_exp(tape, a);
    arix_tape_backward(tape, e);
    ASSERT(a->grad != NULL, "exp grad");
    float* ag = (float*)a->grad->data;
    ASSERT(float_close(ag[0], 1.0f), "exp grad at 0 = 1");
    ASSERT(float_close(ag[1], 2.71828f), "exp grad at 1 = e");
    arix_tape_destroy(tape);
}

static void test_sum_backward(void) {
    size_t shape[] = {2, 3};
    ArixTensor* ta = arix_tensor_ones(shape, 2, ARIX_FLOAT32);
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* s = arix_sum(tape, a, 0);
    arix_tape_backward(tape, s);
    ASSERT(a->grad != NULL, "sum grad");
    float* ag = (float*)a->grad->data;
    for (size_t i = 0; i < 6; i++) ASSERT(float_close(ag[i], 1.0f), "sum grad=1 expanded");
    arix_tape_destroy(tape);
}

static void test_chain_rule(void) {
    size_t shape[] = {2};
    float adata[] = {2.0f, 3.0f};
    ArixTensor* ta = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    memcpy(ta->data, adata, 2 * sizeof(float));
    memset(tb->data, 0, 2 * sizeof(float));
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixVariable* b = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* c = arix_add(tape, a, b);
    ArixVariable* d = arix_mul(tape, c, a);
    ArixVariable* loss = arix_sum(tape, d, 0);
    arix_tape_backward(tape, loss);
    ASSERT(a->grad != NULL, "chain a grad");
    float* ag = (float*)a->grad->data;
    ASSERT(float_close(ag[0], 4.0f), "chain grad a[0]");
    ASSERT(float_close(ag[1], 6.0f), "chain grad a[1]");
    arix_tape_destroy(tape);
}

static void test_dropout_backward(void) {
    size_t shape[] = {4};
    ArixTensor* ta = arix_tensor_ones(shape, 1, ARIX_FLOAT32);
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* d = arix_dropout(tape, a, 0.5f, 42);
    ASSERT(d != NULL, "dropout result");
    ASSERT(d->backward_fn != NULL, "dropout has backward");
    arix_tape_backward(tape, d);
    ASSERT(a->grad != NULL, "dropout grad exists");
    arix_tape_destroy(tape);
}

static void test_layer_norm_backward(void) {
    size_t shape[] = {2, 3};
    ArixTensor* ta = arix_tensor_ones(shape, 2, ARIX_FLOAT32);
    ArixTensor* tg = arix_tensor_ones(shape, 2, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_zeros(shape, 2, ARIX_FLOAT32);
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixVariable* g = arix_variable_create(tg, 1);
    ArixVariable* beta = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* ln = arix_layer_norm(tape, a, g, beta, 1e-5f);
    ASSERT(ln != NULL, "layer_norm result");
    arix_tape_backward(tape, ln);
    ASSERT(a->grad != NULL, "ln grad a");
    ASSERT(g->grad != NULL, "ln grad gamma");
    arix_tape_destroy(tape);
}

static void test_sub_backward(void) {
    size_t shape[] = {3};
    float adata[] = {5.0f, 6.0f, 7.0f};
    float bdata[] = {1.0f, 2.0f, 3.0f};
    ArixTensor* ta = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    memcpy(ta->data, adata, 3 * sizeof(float));
    memcpy(tb->data, bdata, 3 * sizeof(float));
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixVariable* b = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* c = arix_sub(tape, a, b);
    arix_tape_backward(tape, c);
    ASSERT(a->grad != NULL, "sub a grad");
    ASSERT(b->grad != NULL, "sub b grad");
    float* ag = (float*)a->grad->data;
    float* bg = (float*)b->grad->data;
    ASSERT(float_close(ag[0], 1.0f), "sub da/dc=1");
    ASSERT(float_close(bg[0], -1.0f), "sub db/dc=-1");
    arix_tape_destroy(tape);
}

static void test_div_backward(void) {
    size_t shape[] = {2};
    float adata[] = {8.0f, 9.0f};
    float bdata[] = {2.0f, 3.0f};
    ArixTensor* ta = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    memcpy(ta->data, adata, 2 * sizeof(float));
    memcpy(tb->data, bdata, 2 * sizeof(float));
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixVariable* b = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* c = arix_div(tape, a, b);
    arix_tape_backward(tape, c);
    ASSERT(a->grad != NULL, "div a grad");
    ASSERT(b->grad != NULL, "div b grad");
    float* ag = (float*)a->grad->data;
    float* bg = (float*)b->grad->data;
    ASSERT(float_close(ag[0], 0.5f), "div da=1/b");
    ASSERT(float_close(bg[0], -2.0f), "div db=-a/b^2");
    arix_tape_destroy(tape);
}

static void test_pow_backward(void) {
    size_t shape[] = {2};
    float adata[] = {2.0f, 3.0f};
    ArixTensor* ta = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    memcpy(ta->data, adata, 2 * sizeof(float));
    ((float*)tb->data)[0] = 2.0f; ((float*)tb->data)[1] = 2.0f;
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixVariable* b = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* c = arix_pow(tape, a, b);
    arix_tape_backward(tape, c);
    ASSERT(a->grad != NULL, "pow a grad");
    float* ag = (float*)a->grad->data;
    ASSERT(float_close(ag[0], 4.0f), "pow d(x^2)/dx=2x -> 4");
    ASSERT(float_close(ag[1], 6.0f), "pow d(x^2)/dx=2x -> 6");
    arix_tape_destroy(tape);
}

static void test_neg_backward(void) {
    size_t shape[] = {3};
    float adata[] = {1.0f, -2.0f, 3.0f};
    ArixTensor* ta = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    memcpy(ta->data, adata, 3 * sizeof(float));
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* n = arix_neg(tape, a);
    arix_tape_backward(tape, n);
    ASSERT(a->grad != NULL, "neg grad");
    float* ag = (float*)a->grad->data;
    ASSERT(float_close(ag[0], -1.0f), "neg grad = -1");
    arix_tape_destroy(tape);
}

static void test_gelu_backward(void) {
    size_t shape[] = {2};
    float adata[] = {0.0f, 1.0f};
    ArixTensor* ta = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    memcpy(ta->data, adata, 2 * sizeof(float));
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* g = arix_gelu(tape, a);
    arix_tape_backward(tape, g);
    ASSERT(a->grad != NULL, "gelu grad");
    float* ag = (float*)a->grad->data;
    ASSERT(float_close(ag[0], 0.5f), "gelu d/dx at 0 = 0.5");
    arix_tape_destroy(tape);
}

static void test_silu_backward(void) {
    size_t shape[] = {2};
    float adata[] = {0.0f, 1.0f};
    ArixTensor* ta = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    memcpy(ta->data, adata, 2 * sizeof(float));
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* s = arix_silu(tape, a);
    arix_tape_backward(tape, s);
    ASSERT(a->grad != NULL, "silu grad");
    float* ag = (float*)a->grad->data;
    ASSERT(float_close(ag[0], 0.5f), "silu grad at 0 = 0.5");
    arix_tape_destroy(tape);
}

static void test_log_backward(void) {
    size_t shape[] = {2};
    float adata[] = {1.0f, 2.0f};
    ArixTensor* ta = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    memcpy(ta->data, adata, 2 * sizeof(float));
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* l = arix_log(tape, a);
    arix_tape_backward(tape, l);
    ASSERT(a->grad != NULL, "log grad");
    float* ag = (float*)a->grad->data;
    ASSERT(float_close(ag[0], 1.0f), "log grad at 1 = 1");
    ASSERT(float_close(ag[1], 0.5f), "log grad at 2 = 0.5");
    arix_tape_destroy(tape);
}

static void test_mean_backward(void) {
    size_t shape[] = {2, 3};
    ArixTensor* ta = arix_tensor_ones(shape, 2, ARIX_FLOAT32);
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* m = arix_mean(tape, a, 0);
    arix_tape_backward(tape, m);
    ASSERT(a->grad != NULL, "mean grad");
    float* ag = (float*)a->grad->data;
    ASSERT(float_close(ag[0], 0.5f), "mean grad = 1/N");
    arix_tape_destroy(tape);
}

static void test_transpose_backward(void) {
    size_t shape[] = {2, 3};
    float adata[] = {1,2,3,4,5,6};
    ArixTensor* ta = arix_tensor_zeros(shape, 2, ARIX_FLOAT32);
    memcpy(ta->data, adata, 6 * sizeof(float));
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* t = arix_transpose(tape, a, 0, 1);
    arix_tape_backward(tape, t);
    ASSERT(a->grad != NULL, "transpose grad");
    float* ag = (float*)a->grad->data;
    ASSERT(float_close(ag[0], 1.0f), "transpose grad matches");
    ASSERT(float_close(ag[3], 1.0f), "transpose grad matches");
    arix_tape_destroy(tape);
}

static void test_concat_backward(void) {
    size_t sh1[] = {2, 2}, sh2[] = {2, 3};
    ArixTensor* ta = arix_tensor_ones(sh1, 2, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_ones(sh2, 2, ARIX_FLOAT32);
    ArixVariable* a = arix_variable_create(ta, 1);
    ArixVariable* b = arix_variable_create(tb, 1);
    ArixTape* tape = arix_tape_create();
    ArixVariable* vars[] = {a, b};
    ArixVariable* c = arix_concat(tape, vars, 2, 1);
    arix_tape_backward(tape, c);
    ASSERT(a->grad != NULL, "concat a grad");
    ASSERT(b->grad != NULL, "concat b grad");
    float* ag = (float*)a->grad->data;
    for (size_t i = 0; i < 4; i++) ASSERT(float_close(ag[i], 1.0f), "concat grad ones");
    arix_tape_destroy(tape);
}

int main(void) {
    run_test("variable_create",       test_variable_create);
    run_test("add_forward",           test_add_forward);
    run_test("add_backward",          test_add_backward);
    run_test("mul_backward",          test_mul_backward);
    run_test("matmul_backward",       test_matmul_backward);
    run_test("mse_loss_backward",     test_mse_loss_backward);
    run_test("relu_backward",         test_relu_backward);
    run_test("sigmoid_backward",      test_sigmoid_backward);
    run_test("tanh_backward",         test_tanh_backward);
    run_test("softmax_backward",      test_softmax_backward);
    run_test("no_grad",               test_no_grad);
    run_test("exp_backward",          test_exp_backward);
    run_test("sum_backward",          test_sum_backward);
    run_test("chain_rule",            test_chain_rule);
    run_test("dropout_backward",      test_dropout_backward);
    run_test("layer_norm_backward",   test_layer_norm_backward);
    run_test("sub_backward",          test_sub_backward);
    run_test("div_backward",          test_div_backward);
    run_test("pow_backward",          test_pow_backward);
    run_test("neg_backward",          test_neg_backward);
    run_test("gelu_backward",         test_gelu_backward);
    run_test("silu_backward",         test_silu_backward);
    run_test("log_backward",          test_log_backward);
    run_test("mean_backward",         test_mean_backward);
    run_test("transpose_backward",    test_transpose_backward);
    run_test("concat_backward",       test_concat_backward);

    printf("\n%d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
