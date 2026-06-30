#include "multidimensional_tensor_engine.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <float.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s (%s)\n", msg, #cond); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, eps, msg) do { \
    if (fabsf((a) - (b)) > (eps)) { \
        printf("FAIL: %s (got %f, expected %f)\n", msg, (float)(a), (float)(b)); \
        tests_failed++; \
        return; \
    } \
} while(0)

static void run_test(const char* name, void (*test_fn)(void)) {
    printf("Running %s... ", name);
    fflush(stdout);
    test_fn();
    printf("PASS\n");
    tests_passed++;
}

/* ---------- Element-wise Arithmetic ---------- */

static void test_add_basic(void) {
    size_t shape[] = {2, 3};
    ArixTensor* a = arix_tensor_ones(shape, 2, ARIX_FLOAT32);
    ArixTensor* b = arix_tensor_ones(shape, 2, ARIX_FLOAT32);
    ArixTensor* c = arix_tensor_add(a, b);
    ASSERT(c != NULL, "add result not null");
    float* d = (float*)c->data;
    for (size_t i = 0; i < 6; i++) ASSERT_NEAR(d[i], 2.0f, 1e-6f, "1+1==2");
    arix_tensor_destroy(a); arix_tensor_destroy(b); arix_tensor_destroy(c);
}

static void test_add_broadcast(void) {
    size_t shape_a[] = {2, 3};
    size_t shape_b[] = {3};
    ArixTensor* a = arix_tensor_full(shape_a, 2, ARIX_FLOAT32, (float[]){2.0f});
    ArixTensor* b = arix_tensor_ones(shape_b, 1, ARIX_FLOAT32);
    ArixTensor* c = arix_tensor_add(a, b);
    ASSERT(c != NULL, "broadcast add not null");
    float* d = (float*)c->data;
    for (size_t i = 0; i < 6; i++) ASSERT_NEAR(d[i], 3.0f, 1e-6f, "2+1==3");
    arix_tensor_destroy(a); arix_tensor_destroy(b); arix_tensor_destroy(c);
}

static void test_sub_basic(void) {
    size_t shape[] = {3};
    ArixTensor* a = arix_tensor_arange(5.0f, 0.0f, -1.0f, ARIX_FLOAT32);
    ArixTensor* b = arix_tensor_ones(shape, 1, ARIX_FLOAT32);
    ArixTensor* c = arix_tensor_sub(a, b);
    ASSERT(c != NULL, "sub result not null");
    float* d = (float*)c->data;
    ASSERT_NEAR(d[0], 4.0f, 1e-6f, "5-1==4");
    ASSERT_NEAR(d[1], 3.0f, 1e-6f, "4-1==3");
    ASSERT_NEAR(d[2], 2.0f, 1e-6f, "3-1==2");
    arix_tensor_destroy(a); arix_tensor_destroy(b); arix_tensor_destroy(c);
}

static void test_mul_basic(void) {
    size_t shape[] = {3};
    float two = 2.0f;
    ArixTensor* a = arix_tensor_arange(1.0f, 4.0f, 1.0f, ARIX_FLOAT32);
    ArixTensor* b = arix_tensor_full(shape, 1, ARIX_FLOAT32, &two);
    ArixTensor* c = arix_tensor_mul(a, b);
    ASSERT(c != NULL, "mul result not null");
    float* d = (float*)c->data;
    ASSERT_NEAR(d[0], 2.0f, 1e-6f, "1*2==2");
    ASSERT_NEAR(d[1], 4.0f, 1e-6f, "2*2==4");
    ASSERT_NEAR(d[2], 6.0f, 1e-6f, "3*2==6");
    arix_tensor_destroy(a); arix_tensor_destroy(b); arix_tensor_destroy(c);
}

static void test_div_basic(void) {
    size_t shape[] = {3};
    ArixTensor* a = arix_tensor_arange(2.0f, 8.0f, 2.0f, ARIX_FLOAT32);
    float two = 2.0f;
    ArixTensor* b = arix_tensor_full(shape, 1, ARIX_FLOAT32, &two);
    ArixTensor* c = arix_tensor_div(a, b);
    ASSERT(c != NULL, "div result not null");
    float* d = (float*)c->data;
    ASSERT_NEAR(d[0], 1.0f, 1e-6f, "2/2==1");
    ASSERT_NEAR(d[1], 2.0f, 1e-6f, "4/2==2");
    ASSERT_NEAR(d[2], 3.0f, 1e-6f, "6/2==3");
    arix_tensor_destroy(a); arix_tensor_destroy(b); arix_tensor_destroy(c);
}

static void test_pow_basic(void) {
    size_t shape[] = {3};
    ArixTensor* a = arix_tensor_arange(1.0f, 4.0f, 1.0f, ARIX_FLOAT32);
    float two = 2.0f;
    ArixTensor* b = arix_tensor_full(shape, 1, ARIX_FLOAT32, &two);
    ArixTensor* c = arix_tensor_pow(a, b);
    ASSERT(c != NULL, "pow result not null");
    float* d = (float*)c->data;
    ASSERT_NEAR(d[0], 1.0f, 1e-5f, "1^2==1");
    ASSERT_NEAR(d[1], 4.0f, 1e-5f, "2^2==4");
    ASSERT_NEAR(d[2], 9.0f, 1e-5f, "3^2==9");
    arix_tensor_destroy(a); arix_tensor_destroy(b); arix_tensor_destroy(c);
}

static void test_neg_basic(void) {
    size_t shape[] = {3};
    ArixTensor* a = arix_tensor_arange(1.0f, 4.0f, 1.0f, ARIX_FLOAT32);
    ArixTensor* b = arix_tensor_neg(a);
    ASSERT(b != NULL, "neg result not null");
    float* d = (float*)b->data;
    ASSERT_NEAR(d[0], -1.0f, 1e-6f, "-1");
    ASSERT_NEAR(d[1], -2.0f, 1e-6f, "-2");
    ASSERT_NEAR(d[2], -3.0f, 1e-6f, "-3");
    arix_tensor_destroy(a); arix_tensor_destroy(b);
}

static void test_abs_basic(void) {
    size_t shape[] = {3};
    float vals[] = {-3.0f, 0.0f, 4.0f};
    ArixTensor* a = arix_tensor_empty(NULL, 0, ARIX_FLOAT32);
    a = arix_tensor_create(shape, 1, ARIX_FLOAT32);
    memcpy(a->data, vals, 3 * sizeof(float));
    ArixTensor* b = arix_tensor_abs(a);
    ASSERT(b != NULL, "abs result not null");
    float* d = (float*)b->data;
    ASSERT_NEAR(d[0], 3.0f, 1e-6f, "abs(-3)==3");
    ASSERT_NEAR(d[1], 0.0f, 1e-6f, "abs(0)==0");
    ASSERT_NEAR(d[2], 4.0f, 1e-6f, "abs(4)==4");
    arix_tensor_destroy(a); arix_tensor_destroy(b);
}

static void test_sign_basic(void) {
    size_t shape[] = {3};
    float vals[] = {-5.0f, 0.0f, 3.0f};
    ArixTensor* a = arix_tensor_create(shape, 1, ARIX_FLOAT32);
    memcpy(a->data, vals, 3 * sizeof(float));
    ArixTensor* b = arix_tensor_sign(a);
    ASSERT(b != NULL, "sign result not null");
    float* d = (float*)b->data;
    ASSERT_NEAR(d[0], -1.0f, 1e-6f, "sign(-5)==-1");
    ASSERT_NEAR(d[1], 0.0f, 1e-6f, "sign(0)==0");
    ASSERT_NEAR(d[2], 1.0f, 1e-6f, "sign(3)==1");
    arix_tensor_destroy(a); arix_tensor_destroy(b);
}

static void test_exp_log_sqrt(void) {
    size_t shape[] = {3};
    ArixTensor* a = arix_tensor_arange(1.0f, 4.0f, 1.0f, ARIX_FLOAT32);
    ArixTensor* e = arix_tensor_exp(a);
    ASSERT(e != NULL, "exp result not null");
    float* ed = (float*)e->data;
    ASSERT_NEAR(ed[0], expf(1.0f), 1e-5f, "exp(1)");
    ASSERT_NEAR(ed[1], expf(2.0f), 1e-5f, "exp(2)");
    ASSERT_NEAR(ed[2], expf(3.0f), 1e-5f, "exp(3)");
    ArixTensor* l = arix_tensor_log(e);
    ASSERT(l != NULL, "log result not null");
    float* ld = (float*)l->data;
    ASSERT_NEAR(ld[0], 1.0f, 1e-4f, "log(exp(1))==1");
    ASSERT_NEAR(ld[1], 2.0f, 1e-4f, "log(exp(2))==2");
    ASSERT_NEAR(ld[2], 3.0f, 1e-4f, "log(exp(3))==3");
    ArixTensor* s = arix_tensor_sqrt(a);
    ASSERT(s != NULL, "sqrt result not null");
    float* sd = (float*)s->data;
    ASSERT_NEAR(sd[0], 1.0f, 1e-5f, "sqrt(1)==1");
    ASSERT_NEAR(sd[1], sqrtf(2.0f), 1e-5f, "sqrt(2)");
    ASSERT_NEAR(sd[2], sqrtf(3.0f), 1e-5f, "sqrt(3)");
    arix_tensor_destroy(a); arix_tensor_destroy(e); arix_tensor_destroy(l); arix_tensor_destroy(s);
}

static void test_round_ops(void) {
    size_t shape[] = {4};
    float vals[] = {1.4f, 1.6f, -1.4f, -1.6f};
    ArixTensor* a = arix_tensor_create(shape, 1, ARIX_FLOAT32);
    memcpy(a->data, vals, 4 * sizeof(float));
    ArixTensor* f = arix_tensor_floor(a);
    float* fd = (float*)f->data;
    ASSERT_NEAR(fd[0], 1.0f, 1e-6f, "floor(1.4)==1");
    ASSERT_NEAR(fd[2], -2.0f, 1e-6f, "floor(-1.4)==-2");
    ArixTensor* c = arix_tensor_ceil(a);
    float* cd = (float*)c->data;
    ASSERT_NEAR(cd[1], 2.0f, 1e-6f, "ceil(1.6)==2");
    ASSERT_NEAR(cd[2], -1.0f, 1e-6f, "ceil(-1.4)==-1");
    arix_tensor_destroy(a); arix_tensor_destroy(f); arix_tensor_destroy(c);
}

static void test_trig_ops(void) {
    size_t shape[] = {1};
    float pi = 3.14159265f;
    ArixTensor* a = arix_tensor_full(NULL, 0, ARIX_FLOAT32, &pi);
    ArixTensor* s = arix_tensor_sin(a);
    ASSERT(s != NULL, "sin result not null");
    ASSERT_NEAR(((float*)s->data)[0], 0.0f, 1e-5f, "sin(pi)==0");
    ArixTensor* c = arix_tensor_cos(a);
    ASSERT(c != NULL, "cos result not null");
    ASSERT_NEAR(((float*)c->data)[0], -1.0f, 1e-5f, "cos(pi)==-1");
    float half_pi = pi / 2.0f;
    ArixTensor* h = arix_tensor_full(NULL, 0, ARIX_FLOAT32, &half_pi);
    ArixTensor* sn = arix_tensor_sin(h);
    ASSERT_NEAR(((float*)sn->data)[0], 1.0f, 1e-5f, "sin(pi/2)==1");
    arix_tensor_destroy(a); arix_tensor_destroy(s); arix_tensor_destroy(c);
    arix_tensor_destroy(h); arix_tensor_destroy(sn);
}

static void test_tanh_sigmoid(void) {
    size_t shape[] = {2};
    float vals[] = {0.0f, 1.0f};
    ArixTensor* a = arix_tensor_create(shape, 1, ARIX_FLOAT32);
    memcpy(a->data, vals, 2 * sizeof(float));
    ArixTensor* t = arix_tensor_tanh(a);
    ASSERT(t != NULL, "tanh result not null");
    ASSERT_NEAR(((float*)t->data)[0], 0.0f, 1e-5f, "tanh(0)==0");
    ASSERT_NEAR(((float*)t->data)[1], tanhf(1.0f), 1e-5f, "tanh(1)");
    arix_tensor_destroy(a); arix_tensor_destroy(t);
}

/* ---------- Comparison Ops ---------- */

static void test_eq_basic(void) {
    size_t shape[] = {3};
    ArixTensor* a = arix_tensor_arange(1.0f, 4.0f, 1.0f, ARIX_FLOAT32);
    ArixTensor* r = arix_tensor_eq(a, a);
    ASSERT(r != NULL, "eq result not null");
    uint8_t* d = (uint8_t*)r->data;
    ASSERT(d[0] == 1, "eq self true");
    ASSERT(r->dtype == ARIX_BOOL, "eq dtype bool");
    arix_tensor_destroy(a); arix_tensor_destroy(r);
}

static void test_lt_basic(void) {
    size_t shape[] = {3};
    float vals[] = {3.0f, 3.0f, 3.0f};
    ArixTensor* a = arix_tensor_arange(1.0f, 4.0f, 1.0f, ARIX_FLOAT32);
    ArixTensor* b = arix_tensor_create(shape, 1, ARIX_FLOAT32);
    memcpy(b->data, vals, 3 * sizeof(float));
    ArixTensor* r = arix_tensor_lt(a, b);
    ASSERT(r != NULL, "lt result not null");
    uint8_t* d = (uint8_t*)r->data;
    ASSERT(d[0] == 1, "1<3 true");
    ASSERT(d[2] == 0, "3<3 false");
    arix_tensor_destroy(a); arix_tensor_destroy(b); arix_tensor_destroy(r);
}

static void test_gt_basic(void) {
    size_t shape[] = {3};
    float vals[] = {2.0f, 2.0f, 2.0f};
    ArixTensor* a = arix_tensor_arange(1.0f, 4.0f, 1.0f, ARIX_FLOAT32);
    ArixTensor* b = arix_tensor_create(shape, 1, ARIX_FLOAT32);
    memcpy(b->data, vals, 3 * sizeof(float));
    ArixTensor* r = arix_tensor_gt(a, b);
    ASSERT(r != NULL, "gt result not null");
    uint8_t* d = (uint8_t*)r->data;
    ASSERT(d[0] == 0, "1>2 false");
    ASSERT(d[2] == 1, "3>2 true");
    arix_tensor_destroy(a); arix_tensor_destroy(b); arix_tensor_destroy(r);
}

static void test_null_inputs(void) {
    ASSERT(arix_tensor_add(NULL, NULL) == NULL, "add null returns NULL");
    ASSERT(arix_tensor_neg(NULL) == NULL, "neg null returns NULL");
    ASSERT(arix_tensor_abs(NULL) == NULL, "abs null returns NULL");
    ASSERT(arix_tensor_sin(NULL) == NULL, "sin null returns NULL");
    ASSERT(arix_tensor_eq(NULL, NULL) == NULL, "eq null returns NULL");
    ASSERT(arix_tensor_div(NULL, NULL) == NULL, "div null returns NULL");
}

int main(void) {
    run_test("test_add_basic", test_add_basic);
    run_test("test_add_broadcast", test_add_broadcast);
    run_test("test_sub_basic", test_sub_basic);
    run_test("test_mul_basic", test_mul_basic);
    run_test("test_div_basic", test_div_basic);
    run_test("test_pow_basic", test_pow_basic);
    run_test("test_neg_basic", test_neg_basic);
    run_test("test_abs_basic", test_abs_basic);
    run_test("test_sign_basic", test_sign_basic);
    run_test("test_exp_log_sqrt", test_exp_log_sqrt);
    run_test("test_round_ops", test_round_ops);
    run_test("test_trig_ops", test_trig_ops);
    run_test("test_tanh_sigmoid", test_tanh_sigmoid);
    run_test("test_eq_basic", test_eq_basic);
    run_test("test_lt_basic", test_lt_basic);
    run_test("test_gt_basic", test_gt_basic);
    run_test("test_null_inputs", test_null_inputs);

    printf("\n%d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
