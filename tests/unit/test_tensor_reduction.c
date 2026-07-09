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

/* ---------- Reduction ---------- */

static SNEPPXTensor* make_arange_2d(void) {
    size_t shape[] = {2, 3};
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    memcpy(t->data, data, 6 * sizeof(float));
    return t;
}

static void test_sum_all(void) {
    SNEPPXTensor* t = make_arange_2d();
    SNEPPXTensor* s = SNEPPX_tensor_sum(t, 0);
    ASSERT(s != NULL, "sum result not null");
    float* d = (float*)s->data;
    ASSERT_NEAR(d[0], 5.0f, 1e-5f, "sum dim0 col0: 1+4==5");
    ASSERT_NEAR(d[1], 7.0f, 1e-5f, "sum dim0 col1: 2+5==7");
    ASSERT_NEAR(d[2], 9.0f, 1e-5f, "sum dim0 col2: 3+6==9");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(s);
}

static void test_sum_dim1(void) {
    SNEPPXTensor* t = make_arange_2d();
    SNEPPXTensor* s = SNEPPX_tensor_sum(t, 1);
    ASSERT(s != NULL, "sum dim1 result not null");
    float* d = (float*)s->data;
    ASSERT_NEAR(d[0], 6.0f, 1e-5f, "sum dim1 row0: 1+2+3==6");
    ASSERT_NEAR(d[1], 15.0f, 1e-5f, "sum dim1 row1: 4+5+6==15");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(s);
}

static void test_mean_basic(void) {
    SNEPPXTensor* t = make_arange_2d();
    SNEPPXTensor* m = SNEPPX_tensor_mean(t, 0);
    ASSERT(m != NULL, "mean result not null");
    float* d = (float*)m->data;
    ASSERT_NEAR(d[0], 2.5f, 1e-5f, "mean col0: (1+4)/2==2.5");
    ASSERT_NEAR(d[1], 3.5f, 1e-5f, "mean col1: (2+5)/2==3.5");
    ASSERT_NEAR(d[2], 4.5f, 1e-5f, "mean col2: (3+6)/2==4.5");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(m);
}

static void test_var_std_basic(void) {
    size_t shape[] = {4};
    float data[] = {2.0f, 4.0f, 4.0f, 6.0f};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT32);
    memcpy(t->data, data, 4 * sizeof(float));
    SNEPPXTensor* v = SNEPPX_tensor_var(t, 0);
    ASSERT(v != NULL, "var result not null");
    float var = ((float*)v->data)[0];
    ASSERT_NEAR(var, 2.0f, 1e-5f, "var(2,4,4,6)==2.0");
    SNEPPXTensor* s = SNEPPX_tensor_std(t, 0);
    ASSERT(s != NULL, "std result not null");
    ASSERT_NEAR(((float*)s->data)[0], sqrtf(2.0f), 1e-5f, "std==sqrt(var)");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(v); SNEPPX_tensor_destroy(s);
}

static void test_min_max_basic(void) {
    size_t shape[] = {5};
    float data[] = {3.0f, -1.0f, 7.0f, 0.0f, -5.0f};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT32);
    memcpy(t->data, data, 5 * sizeof(float));
    ASSERT_NEAR(SNEPPX_tensor_min(t), -5.0f, 1e-6f, "min==-5");
    ASSERT_NEAR(SNEPPX_tensor_max(t), 7.0f, 1e-6f, "max==7");
    SNEPPX_tensor_destroy(t);
}

static void test_argmin_argmax_basic(void) {
    size_t shape[] = {5};
    float data[] = {3.0f, -1.0f, 7.0f, 0.0f, -5.0f};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT32);
    memcpy(t->data, data, 5 * sizeof(float));
    ASSERT(SNEPPX_tensor_argmin(t) == 4, "argmin==4 (-5)");
    ASSERT(SNEPPX_tensor_argmax(t) == 2, "argmax==2 (7)");
    SNEPPX_tensor_destroy(t);
}

static void test_cumsum_basic(void) {
    size_t shape[] = {4};
    SNEPPXTensor* t = SNEPPX_tensor_arange(1.0f, 5.0f, 1.0f, SNEPPX_FLOAT32);
    SNEPPXTensor* c = SNEPPX_tensor_cumsum(t, 0);
    ASSERT(c != NULL, "cumsum result not null");
    float* d = (float*)c->data;
    ASSERT_NEAR(d[0], 1.0f, 1e-6f, "cumsum[0]==1");
    ASSERT_NEAR(d[1], 3.0f, 1e-6f, "cumsum[1]==1+2==3");
    ASSERT_NEAR(d[2], 6.0f, 1e-6f, "cumsum[2]==1+2+3==6");
    ASSERT_NEAR(d[3], 10.0f, 1e-6f, "cumsum[3]==1+2+3+4==10");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(c);
}

static void test_cumprod_basic(void) {
    size_t shape[] = {4};
    SNEPPXTensor* t = SNEPPX_tensor_arange(1.0f, 5.0f, 1.0f, SNEPPX_FLOAT32);
    SNEPPXTensor* c = SNEPPX_tensor_cumprod(t, 0);
    ASSERT(c != NULL, "cumprod result not null");
    float* d = (float*)c->data;
    ASSERT_NEAR(d[0], 1.0f, 1e-6f, "cumprod[0]==1");
    ASSERT_NEAR(d[1], 2.0f, 1e-6f, "cumprod[1]==1*2==2");
    ASSERT_NEAR(d[2], 6.0f, 1e-5f, "cumprod[2]==1*2*3==6");
    ASSERT_NEAR(d[3], 24.0f, 1e-5f, "cumprod[3]==1*2*3*4==24");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(c);
}

/* ---------- Linear Algebra ---------- */

static void test_dot_basic(void) {
    size_t shape[] = {3};
    SNEPPXTensor* a = SNEPPX_tensor_arange(1.0f, 4.0f, 1.0f, SNEPPX_FLOAT32);
    SNEPPXTensor* b = SNEPPX_tensor_arange(1.0f, 4.0f, 1.0f, SNEPPX_FLOAT32);
    float d = SNEPPX_tensor_dot(a, b);
    ASSERT_NEAR(d, 14.0f, 1e-5f, "dot([1,2,3],[1,2,3])==1+4+9==14");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b);
}

static void test_matmul_basic(void) {
    size_t shape_a[] = {2, 3};
    size_t shape_b[] = {3, 2};
    float ad[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float bd[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    SNEPPXTensor* a = SNEPPX_tensor_create(shape_a, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* b = SNEPPX_tensor_create(shape_b, 2, SNEPPX_FLOAT32);
    memcpy(a->data, ad, 6 * sizeof(float));
    memcpy(b->data, bd, 6 * sizeof(float));
    SNEPPXTensor* c = SNEPPX_tensor_matmul(a, b);
    ASSERT(c != NULL, "matmul result not null");
    ASSERT(c->shape[0] == 2 && c->shape[1] == 2, "matmul shape 2x2");
    float* rd = (float*)c->data;
    ASSERT_NEAR(rd[0], 58.0f, 1e-5f, "matmul[0,0]: 1*7+2*9+3*11==58");
    ASSERT_NEAR(rd[1], 64.0f, 1e-5f, "matmul[0,1]: 1*8+2*10+3*12==64");
    ASSERT_NEAR(rd[2], 139.0f, 1e-5f, "matmul[1,0]: 4*7+5*9+6*11==139");
    ASSERT_NEAR(rd[3], 154.0f, 1e-5f, "matmul[1,1]: 4*8+5*10+6*12==154");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b); SNEPPX_tensor_destroy(c);
}

static void test_transpose_basic(void) {
    size_t shape[] = {2, 3};
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    memcpy(t->data, data, 6 * sizeof(float));
    SNEPPXTensor* tr = SNEPPX_tensor_transpose(t, 0, 1);
    ASSERT(tr != NULL, "transpose result not null");
    ASSERT(tr->shape[0] == 3 && tr->shape[1] == 2, "transpose shape 3x2");
    size_t idx00[] = {0, 0}, idx01[] = {0, 1}, idx20[] = {2, 0}, idx21[] = {2, 1};
    ASSERT_NEAR(SNEPPX_tensor_get_f32(tr, idx00), 1.0f, 1e-6f, "T[0,0]==1");
    ASSERT_NEAR(SNEPPX_tensor_get_f32(tr, idx01), 4.0f, 1e-6f, "T[0,1]==4");
    ASSERT_NEAR(SNEPPX_tensor_get_f32(tr, idx20), 3.0f, 1e-6f, "T[2,0]==3");
    ASSERT_NEAR(SNEPPX_tensor_get_f32(tr, idx21), 6.0f, 1e-6f, "T[2,1]==6");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(tr);
}

static void test_inverse_2x2(void) {
    size_t shape[] = {2, 2};
    float data[] = {4.0f, 7.0f, 2.0f, 6.0f};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    memcpy(t->data, data, 4 * sizeof(float));
    SNEPPXTensor* inv = SNEPPX_tensor_inverse(t);
    ASSERT(inv != NULL, "inverse result not null");
    float* d = (float*)inv->data;
    float det = 4.0f * 6.0f - 7.0f * 2.0f;
    ASSERT_NEAR(d[0], 6.0f / det, 1e-5f, "inv[0,0]==6/det");
    ASSERT_NEAR(d[1], -7.0f / det, 1e-5f, "inv[0,1]==-7/det");
    ASSERT_NEAR(d[2], -2.0f / det, 1e-5f, "inv[1,0]==-2/det");
    ASSERT_NEAR(d[3], 4.0f / det, 1e-5f, "inv[1,1]==4/det");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(inv);
}

static void test_det_2x2(void) {
    size_t shape[] = {2, 2};
    float data[] = {4.0f, 7.0f, 2.0f, 6.0f};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    memcpy(t->data, data, 4 * sizeof(float));
    ASSERT_NEAR(SNEPPX_tensor_det(t), 10.0f, 1e-5f, "det(2x2)==10");
    SNEPPX_tensor_destroy(t);
}

static void test_inverse_null(void) {
    ASSERT(SNEPPX_tensor_inverse(NULL) == NULL, "inverse null returns NULL");
    ASSERT(SNEPPX_tensor_matmul(NULL, NULL) == NULL, "matmul null returns NULL");
    ASSERT(SNEPPX_tensor_dot(NULL, NULL) == 0.0f, "dot null returns 0");
    ASSERT(SNEPPX_tensor_transpose(NULL, 0, 1) == NULL, "transpose null returns NULL");
    ASSERT(SNEPPX_tensor_sum(NULL, 0) == NULL, "sum null returns NULL");
    ASSERT(SNEPPX_tensor_mean(NULL, 0) == NULL, "mean null returns NULL");
    ASSERT(SNEPPX_tensor_min(NULL) == 0.0f, "min null returns 0");
}

int main(void) {
    run_test("test_sum_all", test_sum_all);
    run_test("test_sum_dim1", test_sum_dim1);
    run_test("test_mean_basic", test_mean_basic);
    run_test("test_var_std_basic", test_var_std_basic);
    run_test("test_min_max_basic", test_min_max_basic);
    run_test("test_argmin_argmax_basic", test_argmin_argmax_basic);
    run_test("test_cumsum_basic", test_cumsum_basic);
    run_test("test_cumprod_basic", test_cumprod_basic);
    run_test("test_dot_basic", test_dot_basic);
    run_test("test_matmul_basic", test_matmul_basic);
    run_test("test_transpose_basic", test_transpose_basic);
    run_test("test_inverse_2x2", test_inverse_2x2);
    run_test("test_det_2x2", test_det_2x2);
    run_test("test_inverse_null", test_inverse_null);

    printf("\n%d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
