#include "multidimensional_tensor_engine.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

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

static void test_create_2d(void) {
    size_t shape[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    ASSERT(t != NULL, "tensor not null");
    ASSERT(t->ndim == 2, "ndim == 2");
    ASSERT(t->shape[0] == 2, "shape[0] == 2");
    ASSERT(t->shape[1] == 3, "shape[1] == 3");
    ASSERT(t->size == 6, "size == 6");
    SNEPPX_tensor_destroy(t);
}

static void test_zeros(void) {
    size_t shape[] = {2, 2};
    SNEPPXTensor* t = SNEPPX_tensor_zeros(shape, 2, SNEPPX_FLOAT32);
    ASSERT(t != NULL, "tensor not null");
    float* data = (float*)t->data;
    for (size_t i = 0; i < 4; i++) {
        ASSERT(data[i] == 0.0f, "all zeros");
    }
    SNEPPX_tensor_destroy(t);
}

static void test_ones(void) {
    size_t shape[] = {2, 2};
    SNEPPXTensor* t = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    ASSERT(t != NULL, "tensor not null");
    float* data = (float*)t->data;
    for (size_t i = 0; i < 4; i++) {
        ASSERT(data[i] == 1.0f, "all ones");
    }
    SNEPPX_tensor_destroy(t);
}

static void test_set_get(void) {
    size_t shape[] = {3, 3};
    SNEPPXTensor* t = SNEPPX_tensor_zeros(shape, 2, SNEPPX_FLOAT32);
    ASSERT(t != NULL, "tensor not null");
    size_t indices[] = {1, 2};
    SNEPPX_tensor_set_f32(t, indices, 3.14f);
    float val = SNEPPX_tensor_get_f32(t, indices);
    ASSERT_NEAR(val, 3.14f, 1e-4f, "value at [1,2] == 3.14");
    SNEPPX_tensor_destroy(t);
}

static void test_randn_range(void) {
    size_t shape[] = {100};
    SNEPPXTensor* t = SNEPPX_tensor_randn(shape, 1, SNEPPX_FLOAT32);
    ASSERT(t != NULL, "tensor not null");
    float* data = (float*)t->data;
    float min_val = data[0], max_val = data[0];
    for (size_t i = 0; i < 100; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    ASSERT(min_val > -5.0f, "min > -5.0");
    ASSERT(max_val < 5.0f, "max < 5.0");
    SNEPPX_tensor_destroy(t);
}

static void test_print(void) {
    size_t shape[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    ASSERT(t != NULL, "tensor not null");
    SNEPPX_tensor_print(t);
    SNEPPX_tensor_destroy(t);
}

int main(void) {
    run_test("test_create_2d", test_create_2d);
    run_test("test_zeros", test_zeros);
    run_test("test_ones", test_ones);
    run_test("test_set_get", test_set_get);
    run_test("test_randn_range", test_randn_range);
    run_test("test_print", test_print);

    printf("\n%d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
