#include "automatic_differentiation_framework.h"
#include "polymorphic_memory_allocator.h"
#include <stdio.h>
#include <math.h>

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
    if (fabsf((float)(a) - (float)(b)) > (eps)) { \
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

static void test_fake_quant_forward_clamp(void) {
    size_t shape[] = {4};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT32);
    ASSERT(t != NULL, "tensor created");
    float* d = (float*)t->data;
    d[0] = 1.4f;   /* rounds to 1 */
    d[1] = -0.6f;  /* rounds to -1 */
    d[2] = 200.0f; /* clamps to qmax */
    d[3] = -200.0f;/* clamps to qmin */
    SNEPPXVariable* v = SNEPPX_variable_create(t, 0);
    ASSERT(v != NULL, "variable created");

    /* scale=1.0, bits=8 -> q in [-128,127], y = round(x) */
    SNEPPXTensor* q = SNEPPX_tensor_fake_quant(v->data, 1.0f, 8);
    ASSERT(q != NULL, "fake_quant tensor");
    float* qd = (float*)q->data;
    ASSERT_NEAR(qd[0], 1.0f, 1e-6f, "1.4 -> 1");
    ASSERT_NEAR(qd[1], -1.0f, 1e-6f, "-0.6 -> -1");
    ASSERT_NEAR(qd[2], 127.0f, 1e-6f, "200 clamps to 127");
    ASSERT_NEAR(qd[3], -128.0f, 1e-6f, "-200 clamps to -128");

    SNEPPX_tensor_destroy(q);
    SNEPPX_variable_destroy(v);
}

static void test_fake_quant_ste_backward(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    ASSERT(tape != NULL, "tape created");
    size_t shape[] = {3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(shape, 1, SNEPPX_FLOAT32);
    ASSERT(t != NULL, "tensor created");
    float* d = (float*)t->data;
    d[0] = 50.0f;  /* in-range with scale 0.5, bits 8 -> q=100 -> 50.0 */
    d[1] = 1.0f;
    d[2] = -1.0f;
    SNEPPXVariable* v = SNEPPX_variable_create(t, 1);
    ASSERT(v != NULL, "variable created (requires_grad)");

    SNEPPXVariable* q = SNEPPX_fake_quant(tape, v, 0.5f, 8);
    ASSERT(q != NULL, "fake_quant op created");
    ASSERT(q->backward_fn != NULL, "backward_fn set for fake_quant");

    /* seed gradient = ones on output -> STE: input grad = ones (identity) */
    SNEPPX_tape_backward(tape, q);
    ASSERT(v->grad != NULL, "gradient computed for input (STE)");
    float* gd = (float*)v->grad->data;
    ASSERT_NEAR(gd[0], 1.0f, 1e-6f, "STE grad identity for elem 0");
    ASSERT_NEAR(gd[1], 1.0f, 1e-6f, "STE grad identity for elem 1");
    ASSERT_NEAR(gd[2], 1.0f, 1e-6f, "STE grad identity for elem 2");

    /* tape owns q (recorded by the op); v is caller-owned */
    SNEPPX_tape_destroy(tape);
    SNEPPX_variable_destroy(v);
}

static void test_fake_quant_invalid_args(void) {
    size_t shape[] = {2};
    SNEPPXTensor* t = SNEPPX_tensor_ones(shape, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* v = SNEPPX_variable_create(t, 1);
    ASSERT(v != NULL, "variable created");
    /* zero scale rejected */
    ASSERT(SNEPPX_tensor_fake_quant(v->data, 0.0f, 8) == NULL, "zero scale rejected");
    /* too-small bits rejected */
    ASSERT(SNEPPX_tensor_fake_quant(v->data, 1.0f, 1) == NULL, "bits<2 rejected");
    /* op-level guard: null input rejected */
    SNEPPXVariable* bad = SNEPPX_fake_quant(NULL, NULL, 1.0f, 8);
    ASSERT(bad == NULL, "null input rejected");
    SNEPPX_variable_destroy(v);
}

int main(void) {
    run_test("fake_quant_forward_clamp", test_fake_quant_forward_clamp);
    run_test("fake_quant_ste_backward", test_fake_quant_ste_backward);
    run_test("fake_quant_invalid_args", test_fake_quant_invalid_args);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
