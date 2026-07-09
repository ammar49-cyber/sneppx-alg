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

static void test_tape_create_destroy(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    ASSERT(tape != NULL, "tape created");
    SNEPPX_tape_destroy(tape);
}

static void test_variable_create(void) {
    size_t shape[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    ASSERT(t != NULL, "tensor created");
    SNEPPXVariable* v = SNEPPX_variable_create(t, 1);
    ASSERT(v != NULL, "variable created");
    ASSERT(v->requires_grad == 1, "requires_grad set");
    SNEPPX_variable_destroy(v);
}

static void test_tape_record_and_backward(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    size_t shape[] = {2, 2};
    SNEPPXTensor* a_t = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(a_t, 1);
    SNEPPX_tape_record(tape, a);

    SNEPPXTensor* b_t = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXVariable* b = SNEPPX_variable_create(b_t, 1);
    SNEPPX_tape_record(tape, b);

    SNEPPXVariable* c = SNEPPX_add(tape, a, b);
    ASSERT(c != NULL, "add op created");

    SNEPPXVariable* d = SNEPPX_mul(tape, c, a);
    ASSERT(d != NULL, "mul op created");

    SNEPPX_tape_backward(tape, d);
    ASSERT(a->grad != NULL, "gradient computed for a");
    ASSERT(b->grad != NULL, "gradient computed for b");

    SNEPPX_variable_destroy(d);
    SNEPPX_variable_destroy(c);
    SNEPPX_variable_destroy(b);
    SNEPPX_variable_destroy(a);
    SNEPPX_tape_destroy(tape);
}

static void test_tape_ops(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    size_t shape[] = {3, 1};
    SNEPPXTensor* t = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXVariable* v = SNEPPX_variable_create(t, 1);
    SNEPPX_tape_record(tape, v);

    SNEPPXVariable* r1 = SNEPPX_relu(tape, v);
    ASSERT(r1 != NULL, "relu op");
    SNEPPXVariable* r2 = SNEPPX_sigmoid(tape, v);
    ASSERT(r2 != NULL, "sigmoid op");
    SNEPPXVariable* r3 = SNEPPX_exp(tape, v);
    ASSERT(r3 != NULL, "exp op");
    SNEPPXVariable* r4 = SNEPPX_log(tape, v);
    ASSERT(r4 != NULL, "log op");
    SNEPPXVariable* r5 = SNEPPX_neg(tape, v);
    ASSERT(r5 != NULL, "neg op");

    SNEPPX_tape_backward(tape, r5);
    ASSERT(v->grad != NULL, "grad computed after sequence");

    SNEPPX_variable_destroy(r5);
    SNEPPX_variable_destroy(r4);
    SNEPPX_variable_destroy(r3);
    SNEPPX_variable_destroy(r2);
    SNEPPX_variable_destroy(r1);
    SNEPPX_variable_destroy(v);
    SNEPPX_tape_destroy(tape);
}

int main(void) {
    run_test("tape_create_destroy", test_tape_create_destroy);
    run_test("variable_create", test_variable_create);
    run_test("tape_record_and_backward", test_tape_record_and_backward);
    run_test("tape_ops", test_tape_ops);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
