#include "inference_engine.h"
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

static void test_inference_engine_create(void) {
    SNEPPXInferenceEngine* engine = SNEPPX_inference_engine_create(42);
    ASSERT(engine != NULL, "engine created");
    SNEPPX_inference_engine_destroy(engine);
}

static void test_inference_engine_run(void) {
    SNEPPXInferenceEngine* engine = SNEPPX_inference_engine_create(42);
    size_t shape[] = {1, 8};
    SNEPPXTensor* input = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* output = SNEPPX_inference_engine_run(engine, input);
    ASSERT(output != NULL, "inference output");
    ASSERT(output->size > 0, "output has data");
    SNEPPX_tensor_destroy(output);
    SNEPPX_tensor_destroy(input);
    SNEPPX_inference_engine_destroy(engine);
}

static void test_inference_engine_reset(void) {
    SNEPPXInferenceEngine* engine = SNEPPX_inference_engine_create(42);
    SNEPPX_inference_engine_reset(engine);
    ASSERT(engine != NULL, "reset does not crash");
    SNEPPX_inference_engine_destroy(engine);
}

int main(void) {
    run_test("inference_engine_create", test_inference_engine_create);
    run_test("inference_engine_run", test_inference_engine_run);
    run_test("inference_engine_reset", test_inference_engine_reset);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
