#include "fractal_memory_orchestrator.h"
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

static void test_fm_forward_simple(void) {
    SNEPPXFMConfig cfg = SNEPPX_fm_config_default();
    cfg.memory_dim = 8;
    cfg.memory_capacity = 16;
    SNEPPXFMController* ctrl = SNEPPX_fm_controller_create(&cfg);
    ASSERT(ctrl != NULL, "controller created");

    size_t shape[] = {2, cfg.memory_dim};
    SNEPPXTensor* input = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* output = NULL;
    int ret = SNEPPX_fm_forward(ctrl, 0, input, &output);
    ASSERT(ret == 0, "forward returned 0");
    ASSERT(output != NULL, "forward output created");
    ASSERT(output->shape[0] == input->shape[0], "forward batch dim matches");
    ASSERT(output->shape[1] == input->shape[1], "forward feat dim matches");

    SNEPPX_tensor_destroy(output);
    SNEPPX_tensor_destroy(input);
    SNEPPX_fm_controller_destroy(ctrl);
}

static void test_fm_forward_memory_write(void) {
    SNEPPXFMConfig cfg = SNEPPX_fm_config_default();
    cfg.memory_dim = 4;
    cfg.memory_capacity = 4;
    SNEPPXFMController* ctrl = SNEPPX_fm_controller_create(&cfg);

    size_t shape[] = {1, cfg.memory_dim};
    SNEPPXTensor* input = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* output1 = NULL;
    int ret = SNEPPX_fm_forward(ctrl, 0, input, &output1);
    ASSERT(ret == 0, "first forward returned 0");
    ASSERT(output1 != NULL, "first forward");

    SNEPPXTensor* input2 = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* output2 = NULL;
    ret = SNEPPX_fm_forward(ctrl, 0, input2, &output2);
    ASSERT(ret == 0, "second forward returned 0");
    ASSERT(output2 != NULL, "second forward");

    SNEPPX_tensor_destroy(output2);
    SNEPPX_tensor_destroy(input2);
    SNEPPX_tensor_destroy(output1);
    SNEPPX_tensor_destroy(input);
    SNEPPX_fm_controller_destroy(ctrl);
}

static void test_fm_forward_batch(void) {
    SNEPPXFMConfig cfg = SNEPPX_fm_config_default();
    cfg.memory_dim = 4;
    cfg.memory_capacity = 8;
    SNEPPXFMController* ctrl = SNEPPX_fm_controller_create(&cfg);

    size_t shape[] = {3, cfg.memory_dim};
    SNEPPXTensor* input = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    float* d = (float*)input->data;
    for (size_t i = 0; i < 12; i++) d[i] = (float)(i % 4) * 0.25f;

    SNEPPXTensor* output = NULL;
    int ret = SNEPPX_fm_forward(ctrl, 0, input, &output);
    ASSERT(ret == 0, "batch forward returned 0");
    ASSERT(output != NULL, "batch forward created");
    ASSERT(output->shape[0] == 3, "batch dim preserved");

    SNEPPX_tensor_destroy(output);
    SNEPPX_tensor_destroy(input);
    SNEPPX_fm_controller_destroy(ctrl);
}

int main(void) {
    run_test("fm_forward_simple", test_fm_forward_simple);
    run_test("fm_forward_memory_write", test_fm_forward_memory_write);
    run_test("fm_forward_batch", test_fm_forward_batch);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
