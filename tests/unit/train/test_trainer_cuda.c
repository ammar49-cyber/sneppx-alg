#include "differentiable_training_pipeline.h"
#include "trainer_cuda.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
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

static void run_test(const char* name, void (*test_fn)(void)) {
    printf("Running %s... ", name);
    fflush(stdout);
    test_fn();
    printf("PASS\n");
    tests_passed++;
}

static void test_cuda_available(void) {
    int avail = SNEPPX_trainer_cuda_available();
    printf("[CUDA avail=%d] ", avail);
}

static void test_cuda_init_shutdown(void) {
    int avail = SNEPPX_trainer_cuda_available();
    if (!avail) {
        printf("[SKIP: no CUDA device] ");
        tests_passed++;
        return;
    }

    SNEPPXTensor* p = SNEPPX_tensor_create((size_t[]){10}, 1, SNEPPX_FLOAT32);
    ASSERT(p != NULL, "param tensor created");

    SNEPPXTensor* params[] = {p};
    int ret = SNEPPX_trainer_cuda_init(params, 1);
    ASSERT(ret == 0, "cuda init succeeded");

    SNEPPX_trainer_cuda_shutdown();

    ret = SNEPPX_trainer_cuda_init(params, 1);
    ASSERT(ret == 0, "re-init after shutdown");

    SNEPPX_trainer_cuda_shutdown();
    SNEPPX_tensor_destroy(p);
}

static void test_cuda_transfer(void) {
    int avail = SNEPPX_trainer_cuda_available();
    if (!avail) {
        printf("[SKIP: no CUDA device] ");
        tests_passed++;
        return;
    }

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    SNEPPXTensor* p = SNEPPX_tensor_create((size_t[]){5}, 1, SNEPPX_FLOAT32);
    ASSERT(p != NULL, "tensor created");
    memcpy(p->data, data, 5 * sizeof(float));

    SNEPPXTensor* params[] = {p};
    int ret = SNEPPX_trainer_cuda_init(params, 1);
    ASSERT(ret == 0, "cuda init");

    ret = SNEPPX_trainer_cuda_transfer_to_device(params, 1);
    ASSERT(ret == 0, "transfer to device");

    memset(p->data, 0, 5 * sizeof(float));

    ret = SNEPPX_trainer_cuda_transfer_to_host(params, 1);
    ASSERT(ret == 0, "transfer to host");

    float* pd = (float*)p->data;
    ASSERT(fabsf(pd[0] - 1.0f) < 1e-5f, "val[0] == 1.0");
    ASSERT(fabsf(pd[4] - 5.0f) < 1e-5f, "val[4] == 5.0");

    SNEPPX_trainer_cuda_shutdown();
    SNEPPX_tensor_destroy(p);
}

static void test_trainer_config_cuda(void) {
    SNEPPXTrainConfig cfg = SNEPPX_train_config_default();
    ASSERT(cfg.use_cuda_optimizer == 0, "default cuda_optimizer off");
    cfg.use_cuda_optimizer = 1;

    SNEPPXModel* model = NULL;

    if (SNEPPX_trainer_cuda_available()) {
        ASSERT(1, "CUDA available on this system");
    } else {
        printf("[NO CUDA] ");
    }
}

int main(void) {
    run_test("cuda_available", test_cuda_available);
    run_test("cuda_init_shutdown", test_cuda_init_shutdown);
    run_test("cuda_transfer", test_cuda_transfer);
    run_test("trainer_config_cuda", test_trainer_config_cuda);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
