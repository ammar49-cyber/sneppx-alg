#include "fractal_memory_orchestrator.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

static int tests_passed = 0, tests_failed = 0;
#define ASSERT(cond, msg) do { if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } } while(0)
static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout); fn(); printf("PASS\n"); tests_passed++;
}

static void test_sync_all_reduce(void) {
    SNEPPXFMConfig cfg = SNEPPX_fm_config_default();
    cfg.num_nodes = 4;
    cfg.memory_dim = 4;
    cfg.memory_capacity = 8;
    cfg.sync_interval = 1000;
    cfg.privacy_epsilon = 100.0f;

    SNEPPXFMController* ctrl = SNEPPX_fm_controller_create(&cfg);
    ASSERT(ctrl != NULL, "ctrl not null");

    size_t sh[] = {4};
    for (size_t n = 0; n < 4; n++) {
        SNEPPXTensor* key = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
        SNEPPXTensor* val = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
        ((float*)key->data)[0] = 1.0f;
        ((float*)val->data)[0] = (float)(n * 10 + 1);
        SNEPPX_fm_memory_bank_write(ctrl->nodes[n]->memory_bank, key, val);
        SNEPPX_tensor_destroy(key);
        SNEPPX_tensor_destroy(val);
    }

    int r = SNEPPX_fm_sync_all_reduce(ctrl);
    ASSERT(r == 0, "sync ok");

    float* v0 = (float*)ctrl->nodes[0]->memory_bank->values->data;
    float* v1 = (float*)ctrl->nodes[1]->memory_bank->values->data;
    ASSERT(fabsf(v0[0] - v1[0]) < 1.0f, "nodes converged after sync");

    SNEPPX_fm_controller_destroy(ctrl);
}

static void test_sync_gossip(void) {
    SNEPPXFMConfig cfg = SNEPPX_fm_config_default();
    cfg.num_nodes = 4;
    cfg.memory_dim = 2;
    cfg.memory_capacity = 4;

    SNEPPXFMController* ctrl = SNEPPX_fm_controller_create(&cfg);
    size_t sh[] = {2};
    for (size_t n = 0; n < 4; n++) {
        SNEPPXTensor* key = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
        SNEPPXTensor* val = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
        ((float*)key->data)[0] = 1.0f;
        ((float*)val->data)[0] = (float)(n * 5);
        SNEPPX_fm_memory_bank_write(ctrl->nodes[n]->memory_bank, key, val);
        SNEPPX_tensor_destroy(key);
        SNEPPX_tensor_destroy(val);
    }

    for (int i = 0; i < 10; i++) {
        SNEPPX_fm_sync_gossip(ctrl, 3);
    }

    float v0 = ((float*)ctrl->nodes[0]->memory_bank->values->data)[0];
    float v1 = ((float*)ctrl->nodes[1]->memory_bank->values->data)[0];
    ASSERT(fabsf(v0 - v1) < 5.0f, "gossip convergence");

    SNEPPX_fm_controller_destroy(ctrl);
}

static void test_compress_gradients(void) {
    size_t sh[] = {100};
    SNEPPXTensor* grad = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    float* gd = (float*)grad->data;
    for (size_t i = 0; i < 100; i++) gd[i] = (float)(i % 10);

    SNEPPXTensor* compressed = SNEPPX_fm_compress_gradients(grad, 0.1f);
    ASSERT(compressed != NULL, "compressed not null");
    float* cd = (float*)compressed->data;

    int non_zero = 0;
    for (size_t i = 0; i < 100; i++) {
        if (fabsf(cd[i]) > 1e-6f) non_zero++;
    }
    ASSERT(non_zero <= 12, "at most ~10 non-zero");

    SNEPPX_tensor_destroy(grad);
    SNEPPX_tensor_destroy(compressed);
}

int main(void) {
    run_test("test_sync_all_reduce", test_sync_all_reduce);
    run_test("test_sync_gossip", test_sync_gossip);
    run_test("test_compress_gradients", test_compress_gradients);
    printf("\nSync tests: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
