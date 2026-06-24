#include "arix_ser.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s (%s)\n", msg, #cond); \
        tests_failed++; \
        return; \
    } \
} while(0)

static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout);
    fn(); printf("PASS\n"); tests_passed++;
}

static void test_layer_forward(void) {
    ArixSERConfig cfg = arix_ser_config_default();
    cfg.num_experts = 4; cfg.num_active = 2; cfg.input_dim = 16;
    cfg.expert_dim = 32; cfg.output_dim = 16;
    ArixSERLayer* layer = arix_ser_layer_create(&cfg, 42);
    ASSERT(layer != NULL, "layer not null");

    size_t shape_in[] = {16, 16};
    ArixTensor* input = arix_tensor_randn(shape_in, 2, ARIX_FLOAT32);
    ASSERT(input != NULL, "input not null");

    ArixTensor* output = NULL;
    arix_ser_forward(layer, input, &output);
    ASSERT(output != NULL, "output not null");
    ASSERT(output->shape[0] == 16, "output tokens == 16");
    ASSERT(output->shape[1] == 16, "output dim == 16");

    float* od = (float*)output->data;
    int ok = 1;
    for (size_t i = 0; i < 16 * 16; i++) {
        if (!isfinite(od[i])) { ok = 0; break; }
    }
    ASSERT(ok, "all finite");

    arix_tensor_destroy(input);
    arix_tensor_destroy(output);
    arix_ser_layer_destroy(layer);
}

static void test_layer_load_balance(void) {
    ArixSERConfig cfg = arix_ser_config_default();
    cfg.num_experts = 4; cfg.num_active = 2; cfg.input_dim = 16;
    cfg.expert_dim = 32; cfg.output_dim = 16;
    ArixSERLayer* layer = arix_ser_layer_create(&cfg, 42);
    ASSERT(layer != NULL, "layer not null");

    size_t shape_in[] = {16, 16};
    ArixTensor* input = arix_tensor_randn(shape_in, 2, ARIX_FLOAT32);
    ASSERT(input != NULL, "input not null");

    ArixTensor* gw = NULL;
    int* ei = NULL;
    arix_ser_route(layer, input, &gw, &ei);
    ASSERT(gw != NULL, "gw not null");

    float loss = arix_ser_load_balance_loss(gw, ei, 16);
    ASSERT(isfinite(loss), "loss is finite");
    ASSERT(loss >= 0.0f, "loss >= 0");

    arix_tensor_destroy(input);
    arix_tensor_destroy(gw);
    free(ei);
    arix_ser_layer_destroy(layer);
}

int main(void) {
    run_test("test_layer_forward", test_layer_forward);
    run_test("test_layer_load_balance", test_layer_load_balance);
    printf("\nLayer tests: %d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
