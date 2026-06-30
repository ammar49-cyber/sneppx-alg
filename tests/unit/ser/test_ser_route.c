#include "sparse_expert_routing.h"
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

static void test_route_shapes(void) {
    ArixSERConfig cfg = arix_ser_config_default();
    cfg.num_experts = 4; cfg.num_active = 2; cfg.input_dim = 16; cfg.expert_dim = 32; cfg.output_dim = 16;
    ArixSERLayer* layer = arix_ser_layer_create(&cfg, 42);
    ASSERT(layer != NULL, "layer not null");

    size_t shape_in[] = {8, 16};
    ArixTensor* input = arix_tensor_randn(shape_in, 2, ARIX_FLOAT32);
    ASSERT(input != NULL, "input not null");

    ArixTensor* gw = NULL;
    int* ei = NULL;
    arix_ser_route(layer, input, &gw, &ei);
    ASSERT(gw != NULL, "gate_weights not null");
    ASSERT(ei != NULL, "expert_indices not null");
    ASSERT(gw->shape[0] == 8, "gw tokens == 8");
    ASSERT(gw->shape[1] == 2, "gw active == 2");

    arix_tensor_destroy(input);
    arix_tensor_destroy(gw);
    free(ei);
    arix_ser_layer_destroy(layer);
}

static void test_route_topk(void) {
    ArixSERConfig cfg = arix_ser_config_default();
    cfg.num_experts = 4; cfg.num_active = 2; cfg.input_dim = 16; cfg.expert_dim = 32; cfg.output_dim = 16;
    ArixSERLayer* layer = arix_ser_layer_create(&cfg, 42);
    ASSERT(layer != NULL, "layer not null");

    size_t shape_in[] = {8, 16};
    ArixTensor* input = arix_tensor_randn(shape_in, 2, ARIX_FLOAT32);
    ASSERT(input != NULL, "input not null");

    ArixTensor* gw = NULL;
    int* ei = NULL;
    arix_ser_route(layer, input, &gw, &ei);
    ASSERT(gw != NULL, "gw not null");

    float* gw_data = (float*)gw->data;
    int in_range = 1;
    float sum_ok = 1;
    for (size_t t = 0; t < 8; t++) {
        float sum = 0.0f;
        for (size_t k = 0; k < 2; k++) {
            int idx = ei[t * 2 + k];
            if (idx < 0 || idx >= 4) in_range = 0;
            sum += gw_data[t * 2 + k];
        }
        if (fabsf(sum - 1.0f) > 0.01f) sum_ok = 0;
    }
    ASSERT(in_range, "all indices in [0,3]");
    ASSERT(sum_ok, "gate weights sum to ~1");

    arix_tensor_destroy(input);
    arix_tensor_destroy(gw);
    free(ei);
    arix_ser_layer_destroy(layer);
}

int main(void) {
    run_test("test_route_shapes", test_route_shapes);
    run_test("test_route_topk", test_route_topk);
    printf("\nRoute tests: %d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
