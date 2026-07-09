#include "sparse_expert_routing.h"
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

static void test_ser_config_default(void) {
    SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
    ASSERT(cfg.num_experts == 8, "default has 8 experts");
    ASSERT(cfg.num_active == 2, "default 2 active experts");
    ASSERT(cfg.top_k_method == SNEPPX_TOPK_GREEDY, "default greedy top-k");
}

static void test_ser_load_balance_loss(void) {
    SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
    cfg.num_experts = 4;
    cfg.num_active = 2;
    cfg.input_dim = 8;
    cfg.expert_dim = 16;
    cfg.output_dim = 8;
    SNEPPXSERLayer* layer = SNEPPX_ser_layer_create(&cfg, 42);
    ASSERT(layer != NULL, "layer created");

    size_t shape_in[] = {4, cfg.input_dim};
    SNEPPXTensor* input = SNEPPX_tensor_create(shape_in, 2, SNEPPX_FLOAT32);
    float* d = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) d[i] = ((float)(i % 5) - 2.0f) * 0.1f;

    SNEPPXTensor* gate_weights = NULL;
    int* expert_indices = NULL;
    SNEPPX_ser_route(layer, input, &gate_weights, &expert_indices);
    ASSERT(gate_weights != NULL, "gate weights computed");
    ASSERT(expert_indices != NULL, "expert indices computed");

    float lb_loss = SNEPPX_ser_load_balance_loss(gate_weights, expert_indices, 4);
    ASSERT(lb_loss >= 0.0f, "load balance loss >= 0");

    SNEPPX_tensor_destroy(gate_weights);
    free(expert_indices);
    SNEPPX_tensor_destroy(input);
    SNEPPX_ser_layer_destroy(layer);
}

static void test_ser_z_loss(void) {
    SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
    cfg.num_experts = 4;
    cfg.num_active = 2;
    SNEPPXSERLayer* layer = SNEPPX_ser_layer_create(&cfg, 42);

    size_t shape_in[] = {2, cfg.input_dim};
    SNEPPXTensor* input = SNEPPX_tensor_create(shape_in, 2, SNEPPX_FLOAT32);
    float* d = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) d[i] = 0.5f;

    SNEPPXTensor* gate_weights = NULL;
    int* expert_indices = NULL;
    SNEPPX_ser_route(layer, input, &gate_weights, &expert_indices);

    float zl = SNEPPX_ser_z_loss(gate_weights);
    ASSERT(zl >= 0.0f, "z-loss >= 0");

    SNEPPX_tensor_destroy(gate_weights);
    free(expert_indices);
    SNEPPX_tensor_destroy(input);
    SNEPPX_ser_layer_destroy(layer);
}

int main(void) {
    run_test("ser_config_default", test_ser_config_default);
    run_test("ser_load_balance_loss", test_ser_load_balance_loss);
    run_test("ser_z_loss", test_ser_z_loss);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
