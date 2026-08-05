#include "sparse_expert_routing.h"
#include "polymorphic_memory_allocator.h"
#include <stdio.h>
#include <math.h>

/*
 * SNEPPX - Test Ser Mlp Gater
 *
 * WHAT
 *   Test Ser Mlp Gater.
 *
 * CONCEPT
 *   Provides the Test Ser Mlp Gater.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


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

static void test_mlp_gater_config_default(void) {
    SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
    ASSERT(cfg.use_mlp_gater == 0, "default is linear gater");
    ASSERT(cfg.gater_hidden_dim == 64, "default gater hidden dim 64");
}

static void test_mlp_gater_route(void) {
    SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
    cfg.use_mlp_gater = 1;
    cfg.gater_hidden_dim = 16;
    cfg.num_experts = 4;
    cfg.num_active = 2;
    cfg.input_dim = 8;
    cfg.expert_dim = 16;
    cfg.output_dim = 8;
    SNEPPXSERLayer* layer = SNEPPX_ser_layer_create(&cfg, 42);
    ASSERT(layer != NULL, "layer with mlp gater created");
    ASSERT(layer->gater_w1 != NULL, "gater_w1 created");
    ASSERT(layer->gater_b1 != NULL, "gater_b1 created");
    ASSERT(layer->gater_w2 != NULL, "gater_w2 created");
    ASSERT(layer->gater_b2 != NULL, "gater_b2 created");

    size_t shape_in[] = {4, cfg.input_dim};
    SNEPPXTensor* input = SNEPPX_tensor_create(shape_in, 2, SNEPPX_FLOAT32);
    float* d = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) d[i] = ((float)(i % 5) - 2.0f) * 0.1f;

    SNEPPXTensor* gate_weights = NULL;
    int* expert_indices = NULL;
    SNEPPX_ser_route_mlp_gated(layer, input, &gate_weights, &expert_indices);
    ASSERT(gate_weights != NULL, "gate weights computed");
    ASSERT(expert_indices != NULL, "expert indices computed");
    ASSERT(gate_weights->shape[0] == 4, "batch dim correct");
    ASSERT(gate_weights->shape[1] == 2, "active dim correct");
    ASSERT(expert_indices[0] >= 0 && expert_indices[0] < 4, "valid expert index");

    float* gw = (float*)gate_weights->data;
    float row_sum = gw[0] + gw[1];
    ASSERT_NEAR(row_sum, 1.0f, 1e-5f, "gate weights sum to 1");

    SNEPPX_tensor_destroy(gate_weights);
    SNEPPX_free(expert_indices, 4 * 2 * sizeof(int));
    SNEPPX_tensor_destroy(input);
    SNEPPX_ser_layer_destroy(layer);
}

static void test_mlp_gater_linear_vs_mlp_interface(void) {
    SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
    cfg.use_mlp_gater = 0;
    cfg.num_experts = 4;
    cfg.num_active = 2;
    cfg.input_dim = 8;
    cfg.expert_dim = 16;
    cfg.output_dim = 8;
    SNEPPXSERLayer* linear_layer = SNEPPX_ser_layer_create(&cfg, 42);
    ASSERT(linear_layer->gater_w1 == NULL, "linear layer has no gater_w1");

    cfg.use_mlp_gater = 1;
    cfg.gater_hidden_dim = 16;
    SNEPPXSERLayer* mlp_layer = SNEPPX_ser_layer_create(&cfg, 42);
    ASSERT(mlp_layer->gater_w1 != NULL, "mlp layer has gater_w1");

    size_t shape_in[] = {2, cfg.input_dim};
    SNEPPXTensor* input = SNEPPX_tensor_create(shape_in, 2, SNEPPX_FLOAT32);
    float* d = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) d[i] = 0.1f;

    SNEPPXTensor* gw1 = NULL; int* ei1 = NULL;
    SNEPPX_ser_route(linear_layer, input, &gw1, &ei1);
    ASSERT(gw1 != NULL, "linear route works");

    SNEPPXTensor* gw2 = NULL; int* ei2 = NULL;
    SNEPPX_ser_route_mlp_gated(mlp_layer, input, &gw2, &ei2);
    ASSERT(gw2 != NULL, "mlp gated route works");

    SNEPPX_tensor_destroy(gw1);
    SNEPPX_free(ei1, 2 * 2 * sizeof(int));
    SNEPPX_tensor_destroy(gw2);
    SNEPPX_free(ei2, 2 * 2 * sizeof(int));
    SNEPPX_tensor_destroy(input);
    SNEPPX_ser_layer_destroy(linear_layer);
    SNEPPX_ser_layer_destroy(mlp_layer);
}

int main(void) {
    run_test("MLP gater config defaults", test_mlp_gater_config_default);
    run_test("MLP gater route output", test_mlp_gater_route);
    run_test("MLP gater vs linear interface", test_mlp_gater_linear_vs_mlp_interface);

    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
