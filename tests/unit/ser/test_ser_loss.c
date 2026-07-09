#include "sparse_expert_routing.h"
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

static void test_ser_importance_loss(void) {
    SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
    cfg.num_experts = 4;
    SNEPPXSERLayer* layer = SNEPPX_ser_layer_create(&cfg, 42);
    size_t shape[] = {2, cfg.input_dim};
    SNEPPXTensor* input = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    float* d = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) d[i] = 1.0f;
    SNEPPXTensor* gate_weights = NULL;
    int* expert_indices = NULL;
    SNEPPX_ser_route(layer, input, &gate_weights, &expert_indices);
    float imp_loss = SNEPPX_ser_importance_loss(gate_weights, expert_indices, 4);
    ASSERT(imp_loss >= 0.0f, "importance loss >= 0");
    SNEPPX_tensor_destroy(gate_weights);
    free(expert_indices);
    SNEPPX_tensor_destroy(input);
    SNEPPX_ser_layer_destroy(layer);
}

static void test_ser_aux_loss_balanced(void) {
    size_t shape_batch[] = {4, 2};
    float data[8] = {1.0f, 0.0f, 0.5f, 0.5f, 0.0f, 1.0f, 0.3f, 0.7f};
    SNEPPXTensor* gate_weights = SNEPPX_tensor_create(shape_batch, 2, SNEPPX_FLOAT32);
    for (size_t i = 0; i < 8; i++) ((float*)gate_weights->data)[i] = data[i];
    int expert_indices[] = {0, 1, 0, 0};
    float aux = SNEPPX_ser_aux_load_balance(gate_weights, expert_indices, 4, 2);
    ASSERT(aux >= 0.0f, "aux loss >= 0");
    SNEPPX_tensor_destroy(gate_weights);
}

int main(void) {
    run_test("ser_importance_loss", test_ser_importance_loss);
    run_test("ser_aux_loss_balanced", test_ser_aux_loss_balanced);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
