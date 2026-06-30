#include "hierarchical_state_space.h"
#include <stdio.h>
#include <string.h>
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

static void test_model_create(void) {
    ArixHSSConfig cfg = arix_hss_config_default();
    cfg.state_dim = 4;
    cfg.input_dim = 8;
    cfg.output_dim = 8;
    cfg.num_layers = 2;
    ArixHSSModel* model = arix_hss_model_create(&cfg, 42);
    ASSERT(model != NULL, "model not null");
    ASSERT(model->layers != NULL, "layers array not null");
    ASSERT(model->layers[0] != NULL, "layer 0 not null");
    ASSERT(model->layers[1] != NULL, "layer 1 not null");
    ASSERT(model->config.num_layers == 2, "num_layers == 2");
    ASSERT(model->norm_gamma[0] != NULL, "norm_gamma[0] not null");
    ASSERT(model->norm_beta[0] != NULL, "norm_beta[0] not null");
    arix_hss_model_destroy(model);
}

static void test_forward_small(void) {
    ArixHSSConfig cfg = arix_hss_config_default();
    cfg.state_dim = 4;
    cfg.input_dim = 8;
    cfg.output_dim = 8;
    cfg.num_layers = 1;
    cfg.use_hierarchical = 0;
    ArixHSSModel* model = arix_hss_model_create(&cfg, 42);
    ASSERT(model != NULL, "model not null");

    size_t shape_in[] = {4, 8, 8};
    ArixTensor* input = arix_tensor_randn(shape_in, 3, ARIX_FLOAT32);
    ASSERT(input != NULL, "input not null");

    ArixTensor* output = NULL;
    int ret = arix_hss_forward(model, input, &output);
    ASSERT(ret == 0, "forward returned 0");
    ASSERT(output != NULL, "output not null");
    ASSERT(output->shape[0] == 4, "output batch == 4");
    ASSERT(output->shape[1] == 8, "output seq_len == 8");
    ASSERT(output->shape[2] == 8, "output dim == 8");

    float* od = (float*)output->data;
    int has_nan = 0;
    for (size_t i = 0; i < output->size; i++) {
        if (!isfinite(od[i])) { has_nan = 1; break; }
    }
    ASSERT(!has_nan, "no NaN/inf in output");

    arix_tensor_destroy(input);
    arix_tensor_destroy(output);
    arix_hss_model_destroy(model);
}

static void test_forward_single_seq(void) {
    ArixHSSConfig cfg = arix_hss_config_default();
    cfg.state_dim = 4;
    cfg.input_dim = 8;
    cfg.output_dim = 8;
    cfg.num_layers = 1;
    cfg.use_hierarchical = 0;
    ArixHSSModel* model = arix_hss_model_create(&cfg, 42);
    ASSERT(model != NULL, "model not null");

    size_t shape_in[] = {16, 8};
    ArixTensor* input = arix_tensor_randn(shape_in, 2, ARIX_FLOAT32);
    ASSERT(input != NULL, "input not null");

    ArixTensor* output = NULL;
    int ret = arix_hss_forward(model, input, &output);
    ASSERT(ret == 0, "forward returned 0");
    ASSERT(output != NULL, "output not null");

    float* od = (float*)output->data;
    int has_nan = 0;
    for (size_t i = 0; i < output->size; i++) {
        if (!isfinite(od[i])) { has_nan = 1; break; }
    }
    ASSERT(!has_nan, "no NaN/inf in output");

    arix_tensor_destroy(input);
    arix_tensor_destroy(output);
    arix_hss_model_destroy(model);
}

int main(void) {
    run_test("test_model_create", test_model_create);
    run_test("test_forward_small", test_forward_small);
    run_test("test_forward_single_seq", test_forward_single_seq);
    printf("\nModel tests: %d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
