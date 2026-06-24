#include "arix_ser.h"
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

static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout);
    fn(); printf("PASS\n"); tests_passed++;
}

static void test_model_create(void) {
    ArixSERConfig cfg = arix_ser_config_default();
    cfg.num_experts = 4; cfg.num_active = 2; cfg.input_dim = 16;
    cfg.expert_dim = 32; cfg.output_dim = 16;
    ArixSERModel* model = arix_ser_model_create(&cfg, 42, 2);
    ASSERT(model != NULL, "model not null");
    ASSERT(model->num_layers == 2, "num_layers == 2");
    ASSERT(model->layers[0] != NULL, "layer 0 not null");
    ASSERT(model->layers[1] != NULL, "layer 1 not null");
    ASSERT(model->layers[0]->experts[0] != NULL, "expert 0 not null");
    arix_ser_model_destroy(model);
}

static void test_model_forward(void) {
    ArixSERConfig cfg = arix_ser_config_default();
    cfg.num_experts = 4; cfg.num_active = 2; cfg.input_dim = 16;
    cfg.expert_dim = 32; cfg.output_dim = 16;
    ArixSERModel* model = arix_ser_model_create(&cfg, 42, 1);
    ASSERT(model != NULL, "model not null");

    size_t shape_in[] = {4, 32, 16};
    ArixTensor* input = arix_tensor_randn(shape_in, 3, ARIX_FLOAT32);
    ASSERT(input != NULL, "input not null");

    ArixTensor* output = NULL;
    size_t num_tokens = 4 * 32;
    ArixTensor flat_input;
    size_t flat_shape[] = {num_tokens, 16};
    flat_input.data = input->data;
    flat_input.shape = flat_shape;
    flat_input.ndim = 2;
    flat_input.size = num_tokens * 16;
    flat_input.item_size = sizeof(float);
    flat_input.dtype = ARIX_FLOAT32;
    flat_input.strides = NULL;

    arix_ser_forward(model->layers[0], &flat_input, &output);
    ASSERT(output != NULL, "output not null");
    ASSERT(output->shape[0] == num_tokens, "output tokens");
    ASSERT(output->shape[1] == 16, "output dim == 16");

    float* od = (float*)output->data;
    int ok = 1;
    for (size_t i = 0; i < output->size; i++) {
        if (!isfinite(od[i])) { ok = 0; break; }
    }
    ASSERT(ok, "all finite");

    arix_tensor_destroy(input);
    arix_tensor_destroy(output);
    arix_ser_model_destroy(model);
}

int main(void) {
    run_test("test_model_create", test_model_create);
    run_test("test_model_forward", test_model_forward);
    printf("\nModel tests: %d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
