#include "hierarchical_state_space.h"
#include "sparse_expert_routing.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

static int tests_passed = 0;
static int tests_failed = 0;
#define ASSERT(cond, msg) do { if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } } while(0)

static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout);
    fn(); printf("PASS\n"); tests_passed++;
}

static void test_hss_ser_stack(void) {
    SNEPPXHSSConfig hss_cfg = SNEPPX_hss_config_default();
    hss_cfg.state_dim = 16; hss_cfg.input_dim = 32; hss_cfg.output_dim = 32;
    hss_cfg.num_layers = 1; hss_cfg.use_hierarchical = 0;

    SNEPPXHSSModel* hss = SNEPPX_hss_model_create(&hss_cfg, 42);
    ASSERT(hss != NULL, "hss model not null");

    SNEPPXSERConfig ser_cfg = SNEPPX_ser_config_default();
    ser_cfg.num_experts = 4; ser_cfg.num_active = 2; ser_cfg.input_dim = 32;
    ser_cfg.expert_dim = 64; ser_cfg.output_dim = 32;

    SNEPPXSERModel* ser = SNEPPX_ser_model_create(&ser_cfg, 99, 1);
    ASSERT(ser != NULL, "ser model not null");

    size_t shape_in[] = {4, 16, 32};
    SNEPPXTensor* input = SNEPPX_tensor_randn(shape_in, 3, SNEPPX_FLOAT32);
    ASSERT(input != NULL, "input not null");

    SNEPPXTensor* hss_out = NULL;
    int ret = SNEPPX_hss_forward(hss, input, &hss_out);
    ASSERT(ret == 0, "hss forward ok");
    ASSERT(hss_out != NULL, "hss output not null");
    ASSERT(hss_out->shape[0] == 4 && hss_out->shape[1] == 16 && hss_out->shape[2] == 32, "hss output shape");

    size_t merged = 4 * 16;
    SNEPPXTensor flat_input;
    size_t flat_shape[] = {merged, 32};
    flat_input.data = hss_out->data;
    flat_input.shape = flat_shape;
    flat_input.ndim = 2;
    flat_input.size = merged * 32;
    flat_input.item_size = sizeof(float);
    flat_input.dtype = SNEPPX_FLOAT32;
    flat_input.strides = NULL;

    SNEPPXTensor* ser_out = NULL;
    SNEPPX_ser_forward(ser->layers[0], &flat_input, &ser_out);
    ASSERT(ser_out != NULL, "ser output not null");
    ASSERT(ser_out->shape[0] == merged, "ser output tokens");
    ASSERT(ser_out->shape[1] == 32, "ser output dim");

    float* od = (float*)ser_out->data;
    int ok = 1;
    for (size_t i = 0; i < ser_out->size; i++) {
        if (!isfinite(od[i])) { ok = 0; break; }
    }
    ASSERT(ok, "all finite in ser output");

    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(hss_out);
    SNEPPX_tensor_destroy(ser_out);
    SNEPPX_hss_model_destroy(hss);
    SNEPPX_ser_model_destroy(ser);
}

int main(void) {
    run_test("test_hss_ser_stack", test_hss_ser_stack);
    printf("\nIntegration tests: %d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
