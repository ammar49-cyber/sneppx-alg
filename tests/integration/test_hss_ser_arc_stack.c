#include "hierarchical_state_space.h"
#include "sparse_expert_routing.h"
#include "adversarial_robustness_certification.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

static int tests_passed = 0, tests_failed = 0;
#define ASSERT(cond, msg) do { if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } } while(0)
static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout); fn(); printf("PASS\n"); tests_passed++;
}

static void test_hss_ser_arc_stack(void) {
    ArixHSSConfig hss_cfg = arix_hss_config_default();
    hss_cfg.state_dim = 8; hss_cfg.input_dim = 16; hss_cfg.output_dim = 16;
    hss_cfg.num_layers = 1; hss_cfg.use_hierarchical = 0;
    ArixHSSModel* hss = arix_hss_model_create(&hss_cfg, 42);
    ASSERT(hss != NULL, "hss model");

    ArixSERConfig ser_cfg = arix_ser_config_default();
    ser_cfg.num_experts = 4; ser_cfg.num_active = 2; ser_cfg.input_dim = 16;
    ser_cfg.expert_dim = 32; ser_cfg.output_dim = 16;
    ArixSERModel* ser = arix_ser_model_create(&ser_cfg, 99, 1);
    ASSERT(ser != NULL, "ser model");

    ArixARCConfig arc_cfg = arix_arc_config_default();
    ArixARCLayer* arc = arix_arc_layer_create(&arc_cfg, 16, 16, 200);
    ASSERT(arc != NULL, "arc layer");

    size_t shape_in[] = {2, 8, 16};
    ArixTensor* input = arix_tensor_randn(shape_in, 3, ARIX_FLOAT32);
    ASSERT(input != NULL, "input");

    ArixTensor* hss_out = NULL;
    int ret = arix_hss_forward(hss, input, &hss_out);
    ASSERT(ret == 0 && hss_out != NULL, "hss forward");
    ASSERT(hss_out->shape[0] == 2 && hss_out->shape[1] == 8 && hss_out->shape[2] == 16, "hss shape");

    size_t merged = 2 * 8;
    ArixTensor flat;
    size_t flat_shape[] = {merged, 16};
    flat.data = hss_out->data;
    flat.shape = flat_shape; flat.ndim = 2;
    flat.size = merged * 16; flat.item_size = sizeof(float);
    flat.dtype = ARIX_FLOAT32; flat.strides = NULL;

    ArixTensor* ser_out = NULL;
    arix_ser_forward(ser->layers[0], &flat, &ser_out);
    ASSERT(ser_out != NULL, "ser output");
    ASSERT(ser_out->shape[0] == merged && ser_out->shape[1] == 16, "ser shape");

    ArixTensor* arc_out = NULL;
    float metrics[4];
    arix_arc_forward(arc, &flat, &arc_out, metrics);
    ASSERT(arc_out != NULL, "arc output");
    ASSERT(arc_out->shape[0] == merged && arc_out->shape[1] == 16, "arc shape");

    float* od = (float*)arc_out->data;
    int ok = 1;
    for (size_t i = 0; i < arc_out->size; i++) {
        if (!isfinite(od[i])) { ok = 0; break; }
    }
    ASSERT(ok, "all finite");
    for (int i = 0; i < 4; i++) ASSERT(isfinite(metrics[i]), "all metrics finite");

    printf("  Security metrics: anomaly=%.4f conf=%.4f clamp=%.4f norm=%.4f\n",
           metrics[0], metrics[1], metrics[2], metrics[3]);

    arix_tensor_destroy(input);
    arix_tensor_destroy(hss_out);
    arix_tensor_destroy(ser_out);
    if (arc_out) arix_tensor_destroy(arc_out);
    arix_hss_model_destroy(hss);
    arix_ser_model_destroy(ser);
    arix_arc_layer_destroy(arc);
}

int main(void) {
    run_test("test_hss_ser_arc_stack", test_hss_ser_arc_stack);
    printf("\nIntegration tests: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
