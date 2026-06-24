#include "arix_hss.h"
#include "arix_ser.h"
#include "arix_arc.h"
#include "arix_npe.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

static int tests_passed = 0, tests_failed = 0;
#define ASSERT(cond, msg) do { if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } } while(0)
static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout); fn(); printf("PASS\n"); tests_passed++;
}

static void test_full_stack(void) {
    ArixHSSConfig hss_cfg = arix_hss_config_default();
    hss_cfg.state_dim = 4; hss_cfg.input_dim = 8; hss_cfg.output_dim = 8;
    hss_cfg.num_layers = 1; hss_cfg.use_hierarchical = 0;
    ArixHSSModel* hss = arix_hss_model_create(&hss_cfg, 42);

    ArixSERConfig ser_cfg = arix_ser_config_default();
    ser_cfg.num_experts = 2; ser_cfg.num_active = 1; ser_cfg.input_dim = 8;
    ser_cfg.expert_dim = 16; ser_cfg.output_dim = 8;
    ArixSERModel* ser = arix_ser_model_create(&ser_cfg, 99, 1);
    ASSERT(ser != NULL, "ser model");

    ArixARCConfig arc_cfg = arix_arc_config_default();
    ArixARCLayer* arc = arix_arc_layer_create(&arc_cfg, 8, 8, 200);
    ASSERT(arc != NULL, "arc layer");

    ArixNPEProgram* npe_prog = arix_npe_compile_mlp(8, 16);
    ASSERT(npe_prog != NULL, "npe program");
    unsigned long s = 42;
    for (size_t i = 0; i < 16 * 8 + 16 + 8 * 16 + 8; i++) {
        s = s * 1103515245UL + 12345UL;
        ((float*)npe_prog->memory->data)[i] = ((float)((s >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 0.1f;
    }

    ArixNPEConfig npe_cfg = arix_npe_config_default();
    ArixNPEVM* vm = arix_npe_vm_create(&npe_cfg);
    arix_npe_vm_load(vm, npe_prog);

    size_t shape_in[] = {1, 4, 8};
    ArixTensor* input = arix_tensor_randn(shape_in, 3, ARIX_FLOAT32);
    ASSERT(input != NULL, "input");

    ArixTensor* hss_out = NULL;
    int ret = arix_hss_forward(hss, input, &hss_out);
    ASSERT(ret == 0 && hss_out != NULL && hss_out->shape[0] == 1 && hss_out->shape[1] == 4 && hss_out->shape[2] == 8, "hss ok");

    size_t merged = 4;
    ArixTensor flat;
    size_t flat_shape[] = {merged, 8};
    flat.data = hss_out->data; flat.shape = flat_shape; flat.ndim = 2;
    flat.size = merged * 8; flat.item_size = sizeof(float);
    flat.dtype = ARIX_FLOAT32; flat.strides = NULL;

    ArixTensor* ser_out = NULL;
    arix_ser_forward(ser->layers[0], &flat, &ser_out);
    ASSERT(ser_out != NULL && ser_out->shape[0] == merged && ser_out->shape[1] == 8, "ser ok");

    ArixTensor* arc_out = NULL;
    float metrics[4];
    arix_arc_forward(arc, ser_out, &arc_out, metrics);
    ASSERT(arc_out != NULL && arc_out->shape[0] == merged && arc_out->shape[1] == 8, "arc ok");

    ArixTensor* npe_out = NULL;
    int r = arix_npe_vm_run(vm, arc_out, &npe_out);
    ASSERT(npe_out != NULL && (r == 0 || r == 2), "npe ok");

    float* od = (float*)npe_out->data;
    int ok = 1;
    for (size_t i = 0; i < npe_out->size; i++) {
        if (!isfinite(od[i])) { ok = 0; break; }
    }
    ASSERT(ok, "all finite");
    ASSERT(vm->trace_length > 0, "execution trace non-empty");

    printf("  Trace length: %zu\n", vm->trace_length);

    arix_tensor_destroy(input);
    arix_tensor_destroy(hss_out);
    arix_tensor_destroy(ser_out);
    arix_tensor_destroy(arc_out);
    arix_tensor_destroy(npe_out);
    arix_hss_model_destroy(hss);
    arix_ser_model_destroy(ser);
    arix_arc_layer_destroy(arc);
    arix_npe_vm_destroy(vm);
    arix_npe_program_destroy(npe_prog);
}

int main(void) {
    run_test("test_full_stack", test_full_stack);
    printf("\nIntegration tests: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
