#include "hierarchical_state_space.h"
#include "sparse_expert_routing.h"
#include "adversarial_robustness_certification.h"
#include "neural_programming_engine.h"
#include "fractal_memory_orchestrator.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

static int tests_passed = 0, tests_failed = 0;
#define ASSERT(cond, msg) do { if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } } while(0)
static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout); fn(); printf("PASS\n"); tests_passed++;
}

static void test_complete_stack(void) {
    ArixHSSConfig hss_cfg = arix_hss_config_default();
    hss_cfg.state_dim = 4; hss_cfg.input_dim = 16; hss_cfg.output_dim = 16;
    hss_cfg.num_layers = 1; hss_cfg.use_hierarchical = 0;
    ArixHSSModel* hss = arix_hss_model_create(&hss_cfg, 42);
    ASSERT(hss != NULL, "hss model");

    ArixSERConfig ser_cfg = arix_ser_config_default();
    ser_cfg.num_experts = 2; ser_cfg.num_active = 1; ser_cfg.input_dim = 16;
    ser_cfg.expert_dim = 32; ser_cfg.output_dim = 16;
    ArixSERModel* ser = arix_ser_model_create(&ser_cfg, 99, 1);
    ASSERT(ser != NULL, "ser model");

    ArixARCConfig arc_cfg = arix_arc_config_default();
    ArixARCLayer* arc = arix_arc_layer_create(&arc_cfg, 16, 16, 200);
    ASSERT(arc != NULL, "arc layer");

    ArixNPEProgram* npe_prog = arix_npe_compile_mlp(16, 32);
    ASSERT(npe_prog != NULL, "npe program");
    unsigned long s = 42;
    for (size_t i = 0; i < 16 * 32 + 32 + 32 * 16 + 16; i++) {
        s = s * 1103515245UL + 12345UL;
        ((float*)npe_prog->memory->data)[i] = ((float)((s >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 0.1f;
    }

    ArixNPEConfig npe_cfg = arix_npe_config_default();
    ArixNPEVM* vm = arix_npe_vm_create(&npe_cfg);
    arix_npe_vm_load(vm, npe_prog);

    ArixFMConfig fm_cfg = arix_fm_config_default();
    fm_cfg.num_nodes = 2;
    fm_cfg.memory_dim = 16;
    fm_cfg.memory_capacity = 32;
    fm_cfg.sync_interval = 1000;
    ArixFMController* fm = arix_fm_controller_create(&fm_cfg);
    ASSERT(fm != NULL, "fm controller");

    size_t shape_in[] = {1, 4, 16};
    ArixTensor* input = arix_tensor_randn(shape_in, 3, ARIX_FLOAT32);
    ASSERT(input != NULL, "input");

    ArixTensor* hss_out = NULL;
    int ret = arix_hss_forward(hss, input, &hss_out);
    ASSERT(ret == 0 && hss_out != NULL && hss_out->shape[0] == 1 && hss_out->shape[1] == 4 && hss_out->shape[2] == 16, "hss ok");

    size_t merged = 4;
    ArixTensor flat;
    size_t flat_shape[] = {merged, 16};
    flat.data = hss_out->data; flat.shape = flat_shape; flat.ndim = 2;
    flat.size = merged * 16; flat.item_size = sizeof(float);
    flat.dtype = ARIX_FLOAT32; flat.strides = NULL;

    ArixTensor* ser_out = NULL;
    arix_ser_forward(ser->layers[0], &flat, &ser_out);
    ASSERT(ser_out != NULL && ser_out->shape[0] == merged && ser_out->shape[1] == 16, "ser ok");

    ArixTensor* arc_out = NULL;
    float metrics[4];
    arix_arc_forward(arc, ser_out, &arc_out, metrics);
    ASSERT(arc_out != NULL && arc_out->shape[0] == merged && arc_out->shape[1] == 16, "arc ok");

    ArixTensor* npe_out = NULL;
    arix_npe_vm_run(vm, arc_out, &npe_out);
    ASSERT(npe_out != NULL && npe_out->shape[0] == merged && npe_out->shape[1] == 16, "npe ok");

    ArixTensor* fm_out = NULL;
    ret = arix_fm_forward(fm, 0, npe_out, &fm_out);
    ASSERT(ret == 0 && fm_out != NULL, "fm ok");
    ASSERT(fm_out->shape[0] == merged && fm_out->shape[1] == 16, "fm shape ok");

    ASSERT(vm->trace_length > 0, "execution trace non-empty");
    int has_nan = 0, has_inf = 0;
    float* od = (float*)fm_out->data;
    for (size_t i = 0; i < fm_out->size; i++) {
        if (isnan(od[i])) has_nan = 1;
        if (isinf(od[i])) has_inf = 1;
    }
    ASSERT(!has_nan && !has_inf, "no nan/inf");
    ASSERT(fm->nodes[0]->memory_bank->num_entries > 0, "memory has entries");

    printf("  FM memory entries: %zu\n", fm->nodes[0]->memory_bank->num_entries);

    arix_tensor_destroy(input);
    arix_tensor_destroy(hss_out);
    arix_tensor_destroy(ser_out);
    arix_tensor_destroy(arc_out);
    arix_tensor_destroy(npe_out);
    arix_tensor_destroy(fm_out);
    arix_hss_model_destroy(hss);
    arix_ser_model_destroy(ser);
    arix_arc_layer_destroy(arc);
    arix_npe_vm_destroy(vm);
    arix_npe_program_destroy(npe_prog);
    arix_fm_controller_destroy(fm);
}

int main(void) {
    run_test("test_complete_stack", test_complete_stack);
    printf("\nComplete stack integration test: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
