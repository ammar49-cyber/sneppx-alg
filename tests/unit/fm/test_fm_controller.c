#include "fractal_memory_orchestrator.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

static int tests_passed = 0, tests_failed = 0;
#define ASSERT(cond, msg) do { if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } } while(0)
static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout); fn(); printf("PASS\n"); tests_passed++;
}

static void test_controller_create(void) {
    ArixFMConfig cfg = arix_fm_config_default();
    cfg.num_nodes = 4;
    cfg.memory_dim = 8;
    cfg.memory_capacity = 16;

    ArixFMController* ctrl = arix_fm_controller_create(&cfg);
    ASSERT(ctrl != NULL, "ctrl not null");
    ASSERT(ctrl->nodes != NULL, "nodes not null");
    ASSERT(ctrl->config.num_nodes == 4, "4 nodes");
    for (size_t i = 0; i < 4; i++) {
        ASSERT(ctrl->nodes[i] != NULL, "node allocated");
        ASSERT(ctrl->nodes[i]->node_id == i, "node id");
        ASSERT(ctrl->nodes[i]->is_online == 1, "online");
    }
    ASSERT(ctrl->sync_state.global_memory != NULL, "global memory");
    ASSERT(ctrl->sync_state.sync_round == 0, "sync round 0");
    arix_fm_controller_destroy(ctrl);
}

static void test_controller_forward(void) {
    ArixFMConfig cfg = arix_fm_config_default();
    cfg.num_nodes = 2;
    cfg.memory_dim = 16;
    cfg.memory_capacity = 8;
    cfg.sync_interval = 1000;

    ArixFMController* ctrl = arix_fm_controller_create(&cfg);
    ASSERT(ctrl != NULL, "ctrl");

    size_t sh[] = {8, 16};
    ArixTensor* input = arix_tensor_zeros(sh, 2, ARIX_FLOAT32);
    ArixTensor* output = NULL;

    int r = arix_fm_forward(ctrl, 0, input, &output);
    ASSERT(r == 0, "forward ok");
    ASSERT(output != NULL, "output not null");
    ASSERT(output->shape[0] == 8, "output rows 8");
    ASSERT(output->shape[1] == 16, "output cols 16");

    int has_nan = 0, has_inf = 0;
    float* od = (float*)output->data;
    for (size_t i = 0; i < output->size; i++) {
        if (isnan(od[i])) has_nan = 1;
        if (isinf(od[i])) has_inf = 1;
    }
    ASSERT(!has_nan, "no nan");
    ASSERT(!has_inf, "no inf");

    arix_tensor_destroy(input);
    arix_tensor_destroy(output);
    arix_fm_controller_destroy(ctrl);
}

int main(void) {
    run_test("test_controller_create", test_controller_create);
    run_test("test_controller_forward", test_controller_forward);
    printf("\nController tests: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
