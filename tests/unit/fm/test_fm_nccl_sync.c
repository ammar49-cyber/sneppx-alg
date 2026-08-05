#include "fractal_memory_orchestrator.h"
#include "polymorphic_memory_allocator.h"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test Fm Nccl Sync
 *
 * WHAT
 *   Test Fm Nccl Sync.
 *
 * CONCEPT
 *   Provides NCCL communication layer.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static int tests_passed = 0, tests_failed = 0;

#define ASSERT(cond, msg) do { if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } } while(0)

static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout); fn(); printf("PASS\n"); tests_passed++;
}

static int dummy_allreduce(void* data, size_t count, void* ctx) {
    (void)ctx;
    return 0;
}

static void test_fm_nccl_sync_null_callback(void) {
    SNEPPXFMConfig cfg = SNEPPX_fm_config_default();
    cfg.num_nodes = 2;
    cfg.memory_dim = 4;
    cfg.memory_capacity = 8;
    SNEPPXFMController* ctrl = SNEPPX_fm_controller_create(&cfg);
    ASSERT(ctrl != NULL, "controller created");
    int rc = SNEPPX_fm_sync_nccl(ctrl, NULL, NULL);
    ASSERT(rc == 0, "sync with null callback ok");
    SNEPPX_fm_controller_destroy(ctrl);
}

static void test_fm_nccl_sync_with_callback(void) {
    SNEPPXFMConfig cfg = SNEPPX_fm_config_default();
    cfg.num_nodes = 2;
    cfg.memory_dim = 4;
    cfg.memory_capacity = 8;
    SNEPPXFMController* ctrl = SNEPPX_fm_controller_create(&cfg);
    ASSERT(ctrl != NULL, "controller created");

    size_t shape_k[] = {1, cfg.memory_dim};
    SNEPPXTensor* key = SNEPPX_tensor_ones(shape_k, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* val = SNEPPX_tensor_ones(shape_k, 2, SNEPPX_FLOAT32);
    for (size_t n = 0; n < cfg.num_nodes; n++) {
        SNEPPX_fm_memory_bank_write(ctrl->nodes[n]->memory_bank, key, val);
    }

    int rc = SNEPPX_fm_sync_nccl(ctrl, dummy_allreduce, &ctrl);
    ASSERT(rc == 0, "sync with dummy callback ok");
    ASSERT(ctrl->sync_state.sync_round > 0, "sync round incremented");

    SNEPPX_tensor_destroy(key);
    SNEPPX_tensor_destroy(val);
    SNEPPX_fm_controller_destroy(ctrl);
}

int main(void) {
    run_test("FM NCCL sync null callback", test_fm_nccl_sync_null_callback);
    run_test("FM NCCL sync with dummy callback", test_fm_nccl_sync_with_callback);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
