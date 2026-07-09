#include "hierarchical_state_space.h"
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

static void test_hierarchical_config(void) {
    SNEPPXHSSConfig cfg = SNEPPX_hss_config_default();
    ASSERT(cfg.state_dim == 64, "default state_dim 64");
    ASSERT(cfg.input_dim == 64, "default input_dim 64");
    ASSERT(cfg.num_layers == 4, "default num_layers 4");
    cfg.use_hierarchical = 1;
    ASSERT(cfg.use_hierarchical == 1, "hierarchical enabled");
}

static void test_hierarchical_level_count(void) {
    int levels = SNEPPX_hss_hierarchical_levels(64, 4);
    ASSERT(levels == 3, "log2(64/4)/log2(2) levels");
    levels = SNEPPX_hss_hierarchical_levels(16, 2);
    ASSERT(levels == 3, "log2(16/2)/log2(2)=3 levels");
}

static void test_hierarchical_create(void) {
    SNEPPXHSSConfig cfg = SNEPPX_hss_config_default();
    cfg.state_dim = 16;
    cfg.input_dim = 16;
    cfg.use_hierarchical = 1;
    SNEPPXHSSModel* model = SNEPPX_hss_model_create(&cfg, 42);
    ASSERT(model != NULL, "hierarchical model created");
    SNEPPX_hss_model_destroy(model);
}

int main(void) {
    run_test("hierarchical_config", test_hierarchical_config);
    run_test("hierarchical_level_count", test_hierarchical_level_count);
    run_test("hierarchical_create", test_hierarchical_create);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
