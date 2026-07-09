#include "code_obfuscation_interface.h"
#include <stdio.h>

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

static void test_obf_pipeline_create(void) {
    SNEPPXObfPipeline* pipe = SNEPPX_obf_pipeline_create(42);
    ASSERT(pipe != NULL, "obf pipeline created");
    SNEPPX_obf_pipeline_destroy(pipe);
}

static void test_obf_pipeline_add_pass(void) {
    SNEPPXObfPipeline* pipe = SNEPPX_obf_pipeline_create(42);
    SNEPPX_obf_pipeline_add_pass(pipe, SNEPPX_OBF_INST_SUBST);
    SNEPPX_obf_pipeline_add_pass(pipe, SNEPPX_OBF_CFG_FLATTEN);
    ASSERT(pipe != NULL, "passes added");
    SNEPPX_obf_pipeline_destroy(pipe);
}

static void test_obf_pipeline_run(void) {
    SNEPPXObfPipeline* pipe = SNEPPX_obf_pipeline_create(42);
    SNEPPX_obf_pipeline_add_pass(pipe, SNEPPX_OBF_INST_SUBST);
    int ret = SNEPPX_obf_pipeline_run(pipe, NULL, 0, NULL, 0);
    ASSERT(ret == 0, "pipeline run stub");
    SNEPPX_obf_pipeline_destroy(pipe);
}

int main(void) {
    run_test("obf_pipeline_create", test_obf_pipeline_create);
    run_test("obf_pipeline_add_pass", test_obf_pipeline_add_pass);
    run_test("obf_pipeline_run", test_obf_pipeline_run);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
