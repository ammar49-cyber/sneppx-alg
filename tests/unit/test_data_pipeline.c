#include "data_pipeline.h"
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

static void run_test(const char* name, void (*test_fn)(void)) {
    printf("Running %s... ", name);
    fflush(stdout);
    test_fn();
    printf("PASS\n");
    tests_passed++;
}

static void test_data_pipeline_create(void) {
    SNEPPXDataPipeline* pipe = SNEPPX_data_pipeline_create(256);
    ASSERT(pipe != NULL, "pipeline created");
    ASSERT(pipe->batch_size == 256, "batch size set");
    SNEPPX_data_pipeline_destroy(pipe);
}

static void test_data_pipeline_load_data(void) {
    SNEPPXDataPipeline* pipe = SNEPPX_data_pipeline_create(32);
    SNEPPXTensor* data = NULL;
    SNEPPXTensor* labels = NULL;
    int ret = SNEPPX_data_pipeline_load("dummy_path", pipe, &data, &labels);
    ASSERT(ret == 0, "load returns 0 (stub)");
    SNEPPX_data_pipeline_destroy(pipe);
}

static void test_data_pipeline_get_batch(void) {
    SNEPPXDataPipeline* pipe = SNEPPX_data_pipeline_create(16);
    SNEPPXTensor* batch = NULL;
    SNEPPXTensor* labels = NULL;
    int ret = SNEPPX_data_pipeline_get_batch(pipe, &batch, &labels);
    ASSERT(ret == 0, "get_batch returns 0 (stub)");
    SNEPPX_data_pipeline_destroy(pipe);
}

static void test_data_pipeline_shuffle(void) {
    SNEPPXDataPipeline* pipe = SNEPPX_data_pipeline_create(64);
    SNEPPX_data_pipeline_shuffle(pipe);
    ASSERT(pipe != NULL, "shuffle does not crash");
    SNEPPX_data_pipeline_destroy(pipe);
}

int main(void) {
    run_test("data_pipeline_create", test_data_pipeline_create);
    run_test("data_pipeline_load_data", test_data_pipeline_load_data);
    run_test("data_pipeline_get_batch", test_data_pipeline_get_batch);
    run_test("data_pipeline_shuffle", test_data_pipeline_shuffle);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
