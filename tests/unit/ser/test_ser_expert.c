#include "sparse_expert_routing.h"
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

static void test_expert_create(void) {
    SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
    cfg.input_dim = 64; cfg.expert_dim = 128; cfg.output_dim = 64;
    SNEPPXExpert* e = SNEPPX_expert_create(&cfg, 42, SNEPPX_ACT_RELU);
    ASSERT(e != NULL, "expert not null");
    ASSERT(e->w1->shape[0] == 128 && e->w1->shape[1] == 64, "w1 shape");
    ASSERT(e->w2->shape[0] == 64 && e->w2->shape[1] == 128, "w2 shape");
    ASSERT(e->b1->shape[0] == 128, "b1 shape");
    ASSERT(e->b2->shape[0] == 64, "b2 shape");
    SNEPPX_expert_destroy(e);
}

static void test_expert_forward(void) {
    SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
    cfg.input_dim = 16; cfg.expert_dim = 32; cfg.output_dim = 16;
    SNEPPXExpert* e = SNEPPX_expert_create(&cfg, 42, SNEPPX_ACT_RELU);
    ASSERT(e != NULL, "expert not null");

    size_t shape_in[] = {4, 16};
    SNEPPXTensor* input = SNEPPX_tensor_randn(shape_in, 2, SNEPPX_FLOAT32);
    ASSERT(input != NULL, "input not null");

    size_t shape_out[] = {4, 16};
    SNEPPXTensor* output = SNEPPX_tensor_zeros(shape_out, 2, SNEPPX_FLOAT32);
    ASSERT(output != NULL, "output not null");

    SNEPPX_ser_expert_forward(e, input, output);
    float* od = (float*)output->data;
    int ok = 1;
    for (size_t i = 0; i < 4 * 16; i++) {
        if (!isfinite(od[i])) { ok = 0; break; }
    }
    ASSERT(ok, "all finite");

    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(output);
    SNEPPX_expert_destroy(e);
}

int main(void) {
    run_test("test_expert_create", test_expert_create);
    run_test("test_expert_forward", test_expert_forward);
    printf("\nExpert tests: %d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
