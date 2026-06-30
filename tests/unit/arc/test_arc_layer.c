#include "adversarial_robustness_certification.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static int tests_passed = 0, tests_failed = 0;
#define ASSERT(cond, msg) do { if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } } while(0)
static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout); fn(); printf("PASS\n"); tests_passed++;
}

static void test_arc_forward(void) {
    ArixARCConfig cfg = arix_arc_config_default();
    ArixARCLayer* layer = arix_arc_layer_create(&cfg, 32, 32, 42);
    ASSERT(layer != NULL, "layer not null");

    size_t shape_in[] = {4, 32};
    ArixTensor* input = arix_tensor_zeros(shape_in, 2, ARIX_FLOAT32);
    float* d = (float*)input->data;
    for (size_t i = 0; i < 4 * 32; i++) d[i] = 0.1f;

    ArixTensor* output = NULL;
    float metrics[4];
    arix_arc_forward(layer, input, &output, metrics);

    ASSERT(output != NULL, "output not null");
    ASSERT(output->shape[0] == 4, "batch == 4");
    ASSERT(output->shape[1] == 32, "dim == 32");

    int all_finite = 1;
    float* od = (float*)output->data;
    for (size_t i = 0; i < output->size; i++) {
        if (!isfinite(od[i])) { all_finite = 0; break; }
    }
    ASSERT(all_finite, "all finite in output");

    for (int i = 0; i < 4; i++) {
        ASSERT(isfinite(metrics[i]), "all metrics finite");
    }

    arix_tensor_destroy(input);
    arix_tensor_destroy(output);
    arix_arc_layer_destroy(layer);
}

int main(void) {
    run_test("test_arc_forward", test_arc_forward);
    printf("\nARC layer tests: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
