#include "adversarial_robustness_certification.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static int tests_passed = 0, tests_failed = 0;
#define ASSERT(cond, msg) do { if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } } while(0)
static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout); fn(); printf("PASS\n"); tests_passed++;
}

static void test_guard_create(void) {
    SNEPPXInputGuard* g = SNEPPX_input_guard_create(64, 42);
    ASSERT(g != NULL, "guard not null");
    ASSERT(g->projection_matrix->shape[0] == 64, "proj rows 64");
    ASSERT(g->projection_matrix->shape[1] == 64, "proj cols 64");
    SNEPPX_input_guard_destroy(g);
}

static void test_guard_normal_input(void) {
    SNEPPXInputGuard* g = SNEPPX_input_guard_create(64, 42);
    ASSERT(g != NULL, "guard not null");

    size_t shape_in[] = {8, 64};
    SNEPPXTensor* input = SNEPPX_tensor_create(shape_in, 2, SNEPPX_FLOAT32);
    float* d = (float*)input->data;
    unsigned long s = 123;
    for (size_t i = 0; i < 8 * 64; i++) {
        s = s * 1103515245UL + 12345UL;
        d[i] = ((float)((s >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 0.2f;
    }

    SNEPPXTensor* sanitized = NULL;
    float score = 0.0f;
    SNEPPX_arc_input_guard_forward(g, input, &sanitized, &score);
    ASSERT(sanitized != NULL, "sanitized not null");
    ASSERT(score < 0.1f, "anomaly_score < 0.1");
    ASSERT(sanitized->shape[0] == 8, "batch ok");
    ASSERT(sanitized->shape[1] == 64, "dim ok");

    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(sanitized);
    SNEPPX_input_guard_destroy(g);
}

static void test_guard_anomaly_input(void) {
    SNEPPXInputGuard* g = SNEPPX_input_guard_create(64, 42);
    ASSERT(g != NULL, "guard not null");

    size_t shape_in[] = {8, 64};
    SNEPPXTensor* input = SNEPPX_tensor_create(shape_in, 2, SNEPPX_FLOAT32);
    float* d = (float*)input->data;
    unsigned long s = 123;
    for (size_t i = 0; i < 8 * 64; i++) {
        s = s * 1103515245UL + 12345UL;
        d[i] = ((float)((s >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 20.0f;
    }

    SNEPPXTensor* sanitized = NULL;
    float score = 0.0f;
    SNEPPX_arc_input_guard_forward(g, input, &sanitized, &score);
    ASSERT(sanitized != NULL, "sanitized not null");
    ASSERT(score > 0.5f, "anomaly_score > 0.5");

    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(sanitized);
    SNEPPX_input_guard_destroy(g);
}

int main(void) {
    run_test("test_guard_create", test_guard_create);
    run_test("test_guard_normal_input", test_guard_normal_input);
    run_test("test_guard_anomaly_input", test_guard_anomaly_input);
    printf("\nInput guard tests: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
