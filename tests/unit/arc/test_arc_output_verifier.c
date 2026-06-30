#include "adversarial_robustness_certification.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static int tests_passed = 0, tests_failed = 0;
#define ASSERT(cond, msg) do { if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } } while(0)
static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout); fn(); printf("PASS\n"); tests_passed++;
}

static void test_verifier_create(void) {
    ArixOutputVerifier* v = arix_output_verifier_create(32, 2, 42);
    ASSERT(v != NULL, "verifier not null");
    ASSERT(v->verification_weights[0]->shape[0] == 32, "w0 rows 32");
    ASSERT(v->verification_weights[0]->shape[1] == 32, "w0 cols 32");
    ASSERT(v->verification_weights[1] != NULL, "w1 not null");
    ASSERT(v->num_layers == 2, "num_layers == 2");
    arix_output_verifier_destroy(v);
}

static void test_verify_normal(void) {
    ArixOutputVerifier* v = arix_output_verifier_create(16, 1, 42);
    ASSERT(v != NULL, "verifier not null");

    size_t shape_in[] = {4, 16};
    ArixTensor* out = arix_tensor_zeros(shape_in, 2, ARIX_FLOAT32);
    float* d = (float*)out->data;
    for (size_t i = 0; i < 4 * 16; i++) d[i] = 0.1f;

    ArixTensor* verified = NULL;
    float conf = 0.0f;
    arix_arc_verify_output(v, out, &verified, &conf);
    ASSERT(verified != NULL, "verified not null");
    ASSERT(conf == 1.0f, "confidence 1.0 on first call (no history)");

    arix_arc_verify_output(v, out, &verified, &conf);
    ASSERT(conf > 0.8f, "confidence > 0.8 on second call (consistent)");

    arix_tensor_destroy(out);
    if (verified) arix_tensor_destroy(verified);
    arix_output_verifier_destroy(v);
}

static void test_verify_inconsistent(void) {
    ArixOutputVerifier* v = arix_output_verifier_create(16, 1, 42);
    ASSERT(v != NULL, "verifier not null");

    size_t shape_in[] = {4, 16};
    ArixTensor* out1 = arix_tensor_zeros(shape_in, 2, ARIX_FLOAT32);
    float* d1 = (float*)out1->data;
    for (size_t i = 0; i < 4 * 16; i++) d1[i] = 0.1f;

    ArixTensor* verified1 = NULL;
    float conf1 = 0.0f;
    arix_arc_verify_output(v, out1, &verified1, &conf1);
    arix_tensor_destroy(verified1);

    ArixTensor* out2 = arix_tensor_zeros(shape_in, 2, ARIX_FLOAT32);
    float* d2 = (float*)out2->data;
    unsigned long s2 = 999;
    for (size_t i = 0; i < 4 * 16; i++) {
        s2 = s2 * 1103515245UL + 12345UL;
        d2[i] = ((float)((s2 >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 10.0f;
    }

    ArixTensor* verified2 = NULL;
    float conf2 = 0.0f;
    arix_arc_verify_output(v, out2, &verified2, &conf2);
    ASSERT(verified2 != NULL, "verified2 not null");
    ASSERT(conf2 < 0.8f, "confidence < 0.8 for random vs constant");

    arix_tensor_destroy(out1);
    arix_tensor_destroy(out2);
    if (verified2) arix_tensor_destroy(verified2);
    arix_output_verifier_destroy(v);
}

int main(void) {
    run_test("test_verifier_create", test_verifier_create);
    run_test("test_verify_normal", test_verify_normal);
    run_test("test_verify_inconsistent", test_verify_inconsistent);
    printf("\nVerifier tests: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
