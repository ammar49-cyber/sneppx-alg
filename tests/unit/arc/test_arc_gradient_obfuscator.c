#include "adversarial_robustness_certification.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static int tests_passed = 0, tests_failed = 0;
#define ASSERT(cond, msg) do { if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } } while(0)
static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout); fn(); printf("PASS\n"); tests_passed++;
}

static void test_obf_create(void) {
    ArixGradientObfuscator* obf = arix_gradient_obfuscator_create(1000, 42);
    ASSERT(obf != NULL, "obf not null");
    ASSERT(obf->noise_buffer->shape[0] == 1000, "noise buf size 1000");
    ASSERT(obf->clamp_mask->shape[0] == 1000, "clamp mask size 1000");
    arix_gradient_obfuscator_destroy(obf);
}

static void test_obf_noise(void) {
    ArixGradientObfuscator* obf = arix_gradient_obfuscator_create(100, 42);
    ASSERT(obf != NULL, "obf not null");

    size_t shape_g[] = {100};
    ArixTensor* grad = arix_tensor_zeros(shape_g, 1, ARIX_FLOAT32);
    float* gd = (float*)grad->data;
    for (size_t i = 0; i < 100; i++) gd[i] = 0.5f;

    arix_arc_obfuscate_gradients(obf, grad, ARIX_OBF_NOISE);
    int changed = 0;
    for (size_t i = 0; i < 100; i++) { if (gd[i] != 0.5f) { changed = 1; break; } }
    ASSERT(changed, "values changed by noise");

    arix_tensor_destroy(grad);
    arix_gradient_obfuscator_destroy(obf);
}

static void test_obf_clamp(void) {
    ArixGradientObfuscator* obf = arix_gradient_obfuscator_create(100, 42);
    ASSERT(obf != NULL, "obf not null");

    size_t shape_g[] = {100};
    ArixTensor* grad = arix_tensor_zeros(shape_g, 1, ARIX_FLOAT32);
    float* gd = (float*)grad->data;
    for (size_t i = 0; i < 100; i++) gd[i] = (i % 2 == 0) ? 5.0f : 0.1f;

    arix_arc_obfuscate_gradients(obf, grad, ARIX_OBF_CLAMP);
    float max_abs = 0.0f;
    for (size_t i = 0; i < 100; i++) {
        float a = fabsf(gd[i]);
        if (a > max_abs) max_abs = a;
    }
    ASSERT(max_abs <= 1.0f + 1e-6f, "max abs <= 1.0");

    arix_tensor_destroy(grad);
    arix_gradient_obfuscator_destroy(obf);
}

static void test_obf_mixed(void) {
    ArixGradientObfuscator* obf = arix_gradient_obfuscator_create(100, 42);
    ASSERT(obf != NULL, "obf not null");

    size_t shape_g[] = {100};
    ArixTensor* grad = arix_tensor_zeros(shape_g, 1, ARIX_FLOAT32);
    float* gd = (float*)grad->data;
    for (size_t i = 0; i < 100; i++) gd[i] = (i % 2 == 0) ? 5.0f : 0.1f;

    arix_arc_obfuscate_gradients(obf, grad, ARIX_OBF_MIXED);
    float max_abs = 0.0f;
    for (size_t i = 0; i < 100; i++) {
        float a = fabsf(gd[i]);
        if (a > max_abs) max_abs = a;
    }
    ASSERT(max_abs <= 1.0f + 1e-4f, "clamped after mixed");

    arix_tensor_destroy(grad);
    arix_gradient_obfuscator_destroy(obf);
}

int main(void) {
    run_test("test_obf_create", test_obf_create);
    run_test("test_obf_noise", test_obf_noise);
    run_test("test_obf_clamp", test_obf_clamp);
    run_test("test_obf_mixed", test_obf_mixed);
    printf("\nObfuscator tests: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
