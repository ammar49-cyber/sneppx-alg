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
    SNEPPXGradientObfuscator* obf = SNEPPX_gradient_obfuscator_create(1000, 42);
    ASSERT(obf != NULL, "obf not null");
    ASSERT(obf->noise_buffer->shape[0] == 1000, "noise buf size 1000");
    ASSERT(obf->clamp_mask->shape[0] == 1000, "clamp mask size 1000");
    SNEPPX_gradient_obfuscator_destroy(obf);
}

static void test_obf_noise(void) {
    SNEPPXGradientObfuscator* obf = SNEPPX_gradient_obfuscator_create(100, 42);
    ASSERT(obf != NULL, "obf not null");

    size_t shape_g[] = {100};
    SNEPPXTensor* grad = SNEPPX_tensor_zeros(shape_g, 1, SNEPPX_FLOAT32);
    float* gd = (float*)grad->data;
    for (size_t i = 0; i < 100; i++) gd[i] = 0.5f;

    SNEPPX_arc_obfuscate_gradients(obf, grad, SNEPPX_OBF_NOISE);
    int changed = 0;
    for (size_t i = 0; i < 100; i++) { if (gd[i] != 0.5f) { changed = 1; break; } }
    ASSERT(changed, "values changed by noise");

    SNEPPX_tensor_destroy(grad);
    SNEPPX_gradient_obfuscator_destroy(obf);
}

static void test_obf_clamp(void) {
    SNEPPXGradientObfuscator* obf = SNEPPX_gradient_obfuscator_create(100, 42);
    ASSERT(obf != NULL, "obf not null");

    size_t shape_g[] = {100};
    SNEPPXTensor* grad = SNEPPX_tensor_zeros(shape_g, 1, SNEPPX_FLOAT32);
    float* gd = (float*)grad->data;
    for (size_t i = 0; i < 100; i++) gd[i] = (i % 2 == 0) ? 5.0f : 0.1f;

    SNEPPX_arc_obfuscate_gradients(obf, grad, SNEPPX_OBF_CLAMP);
    float max_abs = 0.0f;
    for (size_t i = 0; i < 100; i++) {
        float a = fabsf(gd[i]);
        if (a > max_abs) max_abs = a;
    }
    ASSERT(max_abs <= 1.0f + 1e-6f, "max abs <= 1.0");

    SNEPPX_tensor_destroy(grad);
    SNEPPX_gradient_obfuscator_destroy(obf);
}

static void test_obf_mixed(void) {
    SNEPPXGradientObfuscator* obf = SNEPPX_gradient_obfuscator_create(100, 42);
    ASSERT(obf != NULL, "obf not null");

    size_t shape_g[] = {100};
    SNEPPXTensor* grad = SNEPPX_tensor_zeros(shape_g, 1, SNEPPX_FLOAT32);
    float* gd = (float*)grad->data;
    for (size_t i = 0; i < 100; i++) gd[i] = (i % 2 == 0) ? 5.0f : 0.1f;

    SNEPPX_arc_obfuscate_gradients(obf, grad, SNEPPX_OBF_MIXED);
    float max_abs = 0.0f;
    for (size_t i = 0; i < 100; i++) {
        float a = fabsf(gd[i]);
        if (a > max_abs) max_abs = a;
    }
    ASSERT(max_abs <= 1.0f + 1e-4f, "clamped after mixed");

    SNEPPX_tensor_destroy(grad);
    SNEPPX_gradient_obfuscator_destroy(obf);
}

int main(void) {
    run_test("test_obf_create", test_obf_create);
    run_test("test_obf_noise", test_obf_noise);
    run_test("test_obf_clamp", test_obf_clamp);
    run_test("test_obf_mixed", test_obf_mixed);
    printf("\nObfuscator tests: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
