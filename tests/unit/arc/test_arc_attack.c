#include "adversarial_robustness_certification.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static int tests_passed = 0, tests_failed = 0;
#define ASSERT(cond, msg) do { if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } } while(0)
static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout); fn(); printf("PASS\n"); tests_passed++;
}

static void test_fgsm(void) {
    size_t shape_in[] = {8, 16};
    ArixTensor* clean = arix_tensor_zeros(shape_in, 2, ARIX_FLOAT32);
    float* d = (float*)clean->data;
    for (size_t i = 0; i < 8 * 16; i++) d[i] = 0.1f;

    ArixTensor* adv = NULL;
    arix_arc_simulate_attack(clean, ARIX_ATTACK_FGSM, 0.1f, &adv);
    ASSERT(adv != NULL, "adv not null");
    ASSERT(adv->shape[0] == 8, "batch ok");
    ASSERT(adv->shape[1] == 16, "dim ok");

    float* cd = (float*)clean->data;
    float* ad = (float*)adv->data;
    int different = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < 8 * 16; i++) {
        float diff = fabsf(ad[i] - cd[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-6f) different = 1;
    }
    ASSERT(different, "adversarial != clean");
    ASSERT(max_diff <= 0.1f + 1e-4f, "L_inf <= epsilon");

    arix_tensor_destroy(clean);
    arix_tensor_destroy(adv);
}

static void test_pgd(void) {
    size_t shape_in[] = {8, 16};
    ArixTensor* clean = arix_tensor_zeros(shape_in, 2, ARIX_FLOAT32);
    float* d = (float*)clean->data;
    for (size_t i = 0; i < 8 * 16; i++) d[i] = 0.1f;

    ArixTensor* adv = NULL;
    arix_arc_simulate_attack(clean, ARIX_ATTACK_PGD, 0.1f, &adv);
    ASSERT(adv != NULL, "adv not null");

    float* cd = (float*)clean->data;
    float* ad = (float*)adv->data;
    float max_diff = 0.0f;
    for (size_t i = 0; i < 8 * 16; i++) {
        float diff = fabsf(ad[i] - cd[i]);
        if (diff > max_diff) max_diff = diff;
    }
    ASSERT(max_diff <= 0.1f + 1e-4f, "L_inf <= epsilon");

    arix_tensor_destroy(clean);
    arix_tensor_destroy(adv);
}

static void test_cw(void) {
    size_t shape_in[] = {8, 16};
    ArixTensor* clean = arix_tensor_zeros(shape_in, 2, ARIX_FLOAT32);
    float* d = (float*)clean->data;
    for (size_t i = 0; i < 8 * 16; i++) d[i] = 0.1f;

    ArixTensor* adv = NULL;
    arix_arc_simulate_attack(clean, ARIX_ATTACK_CW, 0.1f, &adv);
    ASSERT(adv != NULL, "adv not null");

    float* cd = (float*)clean->data;
    float* ad = (float*)adv->data;
    float l2 = 0.0f;
    int different = 0;
    for (size_t i = 0; i < 8 * 16; i++) {
        float diff = ad[i] - cd[i];
        l2 += diff * diff;
        if (fabsf(diff) > 1e-6f) different = 1;
    }
    ASSERT(different, "adversarial != clean");

    arix_tensor_destroy(clean);
    arix_tensor_destroy(adv);
}

int main(void) {
    run_test("test_fgsm", test_fgsm);
    run_test("test_pgd", test_pgd);
    run_test("test_cw", test_cw);
    printf("\nAttack tests: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
