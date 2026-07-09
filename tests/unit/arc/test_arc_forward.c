#include "adversarial_robustness_certification.h"
#include "polymorphic_memory_allocator.h"
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

static void test_arc_config_default(void) {
    SNEPPXARCConfig cfg = SNEPPX_arc_config_default();
    ASSERT(cfg.input_guard_strength > 0, "input guard strength > 0");
    ASSERT(cfg.gradient_obfuscation_method >= SNEPPX_OBF_NONE, "obf method valid");
}

static void test_arc_layer_create_destroy(void) {
    SNEPPXARCConfig cfg = SNEPPX_arc_config_default();
    SNEPPXARCLayer* layer = SNEPPX_arc_layer_create(&cfg, 8, 8, 42);
    ASSERT(layer != NULL, "arc layer created");
    ASSERT(layer->input_guard != NULL, "input guard created");
    ASSERT(layer->output_verifier != NULL, "output verifier created");
    ASSERT(layer->gradient_obfuscator != NULL, "gradient obfuscator created");
    SNEPPX_arc_layer_destroy(layer);
}

static void test_arc_forward_pass(void) {
    SNEPPXARCConfig cfg = SNEPPX_arc_config_default();
    cfg.input_guard_strength = 3.0f;
    SNEPPXARCLayer* layer = SNEPPX_arc_layer_create(&cfg, 4, 4, 42);
    ASSERT(layer != NULL, "layer created");

    size_t shape_in[] = {1, 4};
    SNEPPXTensor* input = SNEPPX_tensor_create(shape_in, 2, SNEPPX_FLOAT32);
    float* d = (float*)input->data;
    for (size_t i = 0; i < 4; i++) d[i] = 1.0f;

    SNEPPXTensor* output = NULL;
    float metrics[4];
    SNEPPX_arc_forward(layer, input, &output, metrics);
    ASSERT(output != NULL, "forward output created");
    ASSERT(metrics[0] >= 0.0f, "anomaly score >= 0");
    ASSERT(metrics[1] >= 0.0f, "confidence >= 0");

    SNEPPX_tensor_destroy(output);
    SNEPPX_tensor_destroy(input);
    SNEPPX_arc_layer_destroy(layer);
}

static void test_arc_input_guard(void) {
    SNEPPXARCConfig cfg = SNEPPX_arc_config_default();
    SNEPPXARCLayer* layer = SNEPPX_arc_layer_create(&cfg, 4, 4, 42);

    SNEPPXTensor* input = SNEPPX_tensor_create((size_t[]){1, 4}, 2, SNEPPX_FLOAT32);
    float* d = (float*)input->data;
    for (size_t i = 0; i < 4; i++) d[i] = 0.5f;

    SNEPPXTensor* sanitized = NULL;
    float score = 0.0f;
    SNEPPX_arc_input_guard_forward(layer->input_guard, input, &sanitized, &score);
    ASSERT(sanitized != NULL, "sanitized output");

    SNEPPX_tensor_destroy(sanitized);
    SNEPPX_tensor_destroy(input);
    SNEPPX_arc_layer_destroy(layer);
}

static void test_arc_simulate_attack(void) {
    size_t shape[] = {1, 4};
    SNEPPXTensor* input = SNEPPX_tensor_zeros(shape, 2, SNEPPX_FLOAT32);
    float* d = (float*)input->data;
    for (size_t i = 0; i < 4; i++) d[i] = 1.0f;

    SNEPPXTensor* adv = NULL;
    SNEPPX_arc_simulate_attack(input, SNEPPX_ATTACK_FGSM, 0.1f, &adv);
    ASSERT(adv != NULL, "adversarial sample created");

    float* ad = (float*)adv->data;
    int changed = 0;
    for (size_t i = 0; i < 4; i++) if (ad[i] != 1.0f) changed = 1;
    ASSERT(changed, "adversarial perturbation applied");

    SNEPPX_tensor_destroy(adv);
    SNEPPX_tensor_destroy(input);
}

int main(void) {
    run_test("arc_config_default", test_arc_config_default);
    run_test("arc_layer_create_destroy", test_arc_layer_create_destroy);
    run_test("arc_forward_pass", test_arc_forward_pass);
    run_test("arc_input_guard", test_arc_input_guard);
    run_test("arc_simulate_attack", test_arc_simulate_attack);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
