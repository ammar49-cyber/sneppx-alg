#include "adversarial_robustness_certification.h"
#include "automatic_differentiation_framework.h"
#include "polymorphic_memory_allocator.h"
#include <stdio.h>
#include <math.h>

/*
 * SNEPPX - Test Arc Adversarial Training
 *
 * WHAT
 *   Test Arc Adversarial Training.
 *
 * CONCEPT
 *   Provides the Test Arc Adversarial Training.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


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

static void test_adversarial_training_graph(void) {
    SNEPPXARCConfig cfg = SNEPPX_arc_config_default();
    cfg.attack_epsilon = 0.1f;
    cfg.attack_simulation_types = SNEPPX_ATTACK_FGSM;
    cfg.adversarial_training = 1;
    size_t input_dim = 8;
    size_t output_dim = 4;
    SNEPPXARCLayer* layer = SNEPPX_arc_layer_create(&cfg, input_dim, output_dim, 42);
    ASSERT(layer != NULL, "layer created");

    SNEPPXTape* tape = SNEPPX_tape_create();
    ASSERT(tape != NULL, "tape created");

    size_t shape_in[] = {2, input_dim};
    SNEPPXTensor* input = SNEPPX_tensor_create(shape_in, 2, SNEPPX_FLOAT32);
    float* d = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) d[i] = ((float)(i % 5) - 2.0f) * 0.1f;
    SNEPPXVariable* inv = SNEPPX_variable_create(input, 0);
    SNEPPX_tape_record(tape, inv);

    size_t n_weights = 1 + 2 * layer->output_verifier->num_layers;
    SNEPPXTensor** wt = (SNEPPXTensor**)SNEPPX_malloc(n_weights * sizeof(SNEPPXTensor*), 64);
    ASSERT(wt != NULL, "weight array");
    SNEPPX_arc_get_params(layer, wt, n_weights);

    SNEPPXVariable** wv = (SNEPPXVariable**)SNEPPX_malloc(n_weights * sizeof(SNEPPXVariable*), 64);
    for (size_t i = 0; i < n_weights; i++) {
        wv[i] = SNEPPX_variable_create(wt[i], 1);
        SNEPPX_tape_record(tape, wv[i]);
    }

    SNEPPXVariable* clean_out = NULL;
    SNEPPXVariable* adv_out = NULL;
    int rc = SNEPPX_arc_build_adversarial_train_graph(layer, tape, inv, wv, n_weights, &clean_out, &adv_out);
    ASSERT(rc == 0, "adversarial graph built");
    ASSERT(clean_out != NULL, "clean output");
    ASSERT(adv_out != NULL, "adv output");
    ASSERT(clean_out->data != NULL, "clean data");
    ASSERT(adv_out->data != NULL, "adv data");

    size_t shape_t[] = {2, output_dim};
    SNEPPXTensor* target = SNEPPX_tensor_zeros(shape_t, 2, SNEPPX_FLOAT32);
    SNEPPXVariable* tgv = SNEPPX_variable_create(target, 0);
    SNEPPX_tape_record(tape, tgv);

    SNEPPXVariable* clean_loss = SNEPPX_mse_loss(tape, clean_out, tgv);
    SNEPPXVariable* adv_loss = SNEPPX_mse_loss(tape, adv_out, tgv);
    SNEPPXVariable* combined = SNEPPX_add(tape, clean_loss, adv_loss);
    ASSERT(combined != NULL, "combined loss");

    float loss_before = combined->data ? ((float*)combined->data->data)[0] : 0.0f;

    size_t ones_shape[] = {1};
    SNEPPXTensor* ones = SNEPPX_tensor_ones(ones_shape, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* loss_scalar = SNEPPX_sum_all(tape, combined);
    loss_before = ((float*)loss_scalar->data->data)[0];

    SNEPPX_tape_backward(tape, loss_scalar);

    for (size_t i = 0; i < n_weights; i++) {
        if (wv[i]->grad && wv[i]->grad->data) {
            float* gd = (float*)wv[i]->grad->data;
            for (size_t j = 0; j < wv[i]->grad->size; j++) {
                ASSERT(!isnan(gd[j]), "grad not nan");
            }
        }
    }

    SNEPPXVariable* weights_to_zero = wv;
    for (size_t i = 0; i < n_weights; i++) wv[i]->data = NULL;
    SNEPPX_tape_destroy(tape);
    SNEPPX_free(weights_to_zero, n_weights * sizeof(SNEPPXVariable*));
    SNEPPX_free(wt, n_weights * sizeof(SNEPPXTensor*));
    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(target);
    SNEPPX_arc_layer_destroy(layer);

    printf("  (clean+adv loss before: %f) ", loss_before);
}

int main(void) {
    run_test("ARC adversarial training graph", test_adversarial_training_graph);

    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
