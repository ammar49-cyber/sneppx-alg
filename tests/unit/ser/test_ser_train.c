#include "system_architecture_definitions.h"
#include "differentiable_training_pipeline.h"
#include "automatic_differentiation_framework.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

static int tests_passed = 0, tests_failed = 0;
#define ASSERT(cond, msg) do { if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } } while(0)
static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout); fn(); printf("PASS\n"); tests_passed++;
}

static void test_ser_train_step(void) {
    ArixSERConfig cfg = arix_ser_config_default();
    cfg.input_dim = 4;
    cfg.output_dim = 4;
    cfg.expert_dim = 8;
    cfg.num_experts = 4;
    cfg.num_active = 2;

    ArixSERModel* model = arix_ser_model_create(&cfg, 42, 1);
    ASSERT(model != NULL, "model created");

    size_t nw = arix_ser_get_params(model, NULL, 0);
    ASSERT(nw > 0, "has params");

    ArixTensor** params = (ArixTensor**)malloc(nw * sizeof(ArixTensor*));
    arix_ser_get_params(model, params, nw);

    size_t sh[] = {4, 4};
    ArixTensor* input = arix_tensor_randn(sh, 2, ARIX_FLOAT32);
    size_t tsh[] = {4};
    ArixTensor* target = arix_tensor_randn(tsh, 1, ARIX_FLOAT32);

    ArixTape* tape = arix_tape_create();
    ArixVariable** wv = (ArixVariable**)malloc(nw * sizeof(ArixVariable*));
    for (size_t i = 0; i < nw; i++) wv[i] = arix_variable_create(params[i], 1);

    ArixVariable* inv = arix_variable_create(input, 0);
    ArixVariable* outv = NULL;
    int ret = arix_ser_build_train_graph(model, tape, inv, wv, nw, &outv);
    ASSERT(ret == 0, "build graph ok");
    ASSERT(outv != NULL, "output non-null");

    ArixVariable* tgv = arix_variable_create(target, 0);
    ArixVariable* loss_v = arix_mse_loss(tape, outv, tgv);
    ASSERT(loss_v != NULL, "loss non-null");
    float loss = ((float*)loss_v->data->data)[0];
    ASSERT(isfinite(loss), "loss finite");
    ASSERT(loss >= 0.0f, "loss non-negative");

    arix_tensor_destroy(input);
    arix_tensor_destroy(target);
    for (size_t i = 0; i < nw; i++) { wv[i]->data = NULL; wv[i]->grad = NULL; arix_variable_destroy(wv[i]); }
    inv->data = NULL; arix_variable_destroy(inv);
    tgv->data = NULL; arix_variable_destroy(tgv);
    arix_tape_destroy(tape);
    free(wv);
    free(params);
    arix_ser_model_destroy(model);
}

static void test_ser_train_convergence(void) {
    ArixArchConfig arch_cfg = arix_arch_config_default();
    arch_cfg.hss_config.num_layers = 0;
    arch_cfg.hss_config.input_dim = 0;
    arch_cfg.ser_config.input_dim = 4;
    arch_cfg.ser_config.output_dim = 4;
    arch_cfg.ser_config.expert_dim = 8;
    arch_cfg.ser_config.num_experts = 4;
    arch_cfg.ser_config.num_active = 2;
    arch_cfg.input_dim = 4;
    arch_cfg.output_dim = 4;

    ArixModel* model = arix_model_create(&arch_cfg);
    ArixOptimizerConfig opt_cfg = arix_optimizer_config_default();
    opt_cfg.learning_rate = 0.001f;
    opt_cfg.type = ARIX_OPTIMIZER_ADAM;
    opt_cfg.weight_decay = 0.0f;
    opt_cfg.grad_clip = 1.0f;
    ArixOptimizer* opt = arix_optimizer_create(&opt_cfg);

    size_t nw = arix_ser_get_params(model->ser_model, NULL, 0);
    ASSERT(nw > 0, "ser has params");

    ArixTensor** params = (ArixTensor**)malloc(nw * sizeof(ArixTensor*));
    arix_ser_get_params(model->ser_model, params, nw);

    size_t sh[] = {4, 4};
    ArixTensor* input = arix_tensor_randn(sh, 2, ARIX_FLOAT32);
    size_t tsh[] = {4};
    ArixTensor* target = arix_tensor_randn(tsh, 1, ARIX_FLOAT32);

    float init_loss = 0.0f;
    {
        ArixTape* tape = arix_tape_create();
        ArixVariable** wv = (ArixVariable**)malloc(nw * sizeof(ArixVariable*));
        for (size_t i = 0; i < nw; i++) wv[i] = arix_variable_create(params[i], 1);

        ArixVariable* inv = arix_variable_create(input, 0);
        ArixVariable* outv = NULL;
        arix_ser_build_train_graph(model->ser_model, tape, inv, wv, nw, &outv);

        ArixVariable* tgv = arix_variable_create(target, 0);
        ArixVariable* loss_v = arix_mse_loss(tape, outv, tgv);
        if (loss_v) init_loss = ((float*)loss_v->data->data)[0];
        printf("  init loss=%.6f\n", (double)init_loss); fflush(stdout);

        for (size_t i = 0; i < nw; i++) { wv[i]->data = NULL; wv[i]->grad = NULL; arix_variable_destroy(wv[i]); }
        inv->data = NULL; arix_variable_destroy(inv);
        tgv->data = NULL; arix_variable_destroy(tgv);
        arix_tape_destroy(tape);
        free(wv);
    }

    int steps = 30;
    for (int s = 0; s < steps; s++) {
        ArixTape* tape = arix_tape_create();
        ArixVariable** wv = (ArixVariable**)malloc(nw * sizeof(ArixVariable*));
        for (size_t i = 0; i < nw; i++) wv[i] = arix_variable_create(params[i], 1);

        ArixVariable* inv = arix_variable_create(input, 0);
        ArixVariable* outv = NULL;
        arix_ser_build_train_graph(model->ser_model, tape, inv, wv, nw, &outv);

        ArixVariable* tgv = arix_variable_create(target, 0);
        ArixVariable* loss_v = arix_mse_loss(tape, outv, tgv);

        if (loss_v) {
            arix_tape_backward(tape, loss_v);
            ArixTensor** grads = (ArixTensor**)malloc(nw * sizeof(ArixTensor*));
            for (size_t i = 0; i < nw; i++) grads[i] = wv[i]->grad;
            arix_optimizer_step(opt, params, grads, nw);
            free(grads);
        }

        for (size_t i = 0; i < nw; i++) { wv[i]->data = NULL; wv[i]->grad = NULL; arix_variable_destroy(wv[i]); }
        inv->data = NULL; arix_variable_destroy(inv);
        tgv->data = NULL; arix_variable_destroy(tgv);
        arix_tape_destroy(tape);
        free(wv);
    }
    float final_loss = 0.0f;
    {
        ArixTape* tape = arix_tape_create();
        ArixVariable** wv = (ArixVariable**)malloc(nw * sizeof(ArixVariable*));
        for (size_t i = 0; i < nw; i++) wv[i] = arix_variable_create(params[i], 1);

        ArixVariable* inv = arix_variable_create(input, 0);
        ArixVariable* outv = NULL;
        arix_ser_build_train_graph(model->ser_model, tape, inv, wv, nw, &outv);

        ArixVariable* tgv = arix_variable_create(target, 0);
        ArixVariable* loss_v = arix_mse_loss(tape, outv, tgv);

        if (loss_v) final_loss = ((float*)loss_v->data->data)[0];
        printf("  final loss=%.6f ratio=%.4f\n", (double)final_loss, (double)(final_loss / (init_loss + 1e-10f)));

        for (size_t i = 0; i < nw; i++) { wv[i]->data = NULL; wv[i]->grad = NULL; arix_variable_destroy(wv[i]); }
        inv->data = NULL; arix_variable_destroy(inv);
        tgv->data = NULL; arix_variable_destroy(tgv);
        arix_tape_destroy(tape);
        free(wv);
    }

    ASSERT(final_loss < init_loss * 0.9f, "loss decreased >10%%");

    arix_tensor_destroy(input);
    arix_tensor_destroy(target);
    arix_optimizer_destroy(opt);
    arix_model_destroy(model);
    free(params);
}

int main(void) {
    run_test("test_ser_train_step", test_ser_train_step);
    run_test("test_ser_train_convergence", test_ser_train_convergence);
    printf("\nSER training tests: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
