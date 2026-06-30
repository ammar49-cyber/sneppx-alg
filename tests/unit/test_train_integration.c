#include "hierarchical_state_space.h"
#include "automatic_differentiation_framework.h"
#include "gradient_optimization_suite.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int tests_passed = 0, tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } \
} while(0)

static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout);
    fn(); printf("PASS\n"); tests_passed++;
}

static void test_hss_train_step(void) {
    size_t s_dim = 4, i_dim = 4, o_dim = 4, seq_len = 1;

    ArixHSSConfig cfg;
    cfg.state_dim = s_dim; cfg.input_dim = i_dim; cfg.output_dim = o_dim;
    cfg.num_layers = 1; cfg.seq_len = seq_len;
    cfg.dt_min = 0.01f; cfg.dt_max = 0.1f; cfg.use_hierarchical = 0;

    ArixHSSModel* model = arix_hss_model_create(&cfg, 42);
    ASSERT(model != NULL, "model created");

    size_t shape_in[] = {seq_len, i_dim};
    ArixTensor* input = arix_tensor_randn(shape_in, 2, ARIX_FLOAT32);
    size_t shape_tgt[] = {o_dim};
    ArixTensor* target = arix_tensor_randn(shape_tgt, 1, ARIX_FLOAT32);
    ASSERT(input != NULL && target != NULL, "tensors created");

    ArixTensor* initial_out = NULL;
    int ret = arix_hss_forward(model, input, &initial_out);
    ASSERT(ret == 0 && initial_out != NULL, "initial forward ok");

    float init_loss = 0.0f;
    {
        float* od = (float*)initial_out->data;
        float* td = (float*)target->data;
        size_t n = initial_out->size < o_dim ? initial_out->size : o_dim;
        for (size_t i = 0; i < n; i++) {
            float d = od[i] - td[i];
            init_loss += d * d;
        }
        init_loss /= (float)n;
    }
    arix_tensor_destroy(initial_out);

    ArixOptimizerConfig opt_cfg = arix_optimizer_config_default();
    opt_cfg.learning_rate = 0.01f;
    opt_cfg.type = ARIX_OPTIMIZER_ADAM;
    opt_cfg.weight_decay = 0.0f;
    ArixOptimizer* opt = arix_optimizer_create(&opt_cfg);

    size_t nw = arix_hss_get_params(model, NULL, 0);
    ASSERT(nw == 9, "9 params");

    ArixTensor** pt = (ArixTensor**)malloc(nw * sizeof(ArixTensor*));
    arix_hss_get_params(model, pt, nw);

    int steps = 50;
    for (int s = 0; s < steps; s++) {
        ArixTape* tape = arix_tape_create();

        ArixVariable** wv = (ArixVariable**)malloc(nw * sizeof(ArixVariable*));
        for (size_t i = 0; i < nw; i++) wv[i] = arix_variable_create(pt[i], 1);

        ArixVariable* inv = arix_variable_create(input, 0);

        ArixVariable* outv = NULL;
        ret = arix_hss_build_train_graph(model, tape, inv, wv, nw, &outv);
        ASSERT(ret == 0, "build graph");

        ArixVariable* tgv = arix_variable_create(target, 0);

        ArixVariable* loss = arix_mse_loss(tape, outv, tgv);
        ASSERT(loss != NULL, "loss");

        arix_tape_backward(tape, loss);

        ArixTensor** gt = (ArixTensor**)malloc(nw * sizeof(ArixTensor*));
        for (size_t i = 0; i < nw; i++) gt[i] = wv[i]->grad;

        arix_optimizer_step(opt, pt, gt, nw);

        for (size_t i = 0; i < nw; i++) { wv[i]->data = NULL; wv[i]->grad = NULL; arix_variable_destroy(wv[i]); }
        inv->data = NULL; arix_variable_destroy(inv);
        tgv->data = NULL; arix_variable_destroy(tgv);

        arix_tape_destroy(tape);
        free(wv);
        free(gt);
    }
    free(pt);

    ArixTensor* final_out = NULL;
    arix_hss_forward(model, input, &final_out);
    ASSERT(final_out != NULL, "final forward");

    float final_loss = 0.0f;
    {
        float* od = (float*)final_out->data;
        float* td = (float*)target->data;
        size_t n = final_out->size < o_dim ? final_out->size : o_dim;
        for (size_t i = 0; i < n; i++) {
            float d = od[i] - td[i];
            final_loss += d * d;
        }
        final_loss /= (float)n;
    }

    printf("  init=%.6f final=%.6f ratio=%.4f\n",
           (double)init_loss, (double)final_loss,
           (double)(final_loss / (init_loss + 1e-10f)));
    ASSERT(final_loss < init_loss * 0.5f, "loss decreased >50%%");

    arix_tensor_destroy(final_out);
    arix_tensor_destroy(input);
    arix_tensor_destroy(target);
    arix_optimizer_destroy(opt);
    arix_hss_model_destroy(model);
}

int main(void) {
    run_test("test_hss_train_step", test_hss_train_step);
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
