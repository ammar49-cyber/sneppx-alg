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

    SNEPPXHSSConfig cfg;
    cfg.state_dim = s_dim; cfg.input_dim = i_dim; cfg.output_dim = o_dim;
    cfg.num_layers = 1; cfg.seq_len = seq_len;
    cfg.dt_min = 0.01f; cfg.dt_max = 0.1f; cfg.use_hierarchical = 0;

    SNEPPXHSSModel* model = SNEPPX_hss_model_create(&cfg, 42);
    ASSERT(model != NULL, "model created");

    size_t shape_in[] = {seq_len, i_dim};
    SNEPPXTensor* input = SNEPPX_tensor_randn(shape_in, 2, SNEPPX_FLOAT32);
    size_t shape_tgt[] = {o_dim};
    SNEPPXTensor* target = SNEPPX_tensor_randn(shape_tgt, 1, SNEPPX_FLOAT32);
    ASSERT(input != NULL && target != NULL, "tensors created");

    SNEPPXTensor* initial_out = NULL;
    int ret = SNEPPX_hss_forward(model, input, &initial_out);
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
    SNEPPX_tensor_destroy(initial_out);

    SNEPPXOptimizerConfig opt_cfg = SNEPPX_optimizer_config_default();
    opt_cfg.learning_rate = 0.01f;
    opt_cfg.type = SNEPPX_OPTIMIZER_ADAM;
    opt_cfg.weight_decay = 0.0f;
    SNEPPXOptimizer* opt = SNEPPX_optimizer_create(&opt_cfg);

    size_t nw = SNEPPX_hss_get_params(model, NULL, 0);
    ASSERT(nw == 9, "9 params");

    SNEPPXTensor** pt = (SNEPPXTensor**)malloc(nw * sizeof(SNEPPXTensor*));
    SNEPPX_hss_get_params(model, pt, nw);

    int steps = 50;
    for (int s = 0; s < steps; s++) {
        SNEPPXTape* tape = SNEPPX_tape_create();

        SNEPPXVariable** wv = (SNEPPXVariable**)malloc(nw * sizeof(SNEPPXVariable*));
        for (size_t i = 0; i < nw; i++) wv[i] = SNEPPX_variable_create(pt[i], 1);

        SNEPPXVariable* inv = SNEPPX_variable_create(input, 0);

        SNEPPXVariable* outv = NULL;
        ret = SNEPPX_hss_build_train_graph(model, tape, inv, wv, nw, &outv);
        ASSERT(ret == 0, "build graph");

        SNEPPXVariable* tgv = SNEPPX_variable_create(target, 0);

        SNEPPXVariable* loss = SNEPPX_mse_loss(tape, outv, tgv);
        ASSERT(loss != NULL, "loss");

        SNEPPX_tape_backward(tape, loss);

        SNEPPXTensor** gt = (SNEPPXTensor**)malloc(nw * sizeof(SNEPPXTensor*));
        for (size_t i = 0; i < nw; i++) gt[i] = wv[i]->grad;

        SNEPPX_optimizer_step(opt, pt, gt, nw);

        for (size_t i = 0; i < nw; i++) { wv[i]->data = NULL; wv[i]->grad = NULL; SNEPPX_variable_destroy(wv[i]); }
        inv->data = NULL; SNEPPX_variable_destroy(inv);
        tgv->data = NULL; SNEPPX_variable_destroy(tgv);

        SNEPPX_tape_destroy(tape);
        free(wv);
        free(gt);
    }
    free(pt);

    SNEPPXTensor* final_out = NULL;
    SNEPPX_hss_forward(model, input, &final_out);
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

    SNEPPX_tensor_destroy(final_out);
    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(target);
    SNEPPX_optimizer_destroy(opt);
    SNEPPX_hss_model_destroy(model);
}

int main(void) {
    run_test("test_hss_train_step", test_hss_train_step);
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
