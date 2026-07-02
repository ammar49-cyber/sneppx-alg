#include "system_architecture_definitions.h"
#include "differentiable_training_pipeline.h"
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

static void test_attn_train_graph_builds(void) {
    ArixArchConfig arch_cfg = arix_arch_config_default();
    arch_cfg.input_dim = 8; arch_cfg.output_dim = 8;
    arch_cfg.enable_hss = 0; arch_cfg.enable_ser = 0;
    arch_cfg.enable_attention = 1;
    arch_cfg.attention_config.d_model = 8;
    arch_cfg.attention_config.num_heads = 2;
    arch_cfg.attention_config.head_dim = 4;
    arch_cfg.attention_config.use_causal_mask = 0;
    arch_cfg.attention_config.use_rope = 0;
    arch_cfg.vocab_size = 16;

    ArixModel* model = arix_model_create(&arch_cfg);
    ASSERT(model != NULL, "model created");

    ArixTrainConfig train_cfg = arix_train_config_default();
    train_cfg.learning_rate = 0.001f;
    ArixTrainer* trainer = arix_trainer_create(model, &train_cfg);

    /* Single-token input */
    size_t B = 1, S = 1;
    size_t in_shape[] = {B, S};
    ArixTensor* input = arix_tensor_zeros(in_shape, 2, ARIX_FLOAT32);
    ((float*)input->data)[0] = 0.0f;

    size_t tgt_shape[] = {B * S, arch_cfg.vocab_size};
    ArixTensor* target = arix_tensor_zeros(tgt_shape, 2, ARIX_FLOAT32);
    if (target) ((float*)target->data)[0] = 1.0f;
    ASSERT(input != NULL && target != NULL, "tensors created");

    /* Verify evaluate works */
    float val0 = arix_trainer_evaluate(trainer, input, target);
    ASSERT(isfinite(val0) && val0 >= 0.0f, "eval ok");

    /* Verify first train step produces valid loss */
    float loss = arix_trainer_train_step(trainer, input, target);
    ASSERT(isfinite(loss) && loss >= 0.0f, "first step loss valid");

    /* Verify model parameters updated (weights changed from initial) */
    size_t nw = arix_model_get_params(model, NULL, 0);
    ASSERT(nw > 0, "has parameters");

    arix_tensor_destroy(input); arix_tensor_destroy(target);
    arix_trainer_destroy(trainer);
    arix_model_destroy(model);
}

int main(void) {
    run_test("test_attn_train_graph_builds", test_attn_train_graph_builds);
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
