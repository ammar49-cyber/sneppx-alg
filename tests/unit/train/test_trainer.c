#include "arix_arch.h"
#include "arix_train.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

static int tests_passed = 0, tests_failed = 0;
#define ASSERT(cond, msg) do { if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } } while(0)
static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name); fflush(stdout); fn(); printf("PASS\n"); tests_passed++;
}

static void test_trainer_create(void) {
    ArixArchConfig arch_cfg = arix_arch_config_default();
    arch_cfg.input_dim = 8; arch_cfg.output_dim = 8;
    arch_cfg.hss_config.input_dim = 8; arch_cfg.hss_config.output_dim = 8;
    arch_cfg.hss_config.state_dim = 4;
    arch_cfg.ser_config.input_dim = 8; arch_cfg.ser_config.output_dim = 8;
    arch_cfg.ser_config.expert_dim = 16;
    arch_cfg.fm_config.memory_dim = 8;
    arch_cfg.fm_config.memory_capacity = 16;

    ArixModel* model = arix_model_create(&arch_cfg);
    ASSERT(model != NULL, "model created");

    ArixTrainConfig train_cfg = arix_train_config_default();
    train_cfg.learning_rate = 0.01f;
    train_cfg.log_interval = 100;

    ArixTrainer* trainer = arix_trainer_create(model, &train_cfg);
    ASSERT(trainer != NULL, "trainer created");
    ASSERT(trainer->model == model, "model set");
    ASSERT(trainer->optimizer != NULL, "optimizer created");
    ASSERT(trainer->loss_history != NULL, "loss history");

    arix_trainer_destroy(trainer);
    arix_model_destroy(model);
}

static void test_train_step(void) {
    ArixArchConfig arch_cfg = arix_arch_config_default();
    arch_cfg.input_dim = 8; arch_cfg.output_dim = 8;
    arch_cfg.hss_config.input_dim = 8; arch_cfg.hss_config.output_dim = 8;
    arch_cfg.hss_config.state_dim = 4;
    arch_cfg.ser_config.input_dim = 8; arch_cfg.ser_config.output_dim = 8;
    arch_cfg.ser_config.expert_dim = 16;
    arch_cfg.fm_config.memory_dim = 8;
    arch_cfg.fm_config.memory_capacity = 16;

    ArixModel* model = arix_model_create(&arch_cfg);
    ArixTrainConfig train_cfg = arix_train_config_default();
    train_cfg.log_interval = 100;
    ArixTrainer* trainer = arix_trainer_create(model, &train_cfg);

    size_t in_shape[] = {1, 4, 8};
    ArixTensor* input = arix_tensor_randn(in_shape, 3, ARIX_FLOAT32);
    size_t tgt_shape[] = {1, 4, 8};
    ArixTensor* target = arix_tensor_randn(tgt_shape, 3, ARIX_FLOAT32);

    float loss = arix_trainer_train_step(trainer, input, target);
    ASSERT(isfinite(loss), "loss finite");
    ASSERT(loss >= 0.0f, "loss non-negative");
    ASSERT(trainer->step_count == 1, "step count 1");

    float* lh = (float*)trainer->loss_history->data;
    ASSERT(fabsf(lh[0] - loss) < 1e-5f, "loss recorded");

    arix_tensor_destroy(input);
    arix_tensor_destroy(target);
    arix_trainer_destroy(trainer);
    arix_model_destroy(model);
}

static void test_evaluate(void) {
    ArixArchConfig arch_cfg = arix_arch_config_default();
    arch_cfg.input_dim = 8; arch_cfg.output_dim = 8;
    arch_cfg.hss_config.input_dim = 8; arch_cfg.hss_config.output_dim = 8;
    arch_cfg.hss_config.state_dim = 4;
    arch_cfg.ser_config.input_dim = 8; arch_cfg.ser_config.output_dim = 8;
    arch_cfg.ser_config.expert_dim = 16;
    arch_cfg.fm_config.memory_dim = 8;
    arch_cfg.fm_config.memory_capacity = 16;

    ArixModel* model = arix_model_create(&arch_cfg);
    ArixTrainConfig train_cfg = arix_train_config_default();
    ArixTrainer* trainer = arix_trainer_create(model, &train_cfg);

    size_t sh[] = {1, 4, 8};
    ArixTensor* input = arix_tensor_randn(sh, 3, ARIX_FLOAT32);
    ArixTensor* target = arix_tensor_randn(sh, 3, ARIX_FLOAT32);

    float loss = arix_trainer_evaluate(trainer, input, target);
    ASSERT(isfinite(loss), "eval loss finite");
    ASSERT(loss >= 0.0f, "eval loss non-negative");

    arix_tensor_destroy(input);
    arix_tensor_destroy(target);
    arix_trainer_destroy(trainer);
    arix_model_destroy(model);
}

static void test_checkpoint(void) {
    ArixArchConfig arch_cfg = arix_arch_config_default();
    arch_cfg.input_dim = 8; arch_cfg.output_dim = 8;
    arch_cfg.hss_config.input_dim = 8; arch_cfg.hss_config.output_dim = 8;
    arch_cfg.hss_config.state_dim = 4;
    arch_cfg.ser_config.input_dim = 8; arch_cfg.ser_config.output_dim = 8;
    arch_cfg.ser_config.expert_dim = 16;
    arch_cfg.fm_config.memory_dim = 8;
    arch_cfg.fm_config.memory_capacity = 16;

    ArixModel* model = arix_model_create(&arch_cfg);
    ArixTrainConfig train_cfg = arix_train_config_default();
    ArixTrainer* trainer = arix_trainer_create(model, &train_cfg);

    int r = arix_trainer_save_checkpoint(trainer, "test_checkpoint.bin");
    ASSERT(r == 0, "save ok");

    ArixModel* model2 = arix_model_create(&arch_cfg);
    ArixTrainer* trainer2 = arix_trainer_create(model2, &train_cfg);
    r = arix_trainer_load_checkpoint(trainer2, "test_checkpoint.bin");
    ASSERT(r == 0, "load ok");
    ASSERT(trainer2->step_count == trainer->step_count, "step count restored");

    remove("test_checkpoint.bin");
    arix_trainer_destroy(trainer);
    arix_trainer_destroy(trainer2);
    arix_model_destroy(model);
    arix_model_destroy(model2);
}

int main(void) {
    run_test("test_trainer_create", test_trainer_create);
    run_test("test_train_step", test_train_step);
    run_test("test_evaluate", test_evaluate);
    run_test("test_checkpoint", test_checkpoint);
    printf("\nTraining tests: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
