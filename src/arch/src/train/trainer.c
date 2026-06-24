#include "arix_train.h"
#include "arix_memory.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

ArixTrainConfig arix_train_config_default(void) {
    ArixTrainConfig cfg;
    cfg.num_epochs = 10;
    cfg.batch_size = 32;
    cfg.learning_rate = 0.01f;
    cfg.log_interval = 10;
    cfg.save_interval = 100;
    cfg.device = ARIX_DEVICE_CPU;
    return cfg;
}

ArixTrainer* arix_trainer_create(ArixModel* model, const ArixTrainConfig* config) {
    if (!model || !config) return NULL;
    ArixTrainer* trainer = (ArixTrainer*)arix_malloc(sizeof(ArixTrainer), 64);
    if (!trainer) return NULL;
    memset(trainer, 0, sizeof(ArixTrainer));

    trainer->model = model;
    trainer->config = *config;

    ArixOptimizerConfig opt_cfg = arix_optimizer_config_default();
    opt_cfg.learning_rate = config->learning_rate;
    trainer->optimizer = arix_optimizer_create(&opt_cfg);
    if (!trainer->optimizer) { arix_free(trainer, sizeof(ArixTrainer)); return NULL; }

    size_t hist_shape[] = {1024};
    trainer->loss_history = arix_tensor_zeros(hist_shape, 1, ARIX_FLOAT32);
    if (!trainer->loss_history) {
        arix_optimizer_destroy(trainer->optimizer);
        arix_free(trainer, sizeof(ArixTrainer));
        return NULL;
    }

    trainer->step_count = 0;
    return trainer;
}

void arix_trainer_destroy(ArixTrainer* trainer) {
    if (!trainer) return;
    if (trainer->optimizer) arix_optimizer_destroy(trainer->optimizer);
    if (trainer->loss_history) arix_tensor_destroy(trainer->loss_history);
    arix_free(trainer, sizeof(ArixTrainer));
}

float arix_trainer_train_step(ArixTrainer* trainer, const ArixTensor* batch_input, const ArixTensor* batch_target) {
    if (!trainer || !batch_input || !batch_target) return -1.0f;

    ArixTensor* output = NULL;
    int ret = arix_model_forward(trainer->model, batch_input, &output);
    if (ret != 0 || !output) return -1.0f;

    float* od = (float*)output->data;
    float* td = (float*)batch_target->data;
    size_t sz = output->size < batch_target->size ? output->size : batch_target->size;

    float sum_sq = 0.0f;
    for (size_t i = 0; i < sz; i++) {
        float d = od[i] - td[i];
        sum_sq += d * d;
    }
    float loss = sum_sq / (float)(sz > 0 ? sz : 1);

    size_t idx = trainer->step_count;
    if (idx < trainer->loss_history->size) {
        ((float*)trainer->loss_history->data)[idx] = loss;
    }

    trainer->step_count++;
    arix_tensor_destroy(output);

    if (trainer->config.log_interval > 0 && trainer->step_count % trainer->config.log_interval == 0) {
        printf("[Step %zu] loss = %.6f\n", trainer->step_count, (double)loss);
    }

    return loss;
}

float arix_trainer_evaluate(ArixTrainer* trainer, const ArixTensor* val_input, const ArixTensor* val_target) {
    if (!trainer || !val_input || !val_target) return -1.0f;

    ArixTensor* output = NULL;
    int ret = arix_model_forward(trainer->model, val_input, &output);
    if (ret != 0 || !output) return -1.0f;

    float* od = (float*)output->data;
    float* td = (float*)val_target->data;
    size_t sz = output->size < val_target->size ? output->size : val_target->size;

    float sum_sq = 0.0f;
    for (size_t i = 0; i < sz; i++) {
        float d = od[i] - td[i];
        sum_sq += d * d;
    }
    float loss = sum_sq / (float)(sz > 0 ? sz : 1);

    arix_tensor_destroy(output);
    return loss;
}

int arix_trainer_save_checkpoint(ArixTrainer* trainer, const char* path) {
    if (!trainer || !path) return 1;
    FILE* f = fopen(path, "wb");
    if (!f) return 1;

    unsigned int magic = 0x41524958;
    unsigned int version = 1;
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&version, sizeof(version), 1, f);
    fwrite(&trainer->step_count, sizeof(trainer->step_count), 1, f);
    fclose(f);
    return 0;
}

int arix_trainer_load_checkpoint(ArixTrainer* trainer, const char* path) {
    if (!trainer || !path) return 1;
    FILE* f = fopen(path, "rb");
    if (!f) return 1;

    unsigned int magic, version;
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != 0x41524958) {
        fclose(f); return 1;
    }
    if (fread(&version, sizeof(version), 1, f) != 1 || version != 1) {
        fclose(f); return 1;
    }
    if (fread(&trainer->step_count, sizeof(trainer->step_count), 1, f) != 1) {
        fclose(f); return 1;
    }
    fclose(f);
    return 0;
}
