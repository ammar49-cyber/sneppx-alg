#ifndef ARIX_TRAIN_H
#define ARIX_TRAIN_H

#include "system_architecture_definitions.h"
#include "automatic_differentiation_framework.h"
#include "gradient_optimization_suite.h"
#include "multidimensional_tensor_engine.h"
#include <stddef.h>

typedef struct {
    size_t num_epochs;
    size_t batch_size;
    float learning_rate;
    size_t log_interval;
    size_t save_interval;
    ArixDevice device;
} ArixTrainConfig;

typedef struct {
    ArixModel* model;
    ArixOptimizer* optimizer;
    ArixTrainConfig config;
    ArixTensor* loss_history;
    size_t step_count;
} ArixTrainer;

ArixTrainConfig arix_train_config_default(void);
ArixTrainer* arix_trainer_create(ArixModel* model, const ArixTrainConfig* config);
void arix_trainer_destroy(ArixTrainer* trainer);
float arix_trainer_train_step(ArixTrainer* trainer, const ArixTensor* batch_input, const ArixTensor* batch_target);
float arix_trainer_evaluate(ArixTrainer* trainer, const ArixTensor* val_input, const ArixTensor* val_target);
int arix_trainer_save_checkpoint(ArixTrainer* trainer, const char* path);
int arix_trainer_load_checkpoint(ArixTrainer* trainer, const char* path);

#endif /* ARIX_TRAIN_H */
