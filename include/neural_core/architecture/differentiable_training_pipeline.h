#ifndef SNEPPX_TRAIN_H
#define SNEPPX_TRAIN_H

#include "system_architecture_definitions.h"
#include "automatic_differentiation_framework.h"
#include "gradient_optimization_suite.h"
#include "multidimensional_tensor_engine.h"
#include <stddef.h>
/*
 * SNEPPX - Differentiable Training Pipeline
 *
 * WHAT
 *   Differentiable Training Pipeline.
 *
 * CONCEPT
 *   End-to-end differentiable training pipeline configuration and API.
 *
 * ROLE
 *   Declares the training pipeline API that wraps HSS, SER, and attention modules for end-to-end differentiable training.
 *
 * REFERENCES
 *   None (internal module).
 */



typedef struct {
    size_t num_epochs;
    size_t batch_size;
    float learning_rate;
    size_t log_interval;
    size_t save_interval;
    SNEPPXDevice device;
    int use_cuda_optimizer;
} SNEPPXTrainConfig;

typedef struct {
    SNEPPXModel* model;
    SNEPPXOptimizer* optimizer;
    SNEPPXTrainConfig config;
    SNEPPXTensor* loss_history;
    size_t step_count;
} SNEPPXTrainer;

SNEPPXTrainConfig SNEPPX_train_config_default(void);
SNEPPXTrainer* SNEPPX_trainer_create(SNEPPXModel* model, const SNEPPXTrainConfig* config);
void SNEPPX_trainer_destroy(SNEPPXTrainer* trainer);
float SNEPPX_trainer_train_step(SNEPPXTrainer* trainer, const SNEPPXTensor* batch_input, const SNEPPXTensor* batch_target);
float SNEPPX_trainer_evaluate(SNEPPXTrainer* trainer, const SNEPPXTensor* val_input, const SNEPPXTensor* val_target);
int SNEPPX_trainer_save_checkpoint(SNEPPXTrainer* trainer, const char* path);
int SNEPPX_trainer_load_checkpoint(SNEPPXTrainer* trainer, const char* path);

#endif /* SNEPPX_TRAIN_H */
