#ifndef ARIX_ARCH_H
#define ARIX_ARCH_H

#include "hierarchical_state_space.h"
#include "sparse_expert_routing.h"
#include "adversarial_robustness_certification.h"
#include "neural_programming_engine.h"
#include "fractal_memory_orchestrator.h"

typedef struct {
    ArixHSSConfig hss_config;
    ArixSERConfig ser_config;
    ArixARCConfig arc_config;
    ArixNPEConfig npe_config;
    ArixFMConfig fm_config;
    size_t input_dim;
    size_t output_dim;
    unsigned int seed;
} ArixArchConfig;

typedef struct {
    ArixHSSModel* hss_model;
    ArixSERModel* ser_model;
    ArixARCLayer* arc_layer;
    ArixNPEVM* npe_vm;
    ArixNPEProgram* npe_program;
    ArixFMController* fm_controller;
    ArixArchConfig config;
} ArixModel;

ArixArchConfig arix_arch_config_default(void);
ArixModel* arix_model_create(const ArixArchConfig* config);
void arix_model_destroy(ArixModel* model);
int arix_model_forward(ArixModel* model, const ArixTensor* input, ArixTensor** output);
size_t arix_model_get_params(const ArixModel* model, ArixTensor** out_params, size_t max_params);

#endif /* ARIX_ARCH_H */
