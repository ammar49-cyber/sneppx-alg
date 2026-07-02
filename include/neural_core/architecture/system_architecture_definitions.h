#ifndef ARIX_ARCH_H
#define ARIX_ARCH_H

#include "hierarchical_state_space.h"
#include "sparse_expert_routing.h"
#include "adversarial_robustness_certification.h"
#include "neural_programming_engine.h"
#include "fractal_memory_orchestrator.h"
#include "multi_head_attention_module.h"

typedef struct {
    ArixHSSConfig hss_config;
    ArixSERConfig ser_config;
    ArixARCConfig arc_config;
    ArixNPEConfig npe_config;
    ArixFMConfig fm_config;
    ArixAttentionConfig attention_config;
    int enable_attention;
    int enable_hss;
    int enable_ser;
    int enable_arc;
    int enable_npe;
    int enable_fm;
    size_t input_dim;
    size_t output_dim;
    size_t vocab_size;
    unsigned int seed;
} ArixArchConfig;

typedef struct {
    ArixHSSModel* hss_model;
    ArixSERModel* ser_model;
    ArixARCLayer* arc_layer;
    ArixNPEVM* npe_vm;
    ArixNPEProgram* npe_program;
    ArixFMController* fm_controller;
    ArixAttentionWeights* attention;
    ArixTensor* embed_weight;
    ArixTensor* unembed_weight;
    ArixArchConfig config;
} ArixModel;

ArixArchConfig arix_arch_config_default(void);
ArixModel* arix_model_create(const ArixArchConfig* config);
void arix_model_destroy(ArixModel* model);
int arix_model_forward(ArixModel* model, const ArixTensor* input, ArixTensor** output);
size_t arix_model_get_params(const ArixModel* model, ArixTensor** out_params, size_t max_params);
int arix_model_build_train_graph(ArixModel* model, ArixTape* tape,
                                  ArixVariable* input_var,
                                  ArixVariable** weight_vars, size_t num_weights,
                                  ArixVariable** output_var);

#endif /* ARIX_ARCH_H */
