#ifndef SNEPPX_ARCH_H
#define SNEPPX_ARCH_H

#include "hierarchical_state_space.h"
#include "sparse_expert_routing.h"
#include "adversarial_robustness_certification.h"
#include "neural_programming_engine.h"
#include "fractal_memory_orchestrator.h"
#include "multi_head_attention_module.h"
/*
 * SNEPPX - System Architecture Definitions
 *
 * WHAT
 *   System Architecture Definitions.
 *
 * CONCEPT
 *   Unified configuration and model structs for the HSS-SER-ARC-NPE-FM pipeline.
 *
 * ROLE
 *   Top-level architecture header that wires all algorithm modules together.
 *
 * REFERENCES
 *   None (internal architecture).
 */



typedef struct {
    SNEPPXHSSConfig hss_config;
    SNEPPXSERConfig ser_config;
    SNEPPXARCConfig arc_config;
    SNEPPXNPEConfig npe_config;
    SNEPPXFMConfig fm_config;
    SNEPPXAttentionConfig attention_config;
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
} SNEPPXArchConfig;

typedef struct {
    SNEPPXHSSModel* hss_model;
    SNEPPXSERModel* ser_model;
    SNEPPXARCLayer* arc_layer;
    SNEPPXNPEVM* npe_vm;
    SNEPPXNPEProgram* npe_program;
    SNEPPXFMController* fm_controller;
    SNEPPXAttentionWeights* attention;
    SNEPPXTensor* embed_weight;
    SNEPPXTensor* unembed_weight;
    SNEPPXArchConfig config;
} SNEPPXModel;

SNEPPXArchConfig SNEPPX_arch_config_default(void);
int SNEPPX_arch_has_avx(void);
int SNEPPX_arch_has_avx2(void);
int SNEPPX_arch_has_neon(void);
int SNEPPX_arch_num_cores(void);
int SNEPPX_arch_cache_line_size(void);
SNEPPXModel* SNEPPX_model_create(const SNEPPXArchConfig* config);
void SNEPPX_model_destroy(SNEPPXModel* model);
int SNEPPX_model_forward(SNEPPXModel* model, const SNEPPXTensor* input, SNEPPXTensor** output);
size_t SNEPPX_model_get_params(const SNEPPXModel* model, SNEPPXTensor** out_params, size_t max_params);
int SNEPPX_model_build_train_graph(SNEPPXModel* model, SNEPPXTape* tape,
                                  SNEPPXVariable* input_var,
                                  SNEPPXVariable** weight_vars, size_t num_weights,
                                  SNEPPXVariable** output_var);

#endif /* SNEPPX_ARCH_H */
