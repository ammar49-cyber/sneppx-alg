#ifndef SNEPPX_SER_H
#define SNEPPX_SER_H

#include "multidimensional_tensor_engine.h"
#include "automatic_differentiation_framework.h"
#include <stddef.h>

typedef enum {
    SNEPPX_ACT_RELU,
    SNEPPX_ACT_GELU,
    SNEPPX_ACT_SWISH
} SNEPPXActivation;

typedef enum {
    SNEPPX_TOPK_GREEDY,
    SNEPPX_TOPK_NOISY
} SNEPPXTopKMethod;

typedef struct {
    size_t num_experts;
    size_t num_active;
    size_t input_dim;
    size_t expert_dim;
    size_t output_dim;
    SNEPPXTopKMethod top_k_method;
    float load_balance_coef;
    float dropout_rate;
} SNEPPXSERConfig;

typedef struct {
    SNEPPXTensor* w1;
    SNEPPXTensor* w2;
    SNEPPXTensor* b1;
    SNEPPXTensor* b2;
    SNEPPXActivation activation;
} SNEPPXExpert;

typedef struct {
    SNEPPXExpert** experts;
    SNEPPXTensor* router;
    SNEPPXTensor* router_bias;
    SNEPPXSERConfig config;
    size_t expert_capacity;
} SNEPPXSERLayer;

typedef struct {
    SNEPPXSERLayer** layers;
    size_t num_layers;
    SNEPPXSERConfig config;
} SNEPPXSERModel;

SNEPPXSERConfig SNEPPX_ser_config_default(void);
SNEPPXExpert* SNEPPX_expert_create(const SNEPPXSERConfig* config, unsigned int seed, SNEPPXActivation activation);
void SNEPPX_expert_destroy(SNEPPXExpert* expert);
SNEPPXSERLayer* SNEPPX_ser_layer_create(const SNEPPXSERConfig* config, unsigned int seed);
void SNEPPX_ser_layer_destroy(SNEPPXSERLayer* layer);
SNEPPXSERModel* SNEPPX_ser_model_create(const SNEPPXSERConfig* config, unsigned int seed, size_t num_layers);
void SNEPPX_ser_model_destroy(SNEPPXSERModel* model);
void SNEPPX_ser_route(SNEPPXSERLayer* layer, const SNEPPXTensor* input, SNEPPXTensor** gate_weights, int** expert_indices);
void SNEPPX_ser_gate_forward(const SNEPPXSERLayer* layer, const SNEPPXTensor* input,
                           SNEPPXTensor** gate_weights, int** expert_indices,
                           SNEPPXTensor** gate_logits, float temperature);
void SNEPPX_ser_expert_forward(const SNEPPXExpert* expert, const SNEPPXTensor* input, SNEPPXTensor* output);
void SNEPPX_ser_forward(SNEPPXSERLayer* layer, const SNEPPXTensor* input, SNEPPXTensor** output);
float SNEPPX_ser_load_balance_loss(const SNEPPXTensor* gate_weights, const int* expert_indices, size_t num_tokens);
float SNEPPX_ser_z_loss(const SNEPPXTensor* gate_logits);
float SNEPPX_ser_aux_loss(const SNEPPXTensor* gate_weights, const int* expert_indices,
                        const SNEPPXTensor* gate_logits, size_t num_tokens,
                        float load_balance_coef, float z_loss_coef);
void SNEPPX_ser_expert_capacity_balance(SNEPPXTensor* gate_weights, int* expert_indices,
                                      size_t num_tokens, size_t num_active,
                                      size_t expert_capacity);
size_t SNEPPX_ser_get_params(const SNEPPXSERModel* model, SNEPPXTensor** out_params, size_t max_params);
int SNEPPX_ser_build_train_graph(SNEPPXSERModel* model, SNEPPXTape* tape,
                               SNEPPXVariable* input_var,
                               SNEPPXVariable** weight_vars, size_t num_weights,
                               SNEPPXVariable** output_var);

#endif /* SNEPPX_SER_H */
