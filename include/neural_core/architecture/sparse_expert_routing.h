#ifndef ARIX_SER_H
#define ARIX_SER_H

#include "multidimensional_tensor_engine.h"
#include "automatic_differentiation_framework.h"
#include <stddef.h>

typedef enum {
    ARIX_ACT_RELU,
    ARIX_ACT_GELU,
    ARIX_ACT_SWISH
} ArixActivation;

typedef enum {
    ARIX_TOPK_GREEDY,
    ARIX_TOPK_NOISY
} ArixTopKMethod;

typedef struct {
    size_t num_experts;
    size_t num_active;
    size_t input_dim;
    size_t expert_dim;
    size_t output_dim;
    ArixTopKMethod top_k_method;
    float load_balance_coef;
    float dropout_rate;
} ArixSERConfig;

typedef struct {
    ArixTensor* w1;
    ArixTensor* w2;
    ArixTensor* b1;
    ArixTensor* b2;
    ArixActivation activation;
} ArixExpert;

typedef struct {
    ArixExpert** experts;
    ArixTensor* router;
    ArixTensor* router_bias;
    ArixSERConfig config;
    size_t expert_capacity;
} ArixSERLayer;

typedef struct {
    ArixSERLayer** layers;
    size_t num_layers;
    ArixSERConfig config;
} ArixSERModel;

ArixSERConfig arix_ser_config_default(void);
ArixExpert* arix_expert_create(const ArixSERConfig* config, unsigned int seed, ArixActivation activation);
void arix_expert_destroy(ArixExpert* expert);
ArixSERLayer* arix_ser_layer_create(const ArixSERConfig* config, unsigned int seed);
void arix_ser_layer_destroy(ArixSERLayer* layer);
ArixSERModel* arix_ser_model_create(const ArixSERConfig* config, unsigned int seed, size_t num_layers);
void arix_ser_model_destroy(ArixSERModel* model);
void arix_ser_route(ArixSERLayer* layer, const ArixTensor* input, ArixTensor** gate_weights, int** expert_indices);
void arix_ser_gate_forward(const ArixSERLayer* layer, const ArixTensor* input,
                           ArixTensor** gate_weights, int** expert_indices,
                           ArixTensor** gate_logits, float temperature);
void arix_ser_expert_forward(const ArixExpert* expert, const ArixTensor* input, ArixTensor* output);
void arix_ser_forward(ArixSERLayer* layer, const ArixTensor* input, ArixTensor** output);
float arix_ser_load_balance_loss(const ArixTensor* gate_weights, const int* expert_indices, size_t num_tokens);
float arix_ser_z_loss(const ArixTensor* gate_logits);
float arix_ser_aux_loss(const ArixTensor* gate_weights, const int* expert_indices,
                        const ArixTensor* gate_logits, size_t num_tokens,
                        float load_balance_coef, float z_loss_coef);
void arix_ser_expert_capacity_balance(ArixTensor* gate_weights, int* expert_indices,
                                      size_t num_tokens, size_t num_active,
                                      size_t expert_capacity);
size_t arix_ser_get_params(const ArixSERModel* model, ArixTensor** out_params, size_t max_params);
int arix_ser_build_train_graph(ArixSERModel* model, ArixTape* tape,
                               ArixVariable* input_var,
                               ArixVariable** weight_vars, size_t num_weights,
                               ArixVariable** output_var);

#endif /* ARIX_SER_H */
