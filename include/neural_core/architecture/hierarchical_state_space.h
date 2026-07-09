#ifndef SNEPPX_HSS_H
#define SNEPPX_HSS_H

#include "multidimensional_tensor_engine.h"
#include "automatic_differentiation_framework.h"
#include <stddef.h>

typedef struct {
    size_t state_dim;
    size_t input_dim;
    size_t output_dim;
    size_t num_layers;
    size_t seq_len;
    float dt_min;
    float dt_max;
    int use_hierarchical;
    int use_parallel_scan;
} SNEPPXHSSConfig;

typedef struct {
    SNEPPXTensor* A;
    SNEPPXTensor* B;
    SNEPPXTensor* C;
    SNEPPXTensor* D;
    SNEPPXTensor* dt;
    SNEPPXTensor* h;
    SNEPPXTensor* x_proj;
    SNEPPXTensor* x_proj_bias;
    SNEPPXTensor* A_bar;
    SNEPPXTensor* B_bar;
} SNEPPXHSSLayer;

typedef struct {
    SNEPPXHSSLayer** layers;
    SNEPPXHSSConfig config;
    SNEPPXTensor** norm_gamma;
    SNEPPXTensor** norm_beta;
} SNEPPXHSSModel;

SNEPPXHSSConfig SNEPPX_hss_config_default(void);
SNEPPXHSSLayer* SNEPPX_hss_layer_create(const SNEPPXHSSConfig* config, unsigned int seed);
void SNEPPX_hss_layer_destroy(SNEPPXHSSLayer* layer);
SNEPPXHSSModel* SNEPPX_hss_model_create(const SNEPPXHSSConfig* config, unsigned int seed);
void SNEPPX_hss_model_destroy(SNEPPXHSSModel* model);
int SNEPPX_hss_forward(SNEPPXHSSModel* model, const SNEPPXTensor* input, SNEPPXTensor** output);
void SNEPPX_hss_discretize(SNEPPXHSSLayer* layer);
void SNEPPX_hss_step(const SNEPPXHSSLayer* layer, const SNEPPXTensor* x, SNEPPXTensor* h_next);
void SNEPPX_hss_scan(const SNEPPXHSSLayer* layer, const SNEPPXTensor* x_seq, SNEPPXTensor* h_seq, SNEPPXTensor* y_seq);
void SNEPPX_hss_parallel_scan(const SNEPPXHSSLayer* layer, const SNEPPXTensor* x_seq, SNEPPXTensor* h_seq, SNEPPXTensor* y_seq);
void SNEPPX_hss_hierarchical_scan(const SNEPPXHSSLayer* layer, const SNEPPXTensor* x_seq, SNEPPXTensor* y_seq);
size_t SNEPPX_hss_get_params(const SNEPPXHSSModel* model, SNEPPXTensor** out_params, size_t max_params);
int SNEPPX_hss_build_train_graph(SNEPPXHSSModel* model, SNEPPXTape* tape,
                               SNEPPXVariable* input_var,
                               SNEPPXVariable** weight_vars, size_t num_weights,
                               SNEPPXVariable** output_var);

#endif /* SNEPPX_HSS_H */
