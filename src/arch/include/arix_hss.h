#ifndef ARIX_HSS_H
#define ARIX_HSS_H

#include "arix_tensor.h"
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
} ArixHSSConfig;

typedef struct {
    ArixTensor* A;
    ArixTensor* B;
    ArixTensor* C;
    ArixTensor* D;
    ArixTensor* dt;
    ArixTensor* h;
    ArixTensor* x_proj;
    ArixTensor* x_proj_bias;
    ArixTensor* A_bar;
    ArixTensor* B_bar;
} ArixHSSLayer;

typedef struct {
    ArixHSSLayer** layers;
    ArixHSSConfig config;
    ArixTensor** norm_gamma;
    ArixTensor** norm_beta;
} ArixHSSModel;

ArixHSSConfig arix_hss_config_default(void);
ArixHSSLayer* arix_hss_layer_create(const ArixHSSConfig* config, unsigned int seed);
void arix_hss_layer_destroy(ArixHSSLayer* layer);
ArixHSSModel* arix_hss_model_create(const ArixHSSConfig* config, unsigned int seed);
void arix_hss_model_destroy(ArixHSSModel* model);
int arix_hss_forward(ArixHSSModel* model, const ArixTensor* input, ArixTensor** output);
void arix_hss_discretize(ArixHSSLayer* layer);
void arix_hss_step(const ArixHSSLayer* layer, const ArixTensor* x, ArixTensor* h_next);
void arix_hss_scan(const ArixHSSLayer* layer, const ArixTensor* x_seq, ArixTensor* h_seq, ArixTensor* y_seq);
void arix_hss_hierarchical_scan(const ArixHSSLayer* layer, const ArixTensor* x_seq, ArixTensor* y_seq);

#endif /* ARIX_HSS_H */
