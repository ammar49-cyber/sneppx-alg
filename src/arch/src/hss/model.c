#include "arix_hss.h"
#include "arix_memory.h"
#include <stdlib.h>
#include <string.h>

ArixHSSModel* arix_hss_model_create(const ArixHSSConfig* config, unsigned int seed) {
    ArixHSSModel* model = (ArixHSSModel*)arix_malloc(sizeof(ArixHSSModel), 64);
    if (!model) return NULL;
    memset(model, 0, sizeof(ArixHSSModel));

    model->config = *config;

    model->layers = (ArixHSSLayer**)arix_malloc(config->num_layers * sizeof(ArixHSSLayer*), 64);
    if (!model->layers) {
        arix_free(model, sizeof(ArixHSSModel));
        return NULL;
    }
    memset(model->layers, 0, config->num_layers * sizeof(ArixHSSLayer*));

    model->norm_gamma = (ArixTensor**)arix_malloc(config->num_layers * sizeof(ArixTensor*), 64);
    model->norm_beta = (ArixTensor**)arix_malloc(config->num_layers * sizeof(ArixTensor*), 64);
    if (!model->norm_gamma || !model->norm_beta) {
        arix_free(model->layers, config->num_layers * sizeof(ArixHSSLayer*));
        arix_free(model->norm_gamma, config->num_layers * sizeof(ArixTensor*));
        arix_free(model->norm_beta, config->num_layers * sizeof(ArixTensor*));
        arix_free(model, sizeof(ArixHSSModel));
        return NULL;
    }
    memset(model->norm_gamma, 0, config->num_layers * sizeof(ArixTensor*));
    memset(model->norm_beta, 0, config->num_layers * sizeof(ArixTensor*));

    for (size_t i = 0; i < config->num_layers; i++) {
        model->layers[i] = arix_hss_layer_create(config, seed + (unsigned int)i);

        size_t shape_dim[] = {config->input_dim};
        model->norm_gamma[i] = arix_tensor_ones(shape_dim, 1, ARIX_FLOAT32);
        model->norm_beta[i] = arix_tensor_zeros(shape_dim, 1, ARIX_FLOAT32);
    }

    return model;
}

void arix_hss_model_destroy(ArixHSSModel* model) {
    if (!model) return;
    for (size_t i = 0; i < model->config.num_layers; i++) {
        if (model->layers[i]) arix_hss_layer_destroy(model->layers[i]);
        if (model->norm_gamma[i]) arix_tensor_destroy(model->norm_gamma[i]);
        if (model->norm_beta[i]) arix_tensor_destroy(model->norm_beta[i]);
    }
    arix_free(model->layers, model->config.num_layers * sizeof(ArixHSSLayer*));
    arix_free(model->norm_gamma, model->config.num_layers * sizeof(ArixTensor*));
    arix_free(model->norm_beta, model->config.num_layers * sizeof(ArixTensor*));
    arix_free(model, sizeof(ArixHSSModel));
}
