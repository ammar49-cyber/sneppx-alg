#include "sparse_expert_routing.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>

ArixSERModel* arix_ser_model_create(const ArixSERConfig* config, unsigned int seed, size_t num_layers) {
    ArixSERModel* model = (ArixSERModel*)arix_malloc(sizeof(ArixSERModel), 64);
    if (!model) return NULL;
    memset(model, 0, sizeof(ArixSERModel));

    model->config = *config;
    model->num_layers = num_layers;

    model->layers = (ArixSERLayer**)arix_malloc(num_layers * sizeof(ArixSERLayer*), 64);
    if (!model->layers) {
        arix_free(model, sizeof(ArixSERModel));
        return NULL;
    }
    memset(model->layers, 0, num_layers * sizeof(ArixSERLayer*));

    for (size_t i = 0; i < num_layers; i++) {
        model->layers[i] = arix_ser_layer_create(config, seed + (unsigned int)i * 100);
    }

    return model;
}

void arix_ser_model_destroy(ArixSERModel* model) {
    if (!model) return;
    for (size_t i = 0; i < model->num_layers; i++) {
        if (model->layers[i]) arix_ser_layer_destroy(model->layers[i]);
    }
    arix_free(model->layers, model->num_layers * sizeof(ArixSERLayer*));
    arix_free(model, sizeof(ArixSERModel));
}

size_t arix_ser_get_params(const ArixSERModel* model, ArixTensor** out, size_t max_out) {
    if (!model) return 0;
    size_t count = 0;
    for (size_t l = 0; l < model->num_layers; l++) {
        ArixSERLayer* layer = model->layers[l];
        if (out && count < max_out) out[count] = layer->router;
        count++;
        if (out && count < max_out) out[count] = layer->router_bias;
        count++;
        for (size_t e = 0; e < model->config.num_experts; e++) {
            ArixExpert* exp = layer->experts[e];
            if (out && count < max_out) out[count] = exp->w1; count++;
            if (out && count < max_out) out[count] = exp->b1; count++;
            if (out && count < max_out) out[count] = exp->w2; count++;
            if (out && count < max_out) out[count] = exp->b2; count++;
        }
    }
    return count;
}
