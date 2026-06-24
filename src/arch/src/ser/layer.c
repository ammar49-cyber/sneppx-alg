#include "arix_ser.h"
#include "arix_memory.h"
#include <string.h>
#include <stdlib.h>

ArixSERLayer* arix_ser_layer_create(const ArixSERConfig* config, unsigned int seed) {
    ArixSERLayer* layer = (ArixSERLayer*)arix_malloc(sizeof(ArixSERLayer), 64);
    if (!layer) return NULL;
    memset(layer, 0, sizeof(ArixSERLayer));

    layer->config = *config;

    layer->experts = (ArixExpert**)arix_malloc(config->num_experts * sizeof(ArixExpert*), 64);
    if (!layer->experts) {
        arix_free(layer, sizeof(ArixSERLayer));
        return NULL;
    }
    memset(layer->experts, 0, config->num_experts * sizeof(ArixExpert*));

    for (size_t i = 0; i < config->num_experts; i++) {
        layer->experts[i] = arix_expert_create(config, seed + (unsigned int)i * 10, ARIX_ACT_RELU);
    }

    size_t shape_ri[] = {config->num_experts, config->input_dim};
    size_t shape_r1[] = {config->num_experts};
    layer->router = arix_tensor_create(shape_ri, 2, ARIX_FLOAT32);
    layer->router_bias = arix_tensor_zeros(shape_r1, 1, ARIX_FLOAT32);

    if (layer->router && layer->router->data) {
        unsigned long state = seed + 9999;
        float* rdata = (float*)layer->router->data;
        for (size_t i = 0; i < layer->router->size; i += 2) {
            state = state * 1103515245UL + 12345UL;
            float u1 = (float)((state >> 16) & 0x7FFF) / 32767.0f;
            state = state * 1103515245UL + 12345UL;
            float u2 = (float)((state >> 16) & 0x7FFF) / 32767.0f;
            float r = sqrtf(-2.0f * logf(u1 + 1e-10f));
            float theta = 2.0f * 3.14159265f * u2;
            rdata[i] = r * cosf(theta) * 0.01f;
            if (i + 1 < layer->router->size) {
                rdata[i + 1] = r * sinf(theta) * 0.01f;
            }
        }
    }

    layer->expert_capacity = (config->input_dim * 2) / config->num_experts;
    if (layer->expert_capacity < 1) layer->expert_capacity = 1;

    return layer;
}

void arix_ser_layer_destroy(ArixSERLayer* layer) {
    if (!layer) return;
    for (size_t i = 0; i < layer->config.num_experts; i++) {
        if (layer->experts[i]) arix_expert_destroy(layer->experts[i]);
    }
    arix_free(layer->experts, layer->config.num_experts * sizeof(ArixExpert*));
    if (layer->router) arix_tensor_destroy(layer->router);
    if (layer->router_bias) arix_tensor_destroy(layer->router_bias);
    arix_free(layer, sizeof(ArixSERLayer));
}
