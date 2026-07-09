#include "sparse_expert_routing.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <stdlib.h>

SNEPPXSERLayer* SNEPPX_ser_layer_create(const SNEPPXSERConfig* config, unsigned int seed) {
    SNEPPXSERLayer* layer = (SNEPPXSERLayer*)SNEPPX_malloc(sizeof(SNEPPXSERLayer), 64);
    if (!layer) return NULL;
    memset(layer, 0, sizeof(SNEPPXSERLayer));

    layer->config = *config;

    layer->experts = (SNEPPXExpert**)SNEPPX_malloc(config->num_experts * sizeof(SNEPPXExpert*), 64);
    if (!layer->experts) {
        SNEPPX_free(layer, sizeof(SNEPPXSERLayer));
        return NULL;
    }
    memset(layer->experts, 0, config->num_experts * sizeof(SNEPPXExpert*));

    for (size_t i = 0; i < config->num_experts; i++) {
        layer->experts[i] = SNEPPX_expert_create(config, seed + (unsigned int)i * 10, SNEPPX_ACT_RELU);
    }

    size_t shape_ri[] = {config->num_experts, config->input_dim};
    size_t shape_r1[] = {config->num_experts};
    layer->router = SNEPPX_tensor_create(shape_ri, 2, SNEPPX_FLOAT32);
    layer->router_bias = SNEPPX_tensor_zeros(shape_r1, 1, SNEPPX_FLOAT32);

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

void SNEPPX_ser_layer_destroy(SNEPPXSERLayer* layer) {
    if (!layer) return;
    for (size_t i = 0; i < layer->config.num_experts; i++) {
        if (layer->experts[i]) SNEPPX_expert_destroy(layer->experts[i]);
    }
    SNEPPX_free(layer->experts, layer->config.num_experts * sizeof(SNEPPXExpert*));
    if (layer->router) SNEPPX_tensor_destroy(layer->router);
    if (layer->router_bias) SNEPPX_tensor_destroy(layer->router_bias);
    SNEPPX_free(layer, sizeof(SNEPPXSERLayer));
}
