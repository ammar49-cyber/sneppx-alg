#include "arix_arc.h"
#include "arix_memory.h"
#include <string.h>

ArixARCLayer* arix_arc_layer_create(const ArixARCConfig* config, size_t input_dim, size_t output_dim, unsigned int seed) {
    ArixARCLayer* layer = (ArixARCLayer*)arix_malloc(sizeof(ArixARCLayer), 64);
    if (!layer) return NULL;
    memset(layer, 0, sizeof(ArixARCLayer));

    layer->config = *config;
    layer->input_dim = input_dim;
    layer->output_dim = output_dim;

    layer->input_guard = arix_input_guard_create(input_dim, seed);
    layer->gradient_obfuscator = arix_gradient_obfuscator_create(input_dim * output_dim, seed + 100);
    layer->output_verifier = arix_output_verifier_create(output_dim, config->output_verify_layers, seed + 200);

    size_t shape_ab[] = {input_dim};
    layer->attack_buffer = arix_tensor_zeros(shape_ab, 1, ARIX_FLOAT32);

    return layer;
}

void arix_arc_layer_destroy(ArixARCLayer* layer) {
    if (!layer) return;
    if (layer->input_guard) arix_input_guard_destroy(layer->input_guard);
    if (layer->gradient_obfuscator) arix_gradient_obfuscator_destroy(layer->gradient_obfuscator);
    if (layer->output_verifier) arix_output_verifier_destroy(layer->output_verifier);
    if (layer->attack_buffer) arix_tensor_destroy(layer->attack_buffer);
    arix_free(layer, sizeof(ArixARCLayer));
}
