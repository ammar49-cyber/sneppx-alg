#include "hierarchical_state_space.h"
#include "polymorphic_memory_allocator.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

ArixHSSConfig arix_hss_config_default(void) {
    ArixHSSConfig cfg;
    cfg.state_dim = 64;
    cfg.input_dim = 512;
    cfg.output_dim = 512;
    cfg.num_layers = 2;
    cfg.seq_len = 1024;
    cfg.dt_min = 0.001f;
    cfg.dt_max = 0.1f;
    cfg.use_hierarchical = 1;
    return cfg;
}

static unsigned long lcg_next(unsigned long state) {
    return state * 1103515245UL + 12345UL;
}

static float uniform_float(unsigned long* state, float min, float max) {
    *state = lcg_next(*state);
    float t = (float)((*state >> 16) & 0x7FFF) / 32767.0f;
    return min + t * (max - min);
}

static void fill_randn(ArixTensor* t, unsigned long* state, float scale) {
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->size; i += 2) {
        float u1 = uniform_float(state, 0.0f, 1.0f);
        float u2 = uniform_float(state, 0.0f, 1.0f);
        float r = sqrtf(-2.0f * logf(u1 + 1e-10f));
        float theta = 2.0f * 3.14159265f * u2;
        data[i] = r * cosf(theta) * scale;
        if (i + 1 < t->size) {
            data[i + 1] = r * sinf(theta) * scale;
        }
    }
}

ArixHSSLayer* arix_hss_layer_create(const ArixHSSConfig* config, unsigned int seed) {
    ArixHSSLayer* layer = (ArixHSSLayer*)arix_malloc(sizeof(ArixHSSLayer), 64);
    if (!layer) return NULL;
    memset(layer, 0, sizeof(ArixHSSLayer));

    unsigned long state = seed;
    size_t s_dim = config->state_dim;
    size_t i_dim = config->input_dim;
    size_t o_dim = config->output_dim;
    size_t shape_s[] = {s_dim, s_dim};
    size_t shape_si[] = {s_dim, i_dim};
    size_t shape_os[] = {o_dim, s_dim};
    size_t shape_oi[] = {o_dim, i_dim};
    size_t shape_s1[] = {s_dim};
    size_t shape_ii[] = {i_dim, i_dim};
    size_t shape_i1[] = {i_dim};

    layer->A = arix_tensor_create(shape_s, 2, ARIX_FLOAT32);
    layer->B = arix_tensor_create(shape_si, 2, ARIX_FLOAT32);
    layer->C = arix_tensor_create(shape_os, 2, ARIX_FLOAT32);
    layer->D = arix_tensor_create(shape_oi, 2, ARIX_FLOAT32);
    layer->dt = arix_tensor_create(shape_s1, 1, ARIX_FLOAT32);
    layer->h = arix_tensor_create(shape_s1, 1, ARIX_FLOAT32);
    layer->x_proj = arix_tensor_create(shape_ii, 2, ARIX_FLOAT32);
    layer->x_proj_bias = arix_tensor_create(shape_i1, 1, ARIX_FLOAT32);

    if (layer->A && layer->B && layer->C && layer->D && layer->dt &&
        layer->h && layer->x_proj && layer->x_proj_bias) {
        fill_randn(layer->A, &state, 0.01f);
        fill_randn(layer->B, &state, 0.1f);
        fill_randn(layer->C, &state, 0.1f);

        float* dt_data = (float*)layer->dt->data;
        for (size_t i = 0; i < s_dim; i++) {
            dt_data[i] = uniform_float(&state, config->dt_min, config->dt_max);
        }

        fill_randn(layer->x_proj, &state, 0.02f);
    }

    layer->A_bar = NULL;
    layer->B_bar = NULL;

    return layer;
}

void arix_hss_layer_destroy(ArixHSSLayer* layer) {
    if (!layer) return;
    if (layer->A) arix_tensor_destroy(layer->A);
    if (layer->B) arix_tensor_destroy(layer->B);
    if (layer->C) arix_tensor_destroy(layer->C);
    if (layer->D) arix_tensor_destroy(layer->D);
    if (layer->dt) arix_tensor_destroy(layer->dt);
    if (layer->h) arix_tensor_destroy(layer->h);
    if (layer->x_proj) arix_tensor_destroy(layer->x_proj);
    if (layer->x_proj_bias) arix_tensor_destroy(layer->x_proj_bias);
    if (layer->A_bar) arix_tensor_destroy(layer->A_bar);
    if (layer->B_bar) arix_tensor_destroy(layer->B_bar);
    arix_free(layer, sizeof(ArixHSSLayer));
}
