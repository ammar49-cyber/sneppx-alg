#include "hierarchical_state_space.h"
#include <string.h>

void SNEPPX_hss_discretize(SNEPPXHSSLayer* layer) {
    size_t s_dim = layer->A->shape[0];
    size_t i_dim = layer->B->shape[1];

    size_t shape_s[] = {s_dim, s_dim};
    size_t shape_si[] = {s_dim, i_dim};

    if (layer->A_bar) SNEPPX_tensor_destroy(layer->A_bar);
    if (layer->B_bar) SNEPPX_tensor_destroy(layer->B_bar);

    layer->A_bar = SNEPPX_tensor_zeros(shape_s, 2, SNEPPX_FLOAT32);
    layer->B_bar = SNEPPX_tensor_zeros(shape_si, 2, SNEPPX_FLOAT32);
    if (!layer->A_bar || !layer->B_bar) return;

    float* A = (float*)layer->A->data;
    float* B = (float*)layer->B->data;
    float* dt = (float*)layer->dt->data;
    float* A_bar = (float*)layer->A_bar->data;
    float* B_bar = (float*)layer->B_bar->data;

    for (size_t i = 0; i < s_dim; i++) {
        for (size_t j = 0; j < s_dim; j++) {
            float a_ij = A[i * s_dim + j];
            if (i == j) {
                A_bar[i * s_dim + j] = 1.0f + a_ij * dt[i];
            } else {
                A_bar[i * s_dim + j] = a_ij * dt[i];
            }
        }
    }

    for (size_t i = 0; i < s_dim; i++) {
        for (size_t j = 0; j < i_dim; j++) {
            B_bar[i * i_dim + j] = B[i * i_dim + j] * dt[i];
        }
    }
}
