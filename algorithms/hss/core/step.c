#include "hierarchical_state_space.h"

void arix_hss_step(const ArixHSSLayer* layer, const ArixTensor* x, ArixTensor* h_next) {
    size_t s_dim = layer->A_bar->shape[0];
    size_t i_dim = x->size;

    float* A_bar = (float*)layer->A_bar->data;
    float* B_bar = (float*)layer->B_bar->data;
    float* h = (float*)layer->h->data;
    float* x_data = (float*)x->data;
    float* h_next_data = (float*)h_next->data;

    for (size_t i = 0; i < s_dim; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < s_dim; j++) {
            sum += A_bar[i * s_dim + j] * h[j];
        }
        for (size_t k = 0; k < i_dim; k++) {
            sum += B_bar[i * i_dim + k] * x_data[k];
        }
        h_next_data[i] = sum;
    }
}
