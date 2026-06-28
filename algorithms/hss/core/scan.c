#include "arix_hss.h"
#include <string.h>

void arix_hss_scan(const ArixHSSLayer* layer, const ArixTensor* x_seq, ArixTensor* h_seq, ArixTensor* y_seq) {
    size_t seq_len = x_seq->shape[0];
    size_t i_dim = x_seq->shape[1];
    size_t s_dim = layer->A_bar->shape[0];
    size_t o_dim = layer->C->shape[0];

    float* A_bar = (float*)layer->A_bar->data;
    float* B_bar = (float*)layer->B_bar->data;
    float* C = (float*)layer->C->data;
    float* D = (float*)layer->D->data;
    float* x_data = (float*)x_seq->data;
    float* h_data = (float*)layer->h->data;
    float* h_seq_data = (float*)h_seq->data;
    float* y_seq_data = (float*)y_seq->data;

    memset(h_data, 0, s_dim * sizeof(float));

    for (size_t t = 0; t < seq_len; t++) {
        float* x_t = &x_data[t * i_dim];

        float h_next[4096];
        for (size_t i = 0; i < s_dim; i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < s_dim; j++) {
                sum += A_bar[i * s_dim + j] * h_data[j];
            }
            for (size_t k = 0; k < i_dim; k++) {
                sum += B_bar[i * i_dim + k] * x_t[k];
            }
            h_next[i] = sum;
        }

        memcpy(h_data, h_next, s_dim * sizeof(float));

        for (size_t i = 0; i < s_dim; i++) {
            h_seq_data[t * s_dim + i] = h_data[i];
        }

        for (size_t i = 0; i < o_dim; i++) {
            float y = 0.0f;
            for (size_t j = 0; j < s_dim; j++) {
                y += C[i * s_dim + j] * h_data[j];
            }
            for (size_t k = 0; k < i_dim; k++) {
                y += D[i * i_dim + k] * x_t[k];
            }
            y_seq_data[t * o_dim + i] = y;
        }
    }
}
