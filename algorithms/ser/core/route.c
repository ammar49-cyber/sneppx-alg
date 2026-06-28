#include "arix_ser.h"
#include "arix_memory.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

static unsigned long lcg_next(unsigned long state) {
    return state * 1103515245UL + 12345UL;
}

static void softmax(float* data, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; r++) {
        float max_val = data[r * cols];
        for (size_t c = 1; c < cols; c++) {
            if (data[r * cols + c] > max_val) max_val = data[r * cols + c];
        }
        float sum = 0.0f;
        for (size_t c = 0; c < cols; c++) {
            data[r * cols + c] = expf(data[r * cols + c] - max_val);
            sum += data[r * cols + c];
        }
        for (size_t c = 0; c < cols; c++) {
            data[r * cols + c] /= sum;
        }
    }
}

static int cmp_desc(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) ? -1 : (fa < fb) ? 1 : 0;
}

void arix_ser_route(ArixSERLayer* layer, const ArixTensor* input, ArixTensor** gate_weights, int** expert_indices) {
    size_t num_tokens = input->shape[0];
    size_t i_dim = input->shape[1];
    size_t n_exp = layer->config.num_experts;
    size_t n_act = layer->config.num_active;

    float* router_data = (float*)layer->router->data;
    float* bias_data = (float*)layer->router_bias->data;
    float* in_data = (float*)input->data;

    float* logits = (float*)malloc(n_exp * num_tokens * sizeof(float));
    if (!logits) return;

    for (size_t e = 0; e < n_exp; e++) {
        for (size_t t = 0; t < num_tokens; t++) {
            float sum = bias_data[e];
            for (size_t i = 0; i < i_dim; i++) {
                sum += router_data[e * i_dim + i] * in_data[t * i_dim + i];
            }
            logits[e * num_tokens + t] = sum;
        }
    }

    if (layer->config.top_k_method == ARIX_TOPK_NOISY) {
        unsigned long state = 12345;
        for (size_t i = 0; i < n_exp * num_tokens; i++) {
            state = lcg_next(state);
            float noise = ((state >> 16) & 0x7FFF) / 32767.0f * 0.1f;
            logits[i] += noise;
        }
    }

    for (size_t t = 0; t < num_tokens; t++) {
        float max_val = logits[t];
        for (size_t e = 1; e < n_exp; e++) {
            if (logits[e * num_tokens + t] > max_val)
                max_val = logits[e * num_tokens + t];
        }
        float sum = 0.0f;
        for (size_t e = 0; e < n_exp; e++) {
            logits[e * num_tokens + t] = expf(logits[e * num_tokens + t] - max_val);
            sum += logits[e * num_tokens + t];
        }
        for (size_t e = 0; e < n_exp; e++) {
            logits[e * num_tokens + t] /= sum;
        }
    }

    size_t shape_gw[] = {num_tokens, n_act};
    *gate_weights = arix_tensor_create(shape_gw, 2, ARIX_FLOAT32);
    *expert_indices = (int*)malloc(num_tokens * n_act * sizeof(int));
    if (!*gate_weights || !*expert_indices) {
        free(logits);
        return;
    }

    for (size_t t = 0; t < num_tokens; t++) {
        float scores[256];
        int idxs[256];
        for (size_t e = 0; e < n_exp; e++) {
            scores[e] = logits[e * num_tokens + t];
            idxs[e] = (int)e;
        }
        for (size_t i = 0; i < n_exp - 1; i++) {
            for (size_t j = 0; j < n_exp - 1 - i; j++) {
                if (scores[j] < scores[j + 1]) {
                    float tmp_s = scores[j];
                    scores[j] = scores[j + 1];
                    scores[j + 1] = tmp_s;
                    int tmp_i = idxs[j];
                    idxs[j] = idxs[j + 1];
                    idxs[j + 1] = tmp_i;
                }
            }
        }

        float* gw_data = (float*)(*gate_weights)->data;
        int* ei_data = *expert_indices;
        float sum_gw = 0.0f;
        for (size_t k = 0; k < n_act; k++) {
            gw_data[t * n_act + k] = scores[k];
            ei_data[t * n_act + k] = idxs[k];
            sum_gw += scores[k];
        }
        if (sum_gw > 0.0f) {
            for (size_t k = 0; k < n_act; k++) {
                gw_data[t * n_act + k] /= sum_gw;
            }
        }
    }

    free(logits);
}
