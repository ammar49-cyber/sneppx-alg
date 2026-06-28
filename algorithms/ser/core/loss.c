#include "arix_ser.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

float arix_ser_load_balance_loss(const ArixTensor* gate_weights, const int* expert_indices, size_t num_tokens) {
    size_t n_act = gate_weights->shape[1];
    int n_exp = 0;
    for (size_t i = 0; i < num_tokens * n_act; i++) {
        if (expert_indices[i] > n_exp) n_exp = expert_indices[i];
    }
    n_exp++;

    float* frac = (float*)calloc((size_t)n_exp, sizeof(float));
    float* p = (float*)calloc((size_t)n_exp, sizeof(float));
    if (!frac || !p) { free(frac); free(p); return 0.0f; }

    float* gw = (float*)gate_weights->data;
    for (size_t t = 0; t < num_tokens; t++) {
        for (size_t k = 0; k < n_act; k++) {
            int e = expert_indices[t * n_act + k];
            if (e >= 0 && e < n_exp) {
                frac[e] += 1.0f;
                p[e] += gw[t * n_act + k];
            }
        }
    }

    float loss = 0.0f;
    for (int e = 0; e < n_exp; e++) {
        float f_e = frac[e] / (float)num_tokens;
        float p_e = p[e] / (float)num_tokens;
        loss += f_e * p_e;
    }
    loss *= (float)n_exp;

    free(frac);
    free(p);
    return loss;
}
