#include "arix_arc.h"
#include "arix_memory.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

ArixOutputVerifier* arix_output_verifier_create(size_t output_dim, size_t num_layers, unsigned int seed) {
    ArixOutputVerifier* verifier = (ArixOutputVerifier*)arix_malloc(sizeof(ArixOutputVerifier), 64);
    if (!verifier) return NULL;
    memset(verifier, 0, sizeof(ArixOutputVerifier));

    verifier->num_layers = num_layers;
    verifier->history_idx = 0;
    verifier->history_filled = 0;

    verifier->verification_weights = (ArixTensor**)arix_malloc(num_layers * sizeof(ArixTensor*), 64);
    verifier->verification_biases = (ArixTensor**)arix_malloc(num_layers * sizeof(ArixTensor*), 64);
    if (!verifier->verification_weights || !verifier->verification_biases) {
        arix_free(verifier->verification_weights, num_layers * sizeof(ArixTensor*));
        arix_free(verifier->verification_biases, num_layers * sizeof(ArixTensor*));
        arix_free(verifier, sizeof(ArixOutputVerifier));
        return NULL;
    }
    memset(verifier->verification_weights, 0, num_layers * sizeof(ArixTensor*));
    memset(verifier->verification_biases, 0, num_layers * sizeof(ArixTensor*));

    size_t shape_oo[] = {output_dim, output_dim};
    size_t shape_o1[] = {output_dim};

    unsigned long state = seed;
    for (size_t l = 0; l < num_layers; l++) {
        verifier->verification_weights[l] = arix_tensor_create(shape_oo, 2, ARIX_FLOAT32);
        verifier->verification_biases[l] = arix_tensor_zeros(shape_o1, 1, ARIX_FLOAT32);
        if (verifier->verification_weights[l]) {
            float* wd = (float*)verifier->verification_weights[l]->data;
            for (size_t i = 0; i < output_dim * output_dim; i += 2) {
                state = state * 1103515245UL + 12345UL;
                float u1 = (float)((state >> 16) & 0x7FFF) / 32767.0f;
                state = state * 1103515245UL + 12345UL;
                float u2 = (float)((state >> 16) & 0x7FFF) / 32767.0f;
                float r = sqrtf(-2.0f * logf(u1 + 1e-10f));
                float theta = 2.0f * 3.14159265f * u2;
                wd[i] = r * cosf(theta) * 0.1f;
                if (i + 1 < output_dim * output_dim)
                    wd[i + 1] = r * sinf(theta) * 0.1f;
            }
        }
    }

    size_t shape_h[] = {10, output_dim};
    verifier->consistency_history = arix_tensor_zeros(shape_h, 2, ARIX_FLOAT32);

    return verifier;
}

void arix_output_verifier_destroy(ArixOutputVerifier* verifier) {
    if (!verifier) return;
    for (size_t i = 0; i < verifier->num_layers; i++) {
        if (verifier->verification_weights[i]) arix_tensor_destroy(verifier->verification_weights[i]);
        if (verifier->verification_biases[i]) arix_tensor_destroy(verifier->verification_biases[i]);
    }
    arix_free(verifier->verification_weights, verifier->num_layers * sizeof(ArixTensor*));
    arix_free(verifier->verification_biases, verifier->num_layers * sizeof(ArixTensor*));
    if (verifier->consistency_history) arix_tensor_destroy(verifier->consistency_history);
    arix_free(verifier, sizeof(ArixOutputVerifier));
}

void arix_arc_verify_output(ArixOutputVerifier* verifier, const ArixTensor* output, ArixTensor** verified_output, float* confidence) {
    size_t batch = output->shape[0];
    size_t dim = output->shape[1];

    size_t shape_v[] = {batch, dim};
    *verified_output = arix_tensor_create(shape_v, 2, ARIX_FLOAT32);
    if (!*verified_output) { *confidence = 0.0f; return; }
    float* out_data = (float*)output->data;
    float* ver_data = (float*)(*verified_output)->data;

    memcpy(ver_data, out_data, batch * dim * sizeof(float));

    float* work = (float*)malloc(dim * sizeof(float));
    if (!work) { *confidence = 0.0f; return; }

    for (size_t l = 0; l < verifier->num_layers; l++) {
        float* w = (float*)verifier->verification_weights[l]->data;
        float* b = (float*)verifier->verification_biases[l]->data;
        for (size_t t = 0; t < batch; t++) {
            float* vt = ver_data + t * dim;
            for (size_t i = 0; i < dim; i++) {
                float sum = b[i];
                for (size_t j = 0; j < dim; j++)
                    sum += w[i * dim + j] * vt[j];
                work[i] = sum > 0.0f ? sum : 0.0f;
            }
            memcpy(vt, work, dim * sizeof(float));
        }
    }
    free(work);

    float* hist = (float*)verifier->consistency_history->data;
    size_t hist_cap = verifier->consistency_history->shape[0];

    float* avg_hist = (float*)calloc(dim, sizeof(float));
    if (avg_hist && verifier->history_filled > 0) {
        for (size_t i = 0; i < verifier->history_filled; i++)
            for (size_t j = 0; j < dim; j++)
                avg_hist[j] += hist[i * dim + j];
        for (size_t j = 0; j < dim; j++)
            avg_hist[j] /= (float)verifier->history_filled;
    }

    if (avg_hist && verifier->history_filled > 0) {
        float dot = 0.0f, n1 = 0.0f, n2 = 0.0f;
        for (size_t j = 0; j < dim; j++) {
            dot += out_data[j] * avg_hist[j];
            n1 += out_data[j] * out_data[j];
            n2 += avg_hist[j] * avg_hist[j];
        }
        float cos_sim = dot / (sqrtf(n1 + 1e-10f) * sqrtf(n2 + 1e-10f));
        *confidence = (cos_sim + 1.0f) * 0.5f;

        if (*confidence < 0.9f) {
            float alpha = 0.3f;
            for (size_t t = 0; t < batch; t++) {
                float* vt = ver_data + t * dim;
                for (size_t j = 0; j < dim; j++)
                    vt[j] = alpha * out_data[t * dim + j] + (1.0f - alpha) * avg_hist[j];
            }
        }
    } else {
        *confidence = 1.0f;
    }

    free(avg_hist);

    if (verifier->history_filled < hist_cap) verifier->history_filled++;
    memcpy(hist + (verifier->history_idx * dim), out_data, dim * sizeof(float));
    verifier->history_idx = (verifier->history_idx + 1) % hist_cap;
}
