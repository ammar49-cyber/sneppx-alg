#include "arix_arc.h"
#include <string.h>
#include <math.h>

void arix_arc_forward(ArixARCLayer* layer, const ArixTensor* input, ArixTensor** output, float* security_metrics) {
    ArixTensor* sanitized = NULL;
    float anomaly_score = 0.0f;
    arix_arc_input_guard_forward(layer->input_guard, input, &sanitized, &anomaly_score);

    float confidence = 0.0f;
    ArixTensor* verified = NULL;

    if (sanitized) {
        arix_arc_verify_output(layer->output_verifier, sanitized, &verified, &confidence);
        *output = verified;
        arix_tensor_destroy(sanitized);
    } else {
        size_t shape_s[] = {input->shape[0], input->shape[1]};
        *output = arix_tensor_create(shape_s, 2, ARIX_FLOAT32);
        if (*output) {
            memcpy((float*)(*output)->data, (float*)input->data, input->size * sizeof(float));
        }
        confidence = 0.0f;
    }

    float clamp_ratio = 0.0f;
    if (layer->gradient_obfuscator && layer->gradient_obfuscator->clamp_mask) {
        float* cm = (float*)layer->gradient_obfuscator->clamp_mask->data;
        size_t n = layer->gradient_obfuscator->clamp_mask->size;
        size_t clamped = 0;
        for (size_t i = 0; i < n; i++) { if (cm[i] > 0.5f) clamped++; }
        clamp_ratio = n > 0 ? (float)clamped / (float)n : 0.0f;
    }

    if (security_metrics) {
        security_metrics[0] = anomaly_score;
        security_metrics[1] = confidence;
        security_metrics[2] = clamp_ratio;
        float output_norm = 0.0f;
        if (*output) {
            float* od = (float*)(*output)->data;
            for (size_t i = 0; i < (*output)->size; i++)
                output_norm += od[i] * od[i];
            output_norm = sqrtf(output_norm);
        }
        security_metrics[3] = output_norm;
    }
}
