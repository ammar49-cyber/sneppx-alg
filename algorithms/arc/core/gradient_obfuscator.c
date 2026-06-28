#include "arix_arc.h"
#include "arix_memory.h"
#include <string.h>
#include <math.h>

ArixGradientObfuscator* arix_gradient_obfuscator_create(size_t max_params, unsigned int seed) {
    (void)seed;
    ArixGradientObfuscator* obf = (ArixGradientObfuscator*)arix_malloc(sizeof(ArixGradientObfuscator), 64);
    if (!obf) return NULL;
    memset(obf, 0, sizeof(ArixGradientObfuscator));

    size_t shape_m[] = {max_params};
    obf->noise_buffer = arix_tensor_zeros(shape_m, 1, ARIX_FLOAT32);
    obf->clamp_mask = arix_tensor_zeros(shape_m, 1, ARIX_FLOAT32);
    return obf;
}

void arix_gradient_obfuscator_destroy(ArixGradientObfuscator* obf) {
    if (!obf) return;
    if (obf->noise_buffer) arix_tensor_destroy(obf->noise_buffer);
    if (obf->clamp_mask) arix_tensor_destroy(obf->clamp_mask);
    arix_free(obf, sizeof(ArixGradientObfuscator));
}

void arix_arc_obfuscate_gradients(ArixGradientObfuscator* obf, ArixTensor* gradients, int method) {
    if (!obf || !gradients) return;
    float* gd = (float*)gradients->data;
    size_t n = gradients->size;

    if (method & ARIX_OBF_NOISE) {
        float* nb = (float*)obf->noise_buffer->data;
        size_t max_n = obf->noise_buffer->size > n ? n : obf->noise_buffer->size;
        float scale = 0.01f;
        unsigned long state = 12345;
        for (size_t i = 0; i < max_n; i += 2) {
            state = state * 1103515245UL + 12345UL;
            float u1 = (float)((state >> 16) & 0x7FFF) / 32767.0f;
            state = state * 1103515245UL + 12345UL;
            float u2 = (float)((state >> 16) & 0x7FFF) / 32767.0f;
            float r = sqrtf(-2.0f * logf(u1 + 1e-10f));
            float theta = 2.0f * 3.14159265f * u2;
            nb[i] = r * cosf(theta) * scale;
            if (i + 1 < max_n)
                nb[i + 1] = r * sinf(theta) * scale;
        }
        for (size_t i = 0; i < n && i < obf->noise_buffer->size; i++) {
            gd[i] += nb[i];
        }
    }

    if (method & ARIX_OBF_CLAMP) {
        float* cm = (float*)obf->clamp_mask->data;
        float clip_max = 1.0f;
        for (size_t i = 0; i < n && i < obf->clamp_mask->size; i++) {
            if (gd[i] > clip_max) { gd[i] = clip_max; cm[i] = 1.0f; }
            else if (gd[i] < -clip_max) { gd[i] = -clip_max; cm[i] = 1.0f; }
            else { cm[i] = 0.0f; }
        }
    }
}
