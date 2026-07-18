#ifndef SNEPPX_VISION_TRANSFORMER_H
#define SNEPPX_VISION_TRANSFORMER_H
#include <stddef.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct SNEPPXVisionTransformer SNEPPXVisionTransformer;

SNEPPXVisionTransformer* SNEPPX_vit_create(size_t img_size, size_t patch_size,
    size_t in_channels, size_t num_classes, size_t hidden_dim, size_t num_heads,
    size_t num_layers, float dropout);
void SNEPPX_vit_destroy(void* vit);
int SNEPPX_vit_forward(void* vit, const float* images, size_t batch_size, float* logits);
int SNEPPX_vit_extract_features(void* vit, const float* images, size_t batch_size, float* features);

#ifdef __cplusplus
}
#endif
#endif
