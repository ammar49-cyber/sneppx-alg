#ifndef SNEPPX_DIFFUSION_MODEL_H
#define SNEPPX_DIFFUSION_MODEL_H
#include <stddef.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct SNEPPXDiffusion SNEPPXDiffusion;

SNEPPXDiffusion* SNEPPX_diffusion_create(size_t img_channels, size_t img_size,
    size_t hidden_dim, int num_timesteps, int noise_schedule_type);
void SNEPPX_diffusion_destroy(void* model);
int SNEPPX_diffusion_sample(void* model, float* output, size_t num_samples, const float* cond);
int SNEPPX_diffusion_train_step(void* model, const float* images, size_t num_samples, float* loss);

#ifdef __cplusplus
}
#endif
#endif
