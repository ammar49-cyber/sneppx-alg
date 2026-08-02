#ifndef SNEPPX_GENERATIVE_ADVERSARIAL_NETWORK_H
#define SNEPPX_GENERATIVE_ADVERSARIAL_NETWORK_H
#include <stddef.h>
#include <stdbool.h>
/*
 * SNEPPX - Generative Adversarial Network
 *
 * WHAT
 *   Generative Adversarial Network.
 *
 * CONCEPT
 *   GAN configuration and training API for the SNEPPX-Algo system.
 *
 * ROLE
 *   Declares the GAN API for generative adversarial training within the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal module).
 */


#ifdef __cplusplus
extern "C" {
#endif

typedef struct SNEPPXGAN SNEPPXGAN;

SNEPPXGAN* SNEPPX_gan_create(size_t latent_dim, size_t hidden_dim, size_t output_dim,
    int use_batch_norm, int use_spectral_norm);
void SNEPPX_gan_destroy(void* gan);
int SNEPPX_gan_generate(void* gan, const float* noise, size_t num_samples, float* output);
int SNEPPX_gan_train_step(void* gan, const float* real_samples, size_t num_samples,
    float* g_loss, float* d_loss);

#ifdef __cplusplus
}
#endif
#endif
