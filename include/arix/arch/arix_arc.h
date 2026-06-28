#ifndef ARIX_ARC_H
#define ARIX_ARC_H

#include "arix_tensor.h"
#include <stddef.h>

typedef enum {
    ARIX_OBF_NONE = 0,
    ARIX_OBF_NOISE = 1,
    ARIX_OBF_CLAMP = 2,
    ARIX_OBF_MIXED = 3
} ArixObfuscationMethod;

typedef enum {
    ARIX_ATTACK_FGSM = 1,
    ARIX_ATTACK_PGD = 2,
    ARIX_ATTACK_CW = 4
} ArixAttackType;

typedef struct {
    float input_guard_strength;
    int gradient_obfuscation_method;
    float gradient_noise_scale;
    float gradient_clip_max;
    size_t output_verify_layers;
    float output_verify_threshold;
    int adversarial_training;
    int attack_simulation_types;
} ArixARCConfig;

typedef struct {
    ArixTensor* projection_matrix;
    float anomaly_threshold;
    ArixTensor* norm_stats_mean;
    ArixTensor* norm_stats_var;
} ArixInputGuard;

typedef struct {
    ArixTensor* noise_buffer;
    ArixTensor* clamp_mask;
} ArixGradientObfuscator;

typedef struct {
    ArixTensor** verification_weights;
    ArixTensor** verification_biases;
    size_t num_layers;
    ArixTensor* consistency_history;
    size_t history_idx;
    size_t history_filled;
} ArixOutputVerifier;

typedef struct {
    ArixInputGuard* input_guard;
    ArixGradientObfuscator* gradient_obfuscator;
    ArixOutputVerifier* output_verifier;
    ArixARCConfig config;
    ArixTensor* attack_buffer;
    size_t input_dim;
    size_t output_dim;
} ArixARCLayer;

ArixARCConfig arix_arc_config_default(void);
ArixInputGuard* arix_input_guard_create(size_t input_dim, unsigned int seed);
void arix_input_guard_destroy(ArixInputGuard* guard);
ArixGradientObfuscator* arix_gradient_obfuscator_create(size_t max_params, unsigned int seed);
void arix_gradient_obfuscator_destroy(ArixGradientObfuscator* obf);
ArixOutputVerifier* arix_output_verifier_create(size_t output_dim, size_t num_layers, unsigned int seed);
void arix_output_verifier_destroy(ArixOutputVerifier* verifier);
ArixARCLayer* arix_arc_layer_create(const ArixARCConfig* config, size_t input_dim, size_t output_dim, unsigned int seed);
void arix_arc_layer_destroy(ArixARCLayer* layer);
void arix_arc_input_guard_forward(ArixInputGuard* guard, const ArixTensor* input, ArixTensor** sanitized, float* anomaly_score);
void arix_arc_obfuscate_gradients(ArixGradientObfuscator* obf, ArixTensor* gradients, int method);
void arix_arc_verify_output(ArixOutputVerifier* verifier, const ArixTensor* output, ArixTensor** verified_output, float* confidence);
void arix_arc_forward(ArixARCLayer* layer, const ArixTensor* input, ArixTensor** output, float* security_metrics);
void arix_arc_simulate_attack(const ArixTensor* clean_input, int attack_type, float epsilon, ArixTensor** adversarial);

#endif /* ARIX_ARC_H */
