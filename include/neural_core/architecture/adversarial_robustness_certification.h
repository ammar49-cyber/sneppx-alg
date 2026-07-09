#ifndef SNEPPX_ARC_H
#define SNEPPX_ARC_H

#include "multidimensional_tensor_engine.h"
#include "automatic_differentiation_framework.h"
#include <stddef.h>

typedef enum {
    SNEPPX_OBF_NONE = 0,
    SNEPPX_OBF_NOISE = 1,
    SNEPPX_OBF_CLAMP = 2,
    SNEPPX_OBF_MIXED = 3
} SNEPPXObfuscationMethod;

typedef enum {
    SNEPPX_ATTACK_FGSM = 1,
    SNEPPX_ATTACK_PGD = 2,
    SNEPPX_ATTACK_CW = 4
} SNEPPXAttackType;

typedef struct {
    float input_guard_strength;
    int gradient_obfuscation_method;
    float gradient_noise_scale;
    float gradient_clip_max;
    size_t output_verify_layers;
    float output_verify_threshold;
    int adversarial_training;
    int attack_simulation_types;
} SNEPPXARCConfig;

typedef struct {
    SNEPPXTensor* projection_matrix;
    float anomaly_threshold;
    SNEPPXTensor* norm_stats_mean;
    SNEPPXTensor* norm_stats_var;
} SNEPPXInputGuard;

typedef struct {
    SNEPPXTensor* noise_buffer;
    SNEPPXTensor* clamp_mask;
} SNEPPXGradientObfuscator;

typedef struct {
    SNEPPXTensor** verification_weights;
    SNEPPXTensor** verification_biases;
    size_t num_layers;
    SNEPPXTensor* consistency_history;
    size_t history_idx;
    size_t history_filled;
} SNEPPXOutputVerifier;

typedef struct {
    SNEPPXInputGuard* input_guard;
    SNEPPXGradientObfuscator* gradient_obfuscator;
    SNEPPXOutputVerifier* output_verifier;
    SNEPPXARCConfig config;
    SNEPPXTensor* attack_buffer;
    size_t input_dim;
    size_t output_dim;
} SNEPPXARCLayer;

SNEPPXARCConfig SNEPPX_arc_config_default(void);
SNEPPXInputGuard* SNEPPX_input_guard_create(size_t input_dim, unsigned int seed);
void SNEPPX_input_guard_destroy(SNEPPXInputGuard* guard);
SNEPPXGradientObfuscator* SNEPPX_gradient_obfuscator_create(size_t max_params, unsigned int seed);
void SNEPPX_gradient_obfuscator_destroy(SNEPPXGradientObfuscator* obf);
SNEPPXOutputVerifier* SNEPPX_arc_output_verifier_create(size_t output_dim, size_t num_layers, unsigned int seed);
void SNEPPX_arc_output_verifier_destroy(SNEPPXOutputVerifier* verifier);
SNEPPXARCLayer* SNEPPX_arc_layer_create(const SNEPPXARCConfig* config, size_t input_dim, size_t output_dim, unsigned int seed);
void SNEPPX_arc_layer_destroy(SNEPPXARCLayer* layer);
void SNEPPX_arc_input_guard_forward(SNEPPXInputGuard* guard, const SNEPPXTensor* input, SNEPPXTensor** sanitized, float* anomaly_score);
void SNEPPX_arc_obfuscate_gradients(SNEPPXGradientObfuscator* obf, SNEPPXTensor* gradients, int method);
void SNEPPX_arc_verify_output(SNEPPXOutputVerifier* verifier, const SNEPPXTensor* output, SNEPPXTensor** verified_output, float* confidence);
void SNEPPX_arc_forward(SNEPPXARCLayer* layer, const SNEPPXTensor* input, SNEPPXTensor** output, float* security_metrics);
void SNEPPX_arc_simulate_attack(const SNEPPXTensor* clean_input, int attack_type, float epsilon, SNEPPXTensor** adversarial);
size_t SNEPPX_arc_get_params(const SNEPPXARCLayer* layer, SNEPPXTensor** out_params, size_t max_params);
int SNEPPX_arc_build_train_graph(SNEPPXARCLayer* layer, SNEPPXTape* tape,
                                SNEPPXVariable* input_var,
                                SNEPPXVariable** weight_vars, size_t num_weights,
                                SNEPPXVariable** output_var);

#endif /* SNEPPX_ARC_H */
