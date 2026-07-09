#ifndef SNEPPX_S5_EXTENSIONS_H
#define SNEPPX_S5_EXTENSIONS_H
/* S5 AI Sanitizer extensions: semantic injection (NLP), multi-lang jailbreak,
   encoded attack decoder, token anomaly scoring, model inversion defense,
   membership inference, data extraction prevention, training data sanitization,
   model watermarking, adversarial perturbation, factuality scorer, bias measurement,
   prompt policy engine */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_S5_MAX_EMBEDDING 256
#define SNEPPX_S5_MAX_ENCODED_PATTERNS 64

/* Semantic injection detection via embedding similarity */
typedef struct {
    double known_attack_embeddings[SNEPPX_S5_MAX_EMBEDDING][8];
    int attack_count;
    double threshold;
} SNEPPXSemanticInjectionDetector;

int  SNEPPX_semantic_injection_init(SNEPPXSemanticInjectionDetector* sid);
int  SNEPPX_semantic_injection_add_attack(SNEPPXSemanticInjectionDetector* sid, const double embedding[8]);
int  SNEPPX_semantic_injection_score(SNEPPXSemanticInjectionDetector* sid, const double embedding[8], double* score);

/* Multi-language jailbreak detection */
int  SNEPPX_ml_jailbreak_detect(const char* text, size_t len);

/* Encoded attack decoder (base64/hex/rot13) */
int  SNEPPX_encoded_attack_decode(const char* input, size_t in_len, char* output, size_t* out_len);
int  SNEPPX_encoded_attack_scan(const char* text, size_t len);

/* Token-level anomaly scoring */
double SNEPPX_token_anomaly_score(const uint32_t* token_ids, size_t token_count, const double* expected_probs);

/* Model inversion defense */
typedef struct {
    double noise_scale;
    int gradient_clipping;
    double clip_norm;
} SNEPPXModelInversionDefense;

int  SNEPPX_model_inversion_init(SNEPPXModelInversionDefense* mid);
int  SNEPPX_model_inversion_apply(SNEPPXModelInversionDefense* mid, double* gradients, size_t grad_count);

/* Membership inference defense */
int  SNEPPX_membership_inference_defense(double* logits, size_t logit_count, double epsilon);

/* Data extraction prevention */
int  SNEPPX_data_extraction_prevent(const char* output, size_t len, int* contains_sensitive);

/* Training data sanitization */
int  SNEPPX_training_sanitize(const char* text, size_t len, char* sanitized, size_t* sanitized_len);

/* Model watermarking */
typedef struct {
    uint8_t watermark[32];
    int embedded;
} SNEPPXModelWatermark;

int  SNEPPX_watermark_init(SNEPPXModelWatermark* mw);
int  SNEPPX_watermark_embed(SNEPPXModelWatermark* mw, double* weights, size_t weight_count);
int  SNEPPX_watermark_verify(SNEPPXModelWatermark* mw, const double* weights, size_t weight_count);

/* Adversarial input perturbation */
int  SNEPPX_adversarial_smooth(double* input, size_t input_dim, double epsilon);

/* Output factuality scorer */
double SNEPPX_factuality_score(const char* statement, const char* reference);

/* Bias measurement */
typedef struct {
    double demographic_parity;
    double equalized_odds;
    int measured;
} SNEPPXBiasMetrics;

int  SNEPPX_bias_measure(SNEPPXBiasMetrics* bm, const double* predictions, const int* sensitive_attr, size_t n);

/* Prompt policy engine (RBAC) */
typedef struct {
    char policies[16][256];
    int policy_count;
    int enabled;
} SNEPPXPromptPolicy;

int  SNEPPX_prompt_policy_init(SNEPPXPromptPolicy* pp);
int  SNEPPX_prompt_policy_add(SNEPPXPromptPolicy* pp, const char* policy_rule);
int  SNEPPX_prompt_policy_enforce(SNEPPXPromptPolicy* pp, const char* prompt, size_t len);

#ifdef __cplusplus
}
#endif
#endif
