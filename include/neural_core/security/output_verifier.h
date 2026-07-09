#ifndef SNEPPX_OUTPUT_VERIFIER_H
#define SNEPPX_OUTPUT_VERIFIER_H
/*
 * S5 AI Sanitizer — Output Verification
 * Validates AI outputs for toxicity, bias, factual consistency,
 * and policy compliance before delivery.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_MAX_TOPIC_BLOCKLIST 128
#define SNEPPX_TOPIC_MAX_LEN 64

typedef struct {
    char topic[SNEPPX_TOPIC_MAX_LEN];
    int is_blocked;
} SNEPPXBlockedTopic;

typedef struct {
    SNEPPXBlockedTopic topics[SNEPPX_MAX_TOPIC_BLOCKLIST];
    int topic_count;
    double toxicity_threshold;
    double bias_threshold;
    int check_factual_consistency;
    int max_output_length;
} SNEPPXS5Verifier;

int  SNEPPX_s5_verifier_init(SNEPPXS5Verifier* ov);
void SNEPPX_s5_verifier_destroy(SNEPPXS5Verifier* ov);
int  SNEPPX_s5_verifier_add_blocked_topic(SNEPPXS5Verifier* ov, const char* topic);
int  SNEPPX_s5_verifier_check(SNEPPXS5Verifier* ov, const char* output, size_t len);
int  SNEPPX_s5_verifier_sanitize(SNEPPXS5Verifier* ov,
                                const char* output, size_t len,
                                char* safe_output, size_t* safe_len);

#ifdef __cplusplus
}
#endif
#endif
