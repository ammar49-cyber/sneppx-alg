#ifndef SNEPPX_RLHF_SAFETY_H
#define SNEPPX_RLHF_SAFETY_H

#include <stdint.h>
#include <stddef.h>

typedef enum {
    RLHF_RULE_HARM = 0,
    RLHF_RULE_BIAS,
    RLHF_RULE_FACTUAL,
    RLHF_RULE_CUSTOM
} rlhf_rule_type_t;

typedef enum {
    RLHF_SEV_LOW = 0,
    RLHF_SEV_MEDIUM,
    RLHF_SEV_HIGH,
    RLHF_SEV_CRITICAL
} rlhf_severity_t;

typedef struct {
    double harm_threshold;
    double bias_threshold;
    double factual_threshold;
    double helpfulness_weight;
    double harmlessness_weight;
    double honesty_weight;
    double enable_refusal;
    double enable_critique;
    double enable_correction;
} rlhf_config_t;

typedef struct {
    double harmfulness;
    double bias;
    double factuality;
    double helpfulness;
    double honesty;
    double overall;
    int should_refuse;
} rlhf_score_t;

typedef struct {
    char policy_name[64];
    double score;
    double threshold;
    rlhf_severity_t severity;
} rlhf_violation_t;

typedef struct {
    uint64_t total_refusals;
    uint64_t total_corrections;
    int active_policies;
    uint64_t total_violations;
    rlhf_config_t config;
} rlhf_stats_t;

int SNEPPX_rlhf_add_policy(const char *name, double weight, rlhf_rule_type_t rule_type, double threshold);
int SNEPPX_rlhf_remove_policy(int policy_id);
int SNEPPX_rlhf_score(const char *input, const char *output, rlhf_score_t *score);
int SNEPPX_rlhf_check_policies(const char *text, rlhf_violation_t *violations, int max_violations);
int SNEPPX_rlhf_generate_refusal(char *out, size_t out_len, rlhf_score_t *score);
int SNEPPX_rlhf_generate_critique(const char *output, char *critique, size_t critique_len);
int SNEPPX_rlhf_correct_output(const char *output, char *corrected, size_t corrected_len);
int SNEPPX_rlhf_update_config(const rlhf_config_t *config);
int SNEPPX_rlhf_get_config(rlhf_config_t *config);
int SNEPPX_rlhf_get_stats(rlhf_stats_t *stats);
int SNEPPX_rlhf_reset(void);
int SNEPPX_rlhf_add_default_policies(void);

#endif
