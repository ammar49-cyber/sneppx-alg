#ifndef SNEPPX_PROMPT_FILTER_H
#define SNEPPX_PROMPT_FILTER_H
/*
 * S5 AI Sanitizer — Prompt Injection & Jailbreak Detection
 * Filters incoming prompts for known injection patterns, jailbreak attempts,
 * and adversarial instructions.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_MAX_PATTERNS 256
#define SNEPPX_PATTERN_MAX_LEN 128

typedef enum {
    SNEPPX_FILTER_CLEAN = 0,
    SNEPPX_FILTER_INJECTION = 1,
    SNEPPX_FILTER_JAILBREAK = 2,
    SNEPPX_FILTER_ADVERSARIAL = 3,
    SNEPPX_FILTER_SUSPICIOUS = 4,
} SNEPPXFilterResult;

typedef struct {
    char pattern[SNEPPX_PATTERN_MAX_LEN];
    SNEPPXFilterResult classification;
    int is_active;
} SNEPPXFilterPattern;

typedef struct {
    SNEPPXFilterPattern patterns[SNEPPX_MAX_PATTERNS];
    int pattern_count;
    int enabled;
    int max_token_length;
    double anomaly_threshold;
} SNEPPXPromptFilter;

int  SNEPPX_prompt_filter_init(SNEPPXPromptFilter* pf);
void SNEPPX_prompt_filter_destroy(SNEPPXPromptFilter* pf);
int  SNEPPX_prompt_filter_add_pattern(SNEPPXPromptFilter* pf, const char* pattern,
                                     SNEPPXFilterResult classification);
SNEPPXFilterResult SNEPPX_prompt_filter_scan(SNEPPXPromptFilter* pf,
                                          const char* prompt, size_t len);
int  SNEPPX_prompt_filter_sanitize(SNEPPXPromptFilter* pf,
                                  const char* prompt, size_t len,
                                  char* sanitized, size_t* sanitized_len);
void SNEPPX_prompt_filter_load_defaults(SNEPPXPromptFilter* pf);

#ifdef __cplusplus
}
#endif
#endif
