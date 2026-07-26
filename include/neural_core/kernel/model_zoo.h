#ifndef SNEPPX_MODEL_ZOO_H
#define SNEPPX_MODEL_ZOO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Model family identifiers
 * ========================================================================= */

typedef enum {
    SNEPPX_MODEL_LLAMA_2   = 0,
    SNEPPX_MODEL_LLAMA_3   = 1,
    SNEPPX_MODEL_MISTRAL   = 2,
    SNEPPX_MODEL_QWEN_2    = 3,
    SNEPPX_MODEL_DEEPSEEK_V2 = 4,
    SNEPPX_MODEL_UNKNOWN   = 255
} SNEPPXModelFamily;

/* =========================================================================
 * LLaMA 2/3 config
 * ========================================================================= */

typedef struct {
    SNEPPXModelFamily family;
    size_t hidden_size;
    size_t intermediate_size;
    size_t num_hidden_layers;
    size_t num_attention_heads;
    size_t num_key_value_heads;     /* GQA: can differ from num_attention_heads */
    size_t vocab_size;
    size_t max_position_embeddings;
    float  rms_norm_eps;
    float  rope_theta;
    int    use_scaled_rope;         /* LLaMA 3: scaled RoPE */
    int    tie_word_embeddings;
    int    hidden_act;              /* 0=SiLU, 1=GELU, 2=ReLU */
    int    num_hidden_groups;       /* grouped-query attention groups */
    int    head_dim;
    size_t num_experts;
    size_t num_experts_per_tok;
    int    use_flash_attn;
    int    use_sdpa;
    int    sliding_window;
    float  attention_dropout;
    float  hidden_dropout;
} SNEPPXLlamaConfig;

/* =========================================================================
 * Mistral config
 * ========================================================================= */

typedef struct {
    SNEPPXModelFamily family;
    size_t hidden_size;
    size_t intermediate_size;
    size_t num_hidden_layers;
    size_t num_attention_heads;
    size_t num_key_value_heads;
    size_t vocab_size;
    size_t max_position_embeddings;
    float  rms_norm_eps;
    float  rope_theta;
    int    sliding_window;
    int    head_dim;
    float  attention_dropout;
    float  hidden_dropout;
} SNEPPXMistralConfig;

/* =========================================================================
 * Qwen 2 config
 * ========================================================================= */

typedef struct {
    SNEPPXModelFamily family;
    size_t hidden_size;
    size_t intermediate_size;
    size_t num_hidden_layers;
    size_t num_attention_heads;
    size_t num_key_value_heads;
    size_t vocab_size;
    size_t max_position_embeddings;
    float  rms_norm_eps;
    float  rope_theta;
    float  rope_scaling_factor;     /* YARN / NTK scaling */
    int    use_rope_scaling;
    int    head_dim;
    float  attention_dropout;
    float  hidden_dropout;
} SNEPPXQwen2Config;

/* =========================================================================
 * DeepSeek V2 config (with MLA)
 * ========================================================================= */

typedef struct {
    SNEPPXModelFamily family;
    size_t hidden_size;
    size_t intermediate_size;
    size_t num_hidden_layers;
    size_t num_attention_heads;
    size_t num_key_value_heads;     /* kv heads for MLA absorption */
    size_t vocab_size;
    size_t max_position_embeddings;
    float  rms_norm_eps;
    float  rope_theta;
    int    head_dim;
    int    kv_lora_rank;            /* MLA: absorbed KV rank (DeepSeek V2: 512) */
    int    q_lora_rank;             /* MLA: absorbed Q rank (optional) */
    float  attention_dropout;
    float  hidden_dropout;
    int    use_flash_attn;
} SNEPPXDeepSeekV2Config;

/* =========================================================================
 * Generic LLM config (union wrapper)
 * ========================================================================= */

typedef struct {
    SNEPPXModelFamily family;
    union {
        SNEPPXLlamaConfig      llama;
        SNEPPXMistralConfig    mistral;
        SNEPPXQwen2Config      qwen2;
        SNEPPXDeepSeekV2Config deepseek_v2;
    } config;
} SNEPPXLLMConfig;

/* =========================================================================
 * API functions
 * ========================================================================= */

/* Create a named config preset by family + size (e.g. "llama3", "8B") */
int SNEPPX_llm_config_from_name(const char* family, const char* size,
                                 SNEPPXLLMConfig* out);

/* Serialize LLM config to JSON (caller must free) */
char* SNEPPX_llm_config_to_json(const SNEPPXLLMConfig* cfg);

/* Parse LLM config from JSON string */
int SNEPPX_llm_config_from_json(const char* json, SNEPPXLLMConfig* out);

/* Extend model context to target_len using NTK-aware/YaRN scaling */
int SNEPPX_llm_config_extend_context(SNEPPXLLMConfig* cfg, size_t target_len);

/* Weight name mapping helpers */
const char* SNEPPX_llm_weight_prefix(SNEPPXModelFamily family);
int SNEPPX_llm_num_weight_tensors(SNEPPXModelFamily family,
                                    size_t num_layers);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_MODEL_ZOO_H */
