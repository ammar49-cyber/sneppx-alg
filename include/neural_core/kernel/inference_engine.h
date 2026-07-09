#ifndef SNEPPX_INFERENCE_ENGINE_H
#define SNEPPX_INFERENCE_ENGINE_H

#include "multi_head_attention_module.h"
#include "subword_tokenization_pipeline.h"
#include "multidimensional_tensor_engine.h"
#include <stddef.h>

typedef struct {
    float temperature;
    float top_p;
    int top_k;
    int max_new_tokens;
    int eos_token_id;
} SNEPPXGenerationConfig;

SNEPPXGenerationConfig SNEPPX_generation_config_default(void);

int SNEPPX_sample_from_logits(const float* logits, size_t vocab_size, SNEPPXGenerationConfig* cfg);

int SNEPPX_argmax(const float* logits, size_t n);

char* SNEPPX_generate(SNEPPXTensor* embed_weight, SNEPPXTensor* unembed_weight,
                    SNEPPXAttentionWeights* attn, SNEPPXTokenizer* tok,
                    const char* prompt, SNEPPXGenerationConfig* cfg);

int SNEPPX_generate_tokens(SNEPPXTensor* embed_weight, SNEPPXTensor* unembed_weight,
                         SNEPPXAttentionWeights* attn,
                         const int* input_ids, size_t input_len,
                         int* output_ids, size_t max_output_len,
                         SNEPPXGenerationConfig* cfg);

#endif
