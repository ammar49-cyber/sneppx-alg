#ifndef ARIX_INFERENCE_ENGINE_H
#define ARIX_INFERENCE_ENGINE_H

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
} ArixGenerationConfig;

ArixGenerationConfig arix_generation_config_default(void);

int arix_sample_from_logits(const float* logits, size_t vocab_size, ArixGenerationConfig* cfg);

int arix_argmax(const float* logits, size_t n);

char* arix_generate(ArixTensor* embed_weight, ArixTensor* unembed_weight,
                    ArixAttentionWeights* attn, ArixTokenizer* tok,
                    const char* prompt, ArixGenerationConfig* cfg);

int arix_generate_tokens(ArixTensor* embed_weight, ArixTensor* unembed_weight,
                         ArixAttentionWeights* attn,
                         const int* input_ids, size_t input_len,
                         int* output_ids, size_t max_output_len,
                         ArixGenerationConfig* cfg);

#endif
