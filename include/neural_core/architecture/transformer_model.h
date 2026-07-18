#ifndef SNEPPX_TRANSFORMER_MODEL_H
#define SNEPPX_TRANSFORMER_MODEL_H
#include <stddef.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct SNEPPXTransformer SNEPPXTransformer;

SNEPPXTransformer* SNEPPX_transformer_create(size_t vocab_size, size_t hidden_dim,
    size_t num_heads, size_t num_layers, size_t ffn_dim, float dropout, int use_rope);
void SNEPPX_transformer_destroy(void* model);
int SNEPPX_transformer_forward(void* model, const int* input_ids, size_t seq_len, float* logits);
int SNEPPX_transformer_generate(void* model, const int* prompt, size_t prompt_len,
    int* output, size_t max_len, int temperature, int top_k);

#ifdef __cplusplus
}
#endif
#endif
