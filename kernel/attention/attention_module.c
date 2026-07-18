#include "multi_head_attention_module.h"
#include <stdlib.h>

typedef struct SNEPPXMultiHeadAttention {
    int num_heads;
    int head_dim;
    int hidden_dim;
    int dropout;
    int is_causal;
} SNEPPXMultiHeadAttention;

SNEPPXMultiHeadAttention* SNEPPX_mha_create(int num_heads, int head_dim, int hidden_dim, int dropout, int is_causal, int use_flash) {
    (void)use_flash;
    SNEPPXMultiHeadAttention* mha = (SNEPPXMultiHeadAttention*)calloc(1, sizeof(SNEPPXMultiHeadAttention));
    if (!mha) return NULL;
    mha->num_heads = num_heads;
    mha->head_dim = head_dim;
    mha->hidden_dim = hidden_dim;
    mha->dropout = dropout;
    mha->is_causal = is_causal;
    return mha;
}

void SNEPPX_mha_destroy(SNEPPXMultiHeadAttention* mha) {
    free(mha);
}

SNEPPXTensor* SNEPPX_mha_forward(SNEPPXMultiHeadAttention* mha, const SNEPPXTensor* query, const SNEPPXTensor* key, const SNEPPXTensor* value, const SNEPPXTensor* mask) {
    (void)mha; (void)query; (void)key; (void)value; (void)mask;
    return NULL;
}

int SNEPPX_mha_get_output_dim(const SNEPPXMultiHeadAttention* mha) {
    return mha ? mha->num_heads * mha->head_dim : 0;
}
