#ifndef SNEPPX_TOKENIZER_H
#define SNEPPX_TOKENIZER_H

#include "multidimensional_tensor_engine.h"
#include <stddef.h>

typedef enum {
    SNEPPX_TOK_BPE,
} SNEPPXTokenizerType;

typedef struct {
    int pad_id;
    int bos_id;
    int eos_id;
    int unk_id;
} SNEPPXSpecialTokens;

typedef struct SNEPPXTokenizer SNEPPXTokenizer;

SNEPPXTokenizer*  SNEPPX_tokenizer_create(int vocab_size);
void            SNEPPX_tokenizer_destroy(SNEPPXTokenizer* tok);

int             SNEPPX_tokenizer_vocab_size(const SNEPPXTokenizer* tok);
SNEPPXSpecialTokens SNEPPX_tokenizer_special(const SNEPPXTokenizer* tok);
void            SNEPPX_tokenizer_set_special(SNEPPXTokenizer* tok, SNEPPXSpecialTokens sp);

int             SNEPPX_tokenizer_add_token(SNEPPXTokenizer* tok, const char* token, int id);

int             SNEPPX_tokenizer_encode(const SNEPPXTokenizer* tok, const char* text, int* out_ids, size_t max_len);
char*           SNEPPX_tokenizer_decode(const SNEPPXTokenizer* tok, const int* ids, size_t len);

SNEPPXTokenizer*  SNEPPX_tokenizer_train_bpe(const char** texts, size_t num_texts, size_t vocab_size);

int             SNEPPX_tokenizer_save(const SNEPPXTokenizer* tok, const char* path);
SNEPPXTokenizer*  SNEPPX_tokenizer_load(const char* path);

SNEPPXTensor*     SNEPPX_tokenizer_ids_to_tensor(const SNEPPXTokenizer* tok, const int* ids, size_t len);
int*            SNEPPX_tokenizer_tensor_to_ids(const SNEPPXTensor* t, size_t* out_len);

#endif
