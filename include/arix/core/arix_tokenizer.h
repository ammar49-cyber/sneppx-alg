#ifndef ARIX_TOKENIZER_H
#define ARIX_TOKENIZER_H

#include "arix_tensor.h"
#include <stddef.h>

typedef enum {
    ARIX_TOK_BPE,
} ArixTokenizerType;

typedef struct {
    int pad_id;
    int bos_id;
    int eos_id;
    int unk_id;
} ArixSpecialTokens;

typedef struct ArixTokenizer ArixTokenizer;

ArixTokenizer*  arix_tokenizer_create(int vocab_size);
void            arix_tokenizer_destroy(ArixTokenizer* tok);

int             arix_tokenizer_vocab_size(const ArixTokenizer* tok);
ArixSpecialTokens arix_tokenizer_special(const ArixTokenizer* tok);
void            arix_tokenizer_set_special(ArixTokenizer* tok, ArixSpecialTokens sp);

int             arix_tokenizer_add_token(ArixTokenizer* tok, const char* token, int id);

int             arix_tokenizer_encode(const ArixTokenizer* tok, const char* text, int* out_ids, size_t max_len);
char*           arix_tokenizer_decode(const ArixTokenizer* tok, const int* ids, size_t len);

ArixTokenizer*  arix_tokenizer_train_bpe(const char** texts, size_t num_texts, size_t vocab_size);

int             arix_tokenizer_save(const ArixTokenizer* tok, const char* path);
ArixTokenizer*  arix_tokenizer_load(const char* path);

ArixTensor*     arix_tokenizer_ids_to_tensor(const ArixTokenizer* tok, const int* ids, size_t len);
int*            arix_tokenizer_tensor_to_ids(const ArixTensor* t, size_t* out_len);

#endif
