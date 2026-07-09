#include "data_pipeline.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

struct SNEPPXTextDataset {
    unsigned char* token_data;
    size_t* sample_offsets;
    size_t num_samples;
    size_t seq_len;
    size_t alloc_size;
    SNEPPXTokenizer* tok;
};

SNEPPXTextDataset* SNEPPX_text_dataset_create(const char* path, SNEPPXTokenizer* tok,
                                           size_t seq_len, int line_by_line) {
    if (!path || !tok || seq_len == 0) return NULL;
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    if (fsize <= 0) { fclose(f); return NULL; }
    fseek(f, 0, SEEK_SET);
    char* text = (char*)SNEPPX_malloc((size_t)fsize + 1, 64);
    if (!text) { fclose(f); return NULL; }
    size_t nread = fread(text, 1, (size_t)fsize, f);
    fclose(f);
    text[nread] = '\0';

    size_t max_lines = 0;
    for (size_t i = 0; i < nread; i++) {
        if (text[i] == '\n') max_lines++;
        if (max_lines == 0 && i == nread - 1 && text[i] != '\n') max_lines = 1;
    }
    if (max_lines == 0) { SNEPPX_free(text, (size_t)fsize + 1); return NULL; }

    size_t max_token_buf = (size_t)fsize * sizeof(int);
    size_t max_samples = (size_t)fsize;
    size_t alloc_size = max_token_buf + max_samples * sizeof(size_t);
    unsigned char* token_data = (unsigned char*)SNEPPX_malloc(alloc_size, 64);
    if (!token_data) { SNEPPX_free(text, (size_t)fsize + 1); return NULL; }
    int* td = (int*)token_data;
    size_t* offsets = (size_t*)(token_data + max_token_buf);
    size_t actual_samples = 0;
    size_t token_cursor = 0;

    if (line_by_line) {
        char* line_start = text;
        for (size_t i = 0; i <= nread && actual_samples < max_lines; i++) {
            if (i == nread || text[i] == '\n') {
                text[i] = '\0';
                if (line_start < text + i) {
                    int n = SNEPPX_tokenizer_encode(tok, line_start, (int*)(td + token_cursor), max_token_buf / sizeof(int) - token_cursor);
                    if (n > 0) {
                        offsets[actual_samples] = token_cursor * sizeof(int);
                        token_cursor += (size_t)n;
                        actual_samples++;
                    }
                }
                line_start = text + i + 1;
            }
        }
    } else {
        int total_tokens = SNEPPX_tokenizer_encode(tok, text, (int*)td, (int)(max_token_buf / sizeof(int)));
        if (total_tokens <= 0) { SNEPPX_free(token_data, alloc_size); SNEPPX_free(text, (size_t)fsize + 1); return NULL; }
        size_t num_chunks = (size_t)total_tokens / seq_len;
        if (num_chunks == 0) { SNEPPX_free(token_data, alloc_size); SNEPPX_free(text, (size_t)fsize + 1); return NULL; }
        for (size_t i = 0; i < num_chunks; i++) {
            offsets[i] = i * seq_len * sizeof(int);
            actual_samples++;
        }
    }

    SNEPPX_free(text, (size_t)fsize + 1);

    SNEPPXTextDataset* ds = (SNEPPXTextDataset*)SNEPPX_malloc(sizeof(SNEPPXTextDataset), 64);
    if (!ds) { SNEPPX_free(token_data, alloc_size); return NULL; }
    ds->token_data = token_data;
    ds->sample_offsets = offsets;
    ds->num_samples = actual_samples;
    ds->seq_len = seq_len;
    ds->alloc_size = max_token_buf + max_samples * sizeof(size_t);
    ds->tok = tok;
    return ds;
}

void SNEPPX_text_dataset_destroy(SNEPPXTextDataset* ds) {
    if (!ds) return;
    SNEPPX_free(ds->token_data, ds->alloc_size);
    SNEPPX_free(ds, sizeof(SNEPPXTextDataset));
}

size_t SNEPPX_text_dataset_size(const SNEPPXTextDataset* ds) {
    return ds ? ds->num_samples : 0;
}

int SNEPPX_text_dataset_get_batch(const SNEPPXTextDataset* ds, size_t start_idx, size_t batch_size,
                                 SNEPPXTensor** input_ids, SNEPPXTensor** target_ids) {
    if (!ds || !input_ids || !target_ids) return 1;
    if (start_idx >= ds->num_samples) return 1;
    size_t actual = batch_size;
    if (start_idx + actual > ds->num_samples) actual = ds->num_samples - start_idx;

    size_t in_shape[] = {actual, ds->seq_len};
    SNEPPXTensor* in_t = SNEPPX_tensor_empty(in_shape, 2, SNEPPX_INT32);
    SNEPPXTensor* tgt_t = SNEPPX_tensor_empty(in_shape, 2, SNEPPX_INT32);
    if (!in_t || !tgt_t) {
        SNEPPX_tensor_destroy(in_t); SNEPPX_tensor_destroy(tgt_t);
        return 1;
    }

    int* in_d = (int*)in_t->data;
    int* tgt_d = (int*)tgt_t->data;
    int* src = (int*)ds->token_data;

    for (size_t i = 0; i < actual; i++) {
        size_t sidx = start_idx + i;
        size_t sample_offset = ds->sample_offsets[sidx] / sizeof(int);
        for (size_t j = 0; j + 1 < ds->seq_len && sample_offset + j + 1 < ds->num_samples * ds->seq_len; j++) {
            in_d[i * ds->seq_len + j] = src[sample_offset + j];
            tgt_d[i * ds->seq_len + j] = src[sample_offset + j + 1];
        }
    }

    *input_ids = in_t;
    *target_ids = tgt_t;
    return 0;
}
