#ifndef ARIX_DATA_PIPELINE_H
#define ARIX_DATA_PIPELINE_H

#include "subword_tokenization_pipeline.h"
#include "multidimensional_tensor_engine.h"
#include <stddef.h>

typedef struct ArixTextDataset ArixTextDataset;

ArixTextDataset* arix_text_dataset_create(const char* path, ArixTokenizer* tok,
                                           size_t seq_len, int line_by_line);
void arix_text_dataset_destroy(ArixTextDataset* ds);

size_t arix_text_dataset_size(const ArixTextDataset* ds);

int arix_text_dataset_get_batch(const ArixTextDataset* ds, size_t start_idx, size_t batch_size,
                                 ArixTensor** input_ids, ArixTensor** target_ids);

#endif
