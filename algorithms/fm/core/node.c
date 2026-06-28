#include "arix_fm.h"
#include "arix_memory.h"
#include <string.h>

ArixFMNode* arix_fm_node_create(size_t node_id, size_t memory_dim, size_t capacity) {
    ArixFMNode* node = (ArixFMNode*)arix_malloc(sizeof(ArixFMNode), 64);
    if (!node) return NULL;
    memset(node, 0, sizeof(ArixFMNode));

    node->node_id = node_id;
    node->memory_bank = arix_fm_memory_bank_create(memory_dim, capacity);
    if (!node->memory_bank) {
        arix_free(node, sizeof(ArixFMNode));
        return NULL;
    }

    size_t grad_shape[] = {memory_dim, memory_dim};
    node->gradient_accumulator = arix_tensor_zeros(grad_shape, 2, ARIX_FLOAT32);
    if (!node->gradient_accumulator) {
        arix_fm_memory_bank_destroy(node->memory_bank);
        arix_free(node, sizeof(ArixFMNode));
        return NULL;
    }

    node->last_sync_time = 0;
    node->is_online = 1;
    node->trust_score = 1.0f;
    return node;
}

void arix_fm_node_destroy(ArixFMNode* node) {
    if (!node) return;
    if (node->memory_bank) arix_fm_memory_bank_destroy(node->memory_bank);
    if (node->gradient_accumulator) arix_tensor_destroy(node->gradient_accumulator);
    arix_free(node, sizeof(ArixFMNode));
}
