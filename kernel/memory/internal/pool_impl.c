/*
 * Memory Pool Internal Implementation — SKELETON
 * VERSION: v0.5
 */

#include "pool_impl.h"
#include <stdlib.h>
#include <string.h>

int arix_mem_chunk_create(ArixMemChunk** chunk, size_t min_size) {
    (void)min_size;
    if (!chunk) return -1;
    *chunk = (ArixMemChunk*)calloc(1, sizeof(ArixMemChunk));
    return *chunk ? 0 : -1;
}

void arix_mem_chunk_destroy(ArixMemChunk* chunk) {
    if (chunk) free(chunk->mem);
    free(chunk);
}

void* arix_mem_chunk_carve(ArixMemChunk* chunk, size_t block_size, size_t alignment) {
    (void)chunk; (void)block_size; (void)alignment; return NULL;
}

int arix_mem_chunk_has_space(const ArixMemChunk* chunk, size_t block_size) {
    (void)chunk; (void)block_size; return 0;
}

void arix_lockfree_stack_push(void* stack_ptr, void* node_ptr) {
    (void)stack_ptr; (void)node_ptr;
}

void* arix_lockfree_stack_pop(void* stack_ptr) {
    (void)stack_ptr; return NULL;
}

int arix_lockfree_stack_count(const void* stack_ptr) {
    (void)stack_ptr; return 0;
}

void arix_mem_tls_init(void) {}
void arix_mem_tls_cleanup(void) {}
void* arix_mem_tls_get(void) { return NULL; }
void arix_mem_tls_set(void* cache) { (void)cache; }
void arix_mem_tls_flush(void) {}
