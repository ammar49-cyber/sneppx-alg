#ifndef ARIX_MEMORY_INTERNAL_H
#define ARIX_MEMORY_INTERNAL_H
/*
 * Memory Allocator Internal — v0.5
 *
 * PURPOSE: Internal pool management, chunk allocation, and TLS cache
 * operations for the arix_memory pool allocator.  Implements the
 * Treiber-stack free lists, chunk carving, and per-thread caching.
 *
 * DEPENDENCIES: arix_memory.h
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void*  mem;             /* mmap'd chunk */
    size_t size;
    int    size_class_idx;
    int    slab_count;
} ArixMemChunk;

int  arix_mem_chunk_create(ArixMemChunk** chunk, size_t min_size);
void arix_mem_chunk_destroy(ArixMemChunk* chunk);

void* arix_mem_chunk_carve(ArixMemChunk* chunk, size_t block_size, size_t alignment);
int   arix_mem_chunk_has_space(const ArixMemChunk* chunk, size_t block_size);

/* ---------- Treiber stack operations ---------- */
void arix_lockfree_stack_push(void* stack_ptr, void* node_ptr);
void* arix_lockfree_stack_pop(void* stack_ptr);
int  arix_lockfree_stack_count(const void* stack_ptr);

/* ---------- TLS cache ---------- */
void arix_mem_tls_init(void);
void arix_mem_tls_cleanup(void);
void* arix_mem_tls_get(void);
void  arix_mem_tls_set(void* cache);
void  arix_mem_tls_flush(void);

#ifdef __cplusplus
}
#endif

#endif /* ARIX_MEMORY_INTERNAL_H */
