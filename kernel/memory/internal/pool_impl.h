#ifndef SNEPPX_MEMORY_INTERNAL_H
#define SNEPPX_MEMORY_INTERNAL_H
/*
 * Memory Allocator Internal — v0.5
 *
 * PURPOSE: Internal pool management, chunk allocation, and TLS cache
 * operations for the SNEPPX_memory pool allocator.  Implements the
 * Treiber-stack free lists, chunk carving, and per-thread caching.
 *
 * DEPENDENCIES: polymorphic_memory_allocator.h
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
} SNEPPXMemChunk;

int  SNEPPX_mem_chunk_create(SNEPPXMemChunk** chunk, size_t min_size);
void SNEPPX_mem_chunk_destroy(SNEPPXMemChunk* chunk);

void* SNEPPX_mem_chunk_carve(SNEPPXMemChunk* chunk, size_t block_size, size_t alignment);
int   SNEPPX_mem_chunk_has_space(const SNEPPXMemChunk* chunk, size_t block_size);

/* ---------- Treiber stack operations ---------- */
void SNEPPX_lockfree_stack_push(void* stack_ptr, void* node_ptr);
void* SNEPPX_lockfree_stack_pop(void* stack_ptr);
int  SNEPPX_lockfree_stack_count(const void* stack_ptr);

/* ---------- TLS cache ---------- */
void SNEPPX_mem_tls_init(void);
void SNEPPX_mem_tls_cleanup(void);
void* SNEPPX_mem_tls_get(void);
void  SNEPPX_mem_tls_set(void* cache);
void  SNEPPX_mem_tls_flush(void);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_MEMORY_INTERNAL_H */
