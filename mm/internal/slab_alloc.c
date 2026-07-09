/*
 * Slab Allocator Implementation — SKELETON
 * VERSION: v0.5
 */

#include "slab_alloc.h"
#include <stdlib.h>
#include <string.h>

int SNEPPX_slab_cache_create(SNEPPXSlabCache** cache, size_t block_size, size_t alignment) {
    (void)block_size; (void)alignment;
    if (!cache) return -1;
    *cache = (SNEPPXSlabCache*)calloc(1, sizeof(SNEPPXSlabCache));
    return *cache ? 0 : -1;
}

void SNEPPX_slab_cache_destroy(SNEPPXSlabCache* cache) { free(cache); }

void* SNEPPX_slab_cache_alloc(SNEPPXSlabCache* cache) { (void)cache; return NULL; }
void  SNEPPX_slab_cache_free(SNEPPXSlabCache* cache, void* ptr) { (void)cache; (void)ptr; }
void  SNEPPX_slab_cache_gc(SNEPPXSlabCache* cache) { (void)cache; }

void SNEPPX_slab_local_init(SNEPPXSlabLocalCache* local, SNEPPXSlabCache* parent) {
    if (local) { memset(local, 0, sizeof(*local)); local->parent = parent; }
}
void SNEPPX_slab_local_destroy(SNEPPXSlabLocalCache* local) { (void)local; }
void* SNEPPX_slab_local_alloc(SNEPPXSlabLocalCache* local) { (void)local; return NULL; }
void  SNEPPX_slab_local_free(SNEPPXSlabLocalCache* local, void* ptr) { (void)local; (void)ptr; }
void  SNEPPX_slab_local_flush(SNEPPXSlabLocalCache* local) { (void)local; }
