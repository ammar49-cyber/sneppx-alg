#ifndef ARIX_INTERNAL_SLAB_H
#define ARIX_INTERNAL_SLAB_H
/*
 * Slab Allocator — v0.5 (internal to arix_memory)
 *
 * PURPOSE: Per-size-class object caching with NUMA-aware hot/warm/cold
 * lists.  Each slab holds blocks of identical size carved from a large
 * chunk (1 MB).  Free blocks are tracked via an intrusive free list
 * (Treiber stack) for lock-free push/pop.
 *
 * Cache coloring offsets slab bases by a per-CPU stride to reduce
 * false sharing on frequently-accessed objects.
 *
 * DEPENDENCIES: polymorphic_memory_allocator.h (ArixMemPool, ArixMemNode)
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ArixSlab {
    struct ArixSlab* next;       /* linked list of slabs in this class */
    void*            mem_base;   /* mmap'd chunk base */
    size_t           block_size;
    size_t           num_blocks;
    size_t           free_count;
    void*            free_list;  /* intrusive LIFO of free blocks */
    int              numa_node;
    int              color;      /* cache color offset */
} ArixSlab;

typedef struct {
    size_t   block_size;
    size_t   blocks_per_slab;
    ArixSlab* partial_list;      /* slabs with some free blocks */
    ArixSlab* full_list;         /* slabs with all blocks allocated */
    ArixSlab* free_list;         /* slabs with all blocks free */
    size_t   active_objects;
    size_t   total_objects;
} ArixSlabCache;

/* ---------- CPU-local cache tier ---------- */
typedef struct {
    void*           free_blocks[32];
    int             count;
    ArixSlabCache*  parent;
} ArixSlabLocalCache;

/* ---------- API ---------- */
int arix_slab_cache_create(ArixSlabCache** cache, size_t block_size, size_t alignment);
void arix_slab_cache_destroy(ArixSlabCache* cache);
void* arix_slab_cache_alloc(ArixSlabCache* cache);
void  arix_slab_cache_free(ArixSlabCache* cache, void* ptr);
void  arix_slab_cache_gc(ArixSlabCache* cache);

/* ---------- Local cache ---------- */
void arix_slab_local_init(ArixSlabLocalCache* local, ArixSlabCache* parent);
void arix_slab_local_destroy(ArixSlabLocalCache* local);
void* arix_slab_local_alloc(ArixSlabLocalCache* local);
void  arix_slab_local_free(ArixSlabLocalCache* local, void* ptr);
void  arix_slab_local_flush(ArixSlabLocalCache* local);

#ifdef __cplusplus
}
#endif

#endif /* ARIX_INTERNAL_SLAB_H */
