#ifndef ARIX_MEMORY_H
#define ARIX_MEMORY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Core Memory Functions
 * ============================================================ */
void* arix_malloc(size_t size, size_t alignment);
void  arix_free(void* ptr, size_t size);
void* arix_realloc(void* ptr, size_t old_size, size_t new_size, size_t alignment);
void  arix_secure_zero(void* ptr, size_t size);
void  arix_secure_copy(void* dst, const void* src, size_t size);

/* ============================================================
 * Pool Allocator — 18 size classes (16–8192 B)
 * Lock-free per-class stacks + thread-local caches
 * ============================================================ */
#define ARIX_NUM_SIZE_CLASSES  18
#define ARIX_POOL_MAX_SIZE     8192UL
#define ARIX_TLS_CACHE_MAX     32
#define ARIX_CHUNK_SIZE        (1024UL * 1024UL)   /* 1 MB */

/* ---------- Lock-free node & stack (Treiber) ---------- */
typedef struct ArixMemNode {
    struct ArixMemNode* next;   /* accessed atomically */
} ArixMemNode;

typedef struct {
    ArixMemNode* head;          /* accessed atomically via platform intrinsics */
} ArixLockFreeStack;

/* ---------- Per-size-class pool ---------- */
typedef struct {
    size_t            block_size;        /* bytes per block          */
    size_t            blocks_per_chunk;  /* blocks carved per chunk */
    ArixLockFreeStack stack;             /* free-list               */
    size_t            alloc_count;       /* lifetime allocs         */
    size_t            free_count;        /* lifetime frees          */
    size_t            chunk_count;       /* chunks allocated        */
} ArixMemPool;

/* ---------- Thread-local cache entry ---------- */
typedef struct {
    void*  free_list;    /* LIFO list of blocks */
    size_t count;        /* entries in list     */
    size_t max;          /* capacity            */
} ArixTlsEntry;

/* ---------- Thread-local cache ---------- */
typedef struct {
    ArixTlsEntry entries[ARIX_NUM_SIZE_CLASSES];
    size_t       hits;       /* TLS cache hits          */
    size_t       capacity;   /* max blocks per class    */
} ArixTlsCache;

/* ---------- Global pool statistics ---------- */
typedef struct {
    size_t total_pool_allocated;  /* bytes allocated lifetime       */
    size_t total_pool_freed;      /* bytes freed lifetime           */
    size_t total_chunks;          /* mmap/VirtualAlloc calls        */
    size_t total_pool_hits;       /* allocations served from pool   */
    size_t total_tls_hits;        /* allocations served from TLS    */
    size_t active_tls_caches;     /* number of active thread caches */
} ArixMemStats;

/* ---------- API ---------- */

/* Initialise the global pool (idempotent, thread-safe).  Returns 0 on success. */
int arix_mem_pool_init(void);

/* Destroy the global pool and release all chunks.  Not thread-safe. */
void arix_mem_pool_destroy(void);

/* Initialise / destroy the calling thread's TLS cache. */
void arix_tls_cache_init(void);
void arix_tls_cache_destroy(void);

/* Allocate a block of at least `size` bytes from the pool.
 * Returns 16-byte aligned zeroed memory, or NULL on failure.
 * Blocks larger than ARIX_POOL_MAX_SIZE fall back to arix_malloc. */
void* arix_pool_alloc(size_t size);

/* Return a block obtained from arix_pool_alloc back to the pool.
 * Safe to call with NULL.  Blocks larger than ARIX_POOL_MAX_SIZE
 * are forwarded to arix_free. */
void arix_pool_free(void* ptr, size_t size);

/* Copy current pool statistics into *stats. */
void arix_mem_pool_stats(ArixMemStats* stats);

/* Print pool statistics to stdout. */
void arix_mem_pool_print_stats(void);

#ifdef __cplusplus
}
#endif

#endif /* ARIX_MEMORY_H */
