#ifndef ARIX_INTERNAL_VMEM_H
#define ARIX_INTERNAL_VMEM_H
/*
 * Virtual Memory Management — v0.5 (internal to arix_memory)
 *
 * PURPOSE: Huge-page-aware virtual memory allocator using mmap/map
 * with MAP_HUGETLB / MADV_HUGEPAGE hints.  Tracks page table entries,
 * handles OOM via registered callbacks, and supports eviction of
 * cold tensor pages to disk (swap).
 *
 * DEPENDENCIES: polymorphic_memory_allocator.h
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ARIX_PAGE_SIZE_4K    4096
#define ARIX_PAGE_SIZE_2M    (2UL * 1024UL * 1024UL)
#define ARIX_PAGE_SIZE_1G    (1024UL * 1024UL * 1024UL)

typedef enum {
    ARIX_VMEM_FLAG_HUGEPAGE = 1 << 0,
    ARIX_VMEM_FLAG_WRITABLE = 1 << 1,
    ARIX_VMEM_FLAG_EXECUTABLE = 1 << 2,
    ARIX_VMEM_FLAG_ZERO_INIT = 1 << 3,
} ArixVMemFlags;

typedef struct {
    void*    addr;
    size_t   size;
    size_t   page_size;
    int      flags;
    int      numa_node;
} ArixVMemRegion;

typedef struct ArixVMemAllocator {
    size_t   total_reserved;
    size_t   total_committed;
    size_t   peak_committed;
    size_t   page_size_default;
    int      (*on_oom)(size_t requested_bytes);
} ArixVMemAllocator;

/* ---------- API ---------- */
void arix_vmem_init(ArixVMemAllocator* alloc);
void arix_vmem_cleanup(ArixVMemAllocator* alloc);

void* arix_vmem_reserve(ArixVMemAllocator* alloc, size_t bytes, size_t alignment, int flags);
int   arix_vmem_commit(ArixVMemAllocator* alloc, void* addr, size_t bytes);
int   arix_vmem_decommit(ArixVMemAllocator* alloc, void* addr, size_t bytes);
void  arix_vmem_release(ArixVMemAllocator* alloc, void* addr, size_t bytes);

int   arix_vmem_advise_hugepage(void* addr, size_t bytes);
int   arix_vmem_advise_nohugepage(void* addr, size_t bytes);

/* ---------- Eviction (v0.5) ---------- */
typedef enum { ARIX_EVICT_LRU, ARIX_EVICT_LFU, ARIX_EVICT_CUSTOM } ArixEvictPolicy;

typedef struct ArixEvictStrategy {
    ArixEvictPolicy policy;
    void*           state;
    int             (*select_victim)(struct ArixEvictStrategy* strat, void** victim_addr, size_t* victim_size);
    void            (*on_access)(struct ArixEvictStrategy* strat, void* addr);
} ArixEvictStrategy;

int arix_vmem_register_evict_strategy(ArixVMemAllocator* alloc, ArixEvictStrategy* strat);
int arix_vmem_evict_page(ArixVMemAllocator* alloc, void* addr, size_t size);

/* ---------- OOM callback ---------- */
void arix_vmem_set_oom_handler(ArixVMemAllocator* alloc, int (*handler)(size_t));

#ifdef __cplusplus
}
#endif

#endif /* ARIX_INTERNAL_VMEM_H */
