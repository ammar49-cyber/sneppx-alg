#ifndef SNEPPX_INTERNAL_VMEM_H
#define SNEPPX_INTERNAL_VMEM_H
/*
 * Virtual Memory Management — v0.5 (internal to SNEPPX_memory)
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

#define SNEPPX_PAGE_SIZE_4K    4096
#define SNEPPX_PAGE_SIZE_2M    (2UL * 1024UL * 1024UL)
#define SNEPPX_PAGE_SIZE_1G    (1024UL * 1024UL * 1024UL)

typedef enum {
    SNEPPX_VMEM_FLAG_HUGEPAGE = 1 << 0,
    SNEPPX_VMEM_FLAG_WRITABLE = 1 << 1,
    SNEPPX_VMEM_FLAG_EXECUTABLE = 1 << 2,
    SNEPPX_VMEM_FLAG_ZERO_INIT = 1 << 3,
} SNEPPXVMemFlags;

typedef struct {
    void*    addr;
    size_t   size;
    size_t   page_size;
    int      flags;
    int      numa_node;
} SNEPPXVMemRegion;

typedef struct SNEPPXVMemAllocator {
    size_t   total_reserved;
    size_t   total_committed;
    size_t   peak_committed;
    size_t   page_size_default;
    int      (*on_oom)(size_t requested_bytes);
} SNEPPXVMemAllocator;

/* ---------- API ---------- */
void SNEPPX_vmem_init(SNEPPXVMemAllocator* alloc);
void SNEPPX_vmem_cleanup(SNEPPXVMemAllocator* alloc);

void* SNEPPX_vmem_reserve(SNEPPXVMemAllocator* alloc, size_t bytes, size_t alignment, int flags);
int   SNEPPX_vmem_commit(SNEPPXVMemAllocator* alloc, void* addr, size_t bytes);
int   SNEPPX_vmem_decommit(SNEPPXVMemAllocator* alloc, void* addr, size_t bytes);
void  SNEPPX_vmem_release(SNEPPXVMemAllocator* alloc, void* addr, size_t bytes);

int   SNEPPX_vmem_advise_hugepage(void* addr, size_t bytes);
int   SNEPPX_vmem_advise_nohugepage(void* addr, size_t bytes);

/* ---------- Eviction (v0.5) ---------- */
typedef enum { SNEPPX_EVICT_LRU, SNEPPX_EVICT_LFU, SNEPPX_EVICT_CUSTOM } SNEPPXEvictPolicy;

typedef struct SNEPPXEvictStrategy {
    SNEPPXEvictPolicy policy;
    void*           state;
    int             (*select_victim)(struct SNEPPXEvictStrategy* strat, void** victim_addr, size_t* victim_size);
    void            (*on_access)(struct SNEPPXEvictStrategy* strat, void* addr);
} SNEPPXEvictStrategy;

int SNEPPX_vmem_register_evict_strategy(SNEPPXVMemAllocator* alloc, SNEPPXEvictStrategy* strat);
int SNEPPX_vmem_evict_page(SNEPPXVMemAllocator* alloc, void* addr, size_t size);

/* ---------- OOM callback ---------- */
void SNEPPX_vmem_set_oom_handler(SNEPPXVMemAllocator* alloc, int (*handler)(size_t));

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_INTERNAL_VMEM_H */
