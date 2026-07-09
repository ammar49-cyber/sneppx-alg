#ifndef SNEPPX_SECURE_ALLOCATOR_H
#define SNEPPX_SECURE_ALLOCATOR_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void*    addr;
    size_t   size;
    size_t   guard_front;
    size_t   guard_back;
    uint64_t canary;
    int      is_freed;
} SNEPPXSecureAllocRecord;

typedef struct SNEPPXSecureAllocator {
    size_t   total_allocated;
    size_t   peak_allocated;
    size_t   num_allocations;
    int      use_guard_pages;
    int      use_canaries;
    void*    live_allocations;
    int      (*on_overflow)(const SNEPPXSecureAllocRecord* record);
} SNEPPXSecureAllocator;

typedef struct {
    size_t total_allocated;
    size_t peak_allocated;
    size_t num_allocations;
    size_t num_frees;
    size_t num_double_free_detected;
    size_t num_canary_violations;
} SNEPPXSecureAllocStats;

int  SNEPPX_secure_allocator_init(SNEPPXSecureAllocator* alloc);
void SNEPPX_secure_allocator_destroy(SNEPPXSecureAllocator* alloc);

void* SNEPPX_secure_alloc(SNEPPXSecureAllocator* alloc, size_t bytes, size_t alignment);
void  SNEPPX_secure_free(SNEPPXSecureAllocator* alloc, void* ptr);
void  SNEPPX_secure_audit(SNEPPXSecureAllocator* alloc);

uint64_t SNEPPX_secure_canary_generate(void);
int      SNEPPX_secure_canary_check(void* ptr, uint64_t canary);

int                    SNEPPX_secure_freelist_check(SNEPPXSecureAllocator* alloc);
int                    SNEPPX_secure_free_quarantine(SNEPPXSecureAllocator* alloc, void* ptr);
void                   SNEPPX_secure_free_flush_quarantine(SNEPPXSecureAllocator* alloc);
SNEPPXSecureAllocStats   SNEPPX_secure_allocator_get_stats(SNEPPXSecureAllocator* alloc);

#ifdef __cplusplus
}
#endif
#endif
