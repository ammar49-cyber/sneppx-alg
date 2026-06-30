#ifndef ARIX_SECURE_ALLOCATOR_H
#define ARIX_SECURE_ALLOCATOR_H
/*
 * Secure Memory Allocator — v3.0 (production security)
 *
 * PURPOSE: Guards sensitive allocations (keys, passwords, model weights)
 * with guard pages, canaries, and automatic zeroing on free.  Allocations
 * are rounded up to page boundaries; adjacent guard pages (PROT_NONE)
 * detect buffer overflows.  A red-black tree tracks all live allocations
 * for audit and crash-report enumeration.
 *
 * DEPENDENCIES: polymorphic_memory_allocator.h, constant_time_operations.h
 * VERSION: v3.0
 */

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
} ArixSecureAllocRecord;

typedef struct ArixSecureAllocator {
    size_t   total_allocated;
    size_t   peak_allocated;
    size_t   num_allocations;
    int      use_guard_pages;
    int      use_canaries;
    void*    live_allocations;   /* red-black tree root */
    int      (*on_overflow)(const ArixSecureAllocRecord* record);
} ArixSecureAllocator;

int  arix_secure_allocator_init(ArixSecureAllocator* alloc);
void arix_secure_allocator_destroy(ArixSecureAllocator* alloc);

void* arix_secure_alloc(ArixSecureAllocator* alloc, size_t bytes, size_t alignment);
void  arix_secure_free(ArixSecureAllocator* alloc, void* ptr);
void  arix_secure_audit(ArixSecureAllocator* alloc);

/* ---------- Canary helpers ---------- */
uint64_t arix_secure_canary_generate(void);
int      arix_secure_canary_check(void* ptr, uint64_t canary);

#ifdef __cplusplus
}
#endif

#endif /* ARIX_SECURE_ALLOCATOR_H */
