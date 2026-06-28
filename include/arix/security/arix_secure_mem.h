#ifndef ARIX_SECURE_MEM_H
#define ARIX_SECURE_MEM_H

#include <stddef.h>
#include <stdint.h>

typedef struct ArixSecurePool ArixSecurePool;

typedef struct {
    int guard_pages;
    int canaries;
    int lock_memory;
    int randomize_layout;
} ArixSecureAllocConfig;

ArixSecurePool* arix_secure_pool_create(size_t size, const ArixSecureAllocConfig* config);
void arix_secure_pool_destroy(ArixSecurePool* pool);
void* arix_secure_malloc(ArixSecurePool* pool, size_t size, size_t alignment);
void arix_secure_free(ArixSecurePool* pool, void* ptr, size_t size);
void* arix_secure_realloc(ArixSecurePool* pool, void* ptr, size_t old_size, size_t new_size, size_t alignment);
void arix_secure_pool_stats(ArixSecurePool* pool, size_t* total, size_t* used, size_t* peak);
void arix_secure_zero(void* ptr, size_t len);

#endif
