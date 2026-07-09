#ifndef SNEPPX_SECURE_MEM_H
#define SNEPPX_SECURE_MEM_H

#include <stddef.h>
#include <stdint.h>

typedef struct SNEPPXSecurePool SNEPPXSecurePool;

typedef struct {
    int guard_pages;
    int canaries;
    int lock_memory;
    int randomize_layout;
} SNEPPXSecureAllocConfig;

SNEPPXSecurePool* SNEPPX_secure_pool_create(size_t size, const SNEPPXSecureAllocConfig* config);
void SNEPPX_secure_pool_destroy(SNEPPXSecurePool* pool);
void* SNEPPX_secure_malloc(SNEPPXSecurePool* pool, size_t size, size_t alignment);
void SNEPPX_secure_free(SNEPPXSecurePool* pool, void* ptr, size_t size);
void* SNEPPX_secure_realloc(SNEPPXSecurePool* pool, void* ptr, size_t old_size, size_t new_size, size_t alignment);
void SNEPPX_secure_pool_stats(SNEPPXSecurePool* pool, size_t* total, size_t* used, size_t* peak);
void SNEPPX_secure_zero(void* ptr, size_t len);

#endif
