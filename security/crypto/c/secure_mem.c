#include "arix_secure_mem.h"
#include "arix_canary.h"
#include "arix_aslr.h"
#include "arix_lock.h"
#include "arix_random.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

struct ArixSecurePool {
    uint8_t* base;
    size_t capacity;
    size_t used;
    size_t peak;
    int has_guards;
    int use_canaries;
    ArixCanary pool_canary;
    uint64_t alloc_count;
};

#if defined(_WIN32)
static size_t get_page_size(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize;
}
#else
static size_t get_page_size(void) {
    return (size_t)sysconf(_SC_PAGESIZE);
}
#endif

void arix_secure_zero(void* ptr, size_t len) {
    volatile uint8_t* p = (volatile uint8_t*)ptr;
    for (size_t i = 0; i < len; i++) p[i] = 0;
}

ArixSecurePool* arix_secure_pool_create(size_t size, const ArixSecureAllocConfig* config) {
    size_t page = get_page_size();
    size_t alloc_size = size + (config && config->guard_pages ? 2 * page : 0);
    size_t random_off = 0;
    uint8_t* base = NULL;

#if defined(_WIN32)
    base = (uint8_t*)VirtualAlloc(NULL, alloc_size + page, MEM_RESERVE, PAGE_NOACCESS);
    if (!base) return NULL;
    if (config && config->guard_pages) {
        VirtualAlloc(base + page, size, MEM_COMMIT, PAGE_READWRITE);
        VirtualAlloc(base, page, MEM_COMMIT, PAGE_NOACCESS);
        VirtualAlloc(base + page + size, page, MEM_COMMIT, PAGE_NOACCESS);
    } else {
        VirtualAlloc(base, size, MEM_COMMIT, PAGE_READWRITE);
    }
#else
    base = (uint8_t*)mmap(NULL, alloc_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) return NULL;
    if (config && config->guard_pages) {
        mprotect(base, page, PROT_NONE);
        mprotect(base + page + size, page, PROT_NONE);
    }
#endif

    ArixSecurePool* pool = (ArixSecurePool*)malloc(sizeof(ArixSecurePool));
    if (!pool) {
#if defined(_WIN32)
        VirtualFree(base, 0, MEM_RELEASE);
#else
        munmap(base, alloc_size);
#endif
        return NULL;
    }

    uint8_t* usable = base + (config && config->guard_pages ? page : 0);
    size_t usable_size = size;

    if (config && config->randomize_layout) {
        size_t max_off = usable_size / 4;
        if (max_off > page) max_off = page;
        random_off = arix_aslr_random_offset(max_off);
        usable += random_off;
        usable_size -= random_off;
    }

    if (config && config->lock_memory) {
        arix_mlock(usable, usable_size);
    }

    pool->base = usable;
    pool->capacity = usable_size;
    pool->used = 0;
    pool->peak = 0;
    pool->has_guards = (config && config->guard_pages) ? 1 : 0;
    pool->use_canaries = (config && config->canaries) ? 1 : 0;
    pool->alloc_count = 0;
    arix_canary_generate(&pool->pool_canary);
    return pool;
}

void arix_secure_pool_destroy(ArixSecurePool* pool) {
    if (!pool) return;
    arix_secure_zero(pool->base, pool->capacity);
    arix_munlock(pool->base, pool->capacity);
    size_t page = get_page_size();
    uint8_t* raw_base = pool->base - (pool->has_guards ? page : 0);
    size_t total = pool->capacity + (pool->has_guards ? 2 * page : 0);
#if defined(_WIN32)
    VirtualFree(raw_base, 0, MEM_RELEASE);
#else
    munmap(raw_base, total);
#endif
    arix_secure_zero((void*)pool, sizeof(ArixSecurePool));
    free(pool);
}

void* arix_secure_malloc(ArixSecurePool* pool, size_t size, size_t alignment) {
    if (!pool || !size) return NULL;
    if (alignment < 16) alignment = 16;
    size_t header = pool->use_canaries ? ARIX_CANARY_SIZE : 0;
    size_t aligned_offset = (pool->used + alignment - 1) & ~(alignment - 1);
    size_t needed = aligned_offset + header + size + header;
    if (needed > pool->capacity) return NULL;
    uint8_t* ptr = pool->base + aligned_offset + header;
    if (pool->use_canaries) {
        ArixCanary canary;
        arix_canary_generate(&canary);
        arix_canary_write(&canary, ptr - ARIX_CANARY_SIZE);
        arix_canary_write(&canary, ptr + size);
    }
    pool->used = needed;
    if (pool->used > pool->peak) pool->peak = pool->used;
    pool->alloc_count++;
    return ptr;
}

void arix_secure_free(ArixSecurePool* pool, void* ptr, size_t size) {
    if (!pool || !ptr) return;
    if (pool->use_canaries) {
        uint8_t* mem = (uint8_t*)ptr;
        ArixCanary* lead = (ArixCanary*)(mem - ARIX_CANARY_SIZE);
        ArixCanary* trail = (ArixCanary*)(mem + size);
        if (!arix_canary_verify(lead, lead->value)) {
            fprintf(stderr, "HEAP CORRUPTION: leading canary mismatch at %p\n", ptr);
            abort();
        }
        if (!arix_canary_verify(trail, trail->value)) {
            fprintf(stderr, "HEAP CORRUPTION: trailing canary mismatch at %p\n", ptr);
            abort();
        }
    }
    arix_secure_zero(ptr, size);
}

void* arix_secure_realloc(ArixSecurePool* pool, void* ptr, size_t old_size, size_t new_size, size_t alignment) {
    if (!pool) return NULL;
    if (!ptr) return arix_secure_malloc(pool, new_size, alignment);
    void* new_ptr = arix_secure_malloc(pool, new_size, alignment);
    if (!new_ptr) return NULL;
    size_t copy = old_size < new_size ? old_size : new_size;
    memcpy(new_ptr, ptr, copy);
    arix_secure_free(pool, ptr, old_size);
    return new_ptr;
}

void arix_secure_pool_stats(ArixSecurePool* pool, size_t* total, size_t* used, size_t* peak) {
    if (!pool) return;
    if (total) *total = pool->capacity;
    if (used) *used = pool->used;
    if (peak) *peak = pool->peak;
}
