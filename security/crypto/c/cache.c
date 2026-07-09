#include "secure_cache_management.h"
#include <stdint.h>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__amd64__)
#include <emmintrin.h>
#elif defined(__aarch64__)
#include <arm_acle.h>
#endif

void SNEPPX_cache_flush(const void* ptr, size_t len) {
    if (!ptr) return;
    const uint8_t* p = (const uint8_t*)ptr;
    for (size_t i = 0; i < len; i += 64) {
#if defined(_MSC_VER)
        _mm_clflush(&p[i]);
#elif defined(__x86_64__) || defined(__amd64__)
        _mm_clflush(&p[i]);
#elif defined(__aarch64__)
        __asm__ volatile("dc civac, %0" :: "r"(&p[i]) : "memory");
#else
        (void)p;
#endif
    }
}

void SNEPPX_cache_prefetch(const void* ptr) {
    if (!ptr) return;
#if defined(_MSC_VER)
    _mm_prefetch((const char*)ptr, _MM_HINT_T0);
#elif defined(__x86_64__) || defined(__amd64__)
    _mm_prefetch((const char*)ptr, _MM_HINT_T0);
#elif defined(__aarch64__)
    __asm__ volatile("prfm pldl1keep, [%0]" :: "r"(ptr));
#else
    (void)ptr;
#endif
}

void SNEPPX_cache_barrier(void) {
#if defined(_MSC_VER)
    _mm_mfence();
#elif defined(__x86_64__) || defined(__amd64__)
    _mm_mfence();
#elif defined(__aarch64__)
    __asm__ volatile("dsb ish" : : : "memory");
#endif
}
