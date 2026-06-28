#include "arix_timing.h"
#include "arix_sc.h"
#include "arix_ct.h"
#include "arix_random.h"

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(_WIN32)
#include <windows.h>
#elif defined(__x86_64__) || defined(__amd64__)
#include <x86intrin.h>
#elif defined(__aarch64__)
#include <arm_acle.h>
#endif

uint64_t arix_timing_start(void) {
#if defined(_MSC_VER)
    __cpuidex((int[]){0,0,0,0}, 0, 0);
    return __rdtsc();
#elif defined(__x86_64__) || defined(__amd64__)
    uint32_t lo, hi;
    __asm__ volatile("cpuid\n\t"
                     "rdtsc\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     : "=r"(hi), "=r"(lo)
                     :
                     : "%rax", "%rbx", "%rcx", "%rdx");
    return ((uint64_t)hi << 32) | (uint64_t)lo;
#elif defined(_WIN32)
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (uint64_t)t.QuadPart;
#elif defined(__aarch64__)
    __asm__ volatile("isb" : : : "memory");
    uint64_t t;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(t));
    return t;
#else
    return 0;
#endif
}

uint64_t arix_timing_end(void) {
#if defined(_MSC_VER)
    unsigned int aux;
    uint64_t t = __rdtscp(&aux);
    __cpuidex((int[]){0,0,0,0}, 0, 0);
    return t;
#elif defined(__x86_64__) || defined(__amd64__)
    uint32_t lo, hi;
    __asm__ volatile("rdtscp\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "cpuid\n\t"
                     : "=r"(hi), "=r"(lo)
                     :
                     : "%rax", "%rbx", "%rcx", "%rdx");
    return ((uint64_t)hi << 32) | (uint64_t)lo;
#elif defined(_WIN32)
    LARGE_INTEGER freq, t;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (uint64_t)(t.QuadPart * 1000000000ULL / freq.QuadPart);
#elif defined(__aarch64__)
    uint64_t t;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(t));
    __asm__ volatile("isb" : : : "memory");
    return t;
#else
    return 0;
#endif
}

void arix_timing_random_delay(uint32_t min_ns, uint32_t max_ns) {
    if (max_ns <= min_ns) return;
    uint32_t range = max_ns - min_ns;
    uint32_t r;
    arix_random_bytes((uint8_t*)&r, sizeof(r));
    uint32_t delay = min_ns + (r % range);
    uint64_t start = arix_timing_start();
    while (1) {
        uint64_t elapsed = arix_timing_end() - start;
        if (elapsed >= delay) break;
    }
}

int arix_timing_safe_equal(const uint8_t* a, const uint8_t* b, size_t len, uint64_t* timing_ns) {
    uint64_t t0 = arix_timing_start();
    int result = arix_ct_equal(a, b, len);
    uint64_t t1 = arix_timing_end();
    if (timing_ns) {
#if defined(__x86_64__) || defined(__amd64__) || defined(_MSC_VER)
        *timing_ns = (t1 - t0) / 2400;
#else
        *timing_ns = t1 - t0;
#endif
    }
    return result;
}
