#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "arix_cache.h"

static int tests_passed = 0, tests_failed = 0;

#define TEST(name, expr) do { \
    if (!(expr)) { printf("FAIL: %s\n", name); tests_failed++; } \
    else { printf("PASS: %s\n", name); tests_passed++; } \
} while(0)

void test_flush(void) {
    printf("\n--- test_flush_prefetch ---\n");
    uint8_t buf[256];
    memset(buf, 0x42, 256);
    arix_cache_flush(buf, 256);
    TEST("cache flush no crash", 1);
}

void test_prefetch(void) {
    uint8_t val = 42;
    arix_cache_prefetch(&val);
    TEST("cache prefetch no crash", 1);
}

void test_barrier(void) {
    arix_cache_barrier();
    TEST("cache barrier no crash", 1);
}

int main(void) {
    printf("=== ARIX-Cache Attack Mitigation Test Suite ===\n");
    test_flush();
    test_prefetch();
    test_barrier();
    printf("\nResults: %d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
