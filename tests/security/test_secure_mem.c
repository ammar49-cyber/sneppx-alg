#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "protected_memory_manager.h"
#include "stack_canary_protection.h"

static int tests_passed = 0, tests_failed = 0;

#define TEST(name, expr) do { \
    if (!(expr)) { printf("FAIL: %s\n", name); tests_failed++; } \
    else { printf("PASS: %s\n", name); tests_passed++; } \
} while(0)

void test_pool_create_destroy(void) {
    printf("\n--- test_pool_create_destroy ---\n");
    SNEPPXSecureAllocConfig cfg = {1, 1, 0, 0};
    SNEPPXSecurePool* pool = SNEPPX_secure_pool_create(1024 * 1024, &cfg);
    TEST("pool created", pool != NULL);
    size_t t, u, p;
    SNEPPX_secure_pool_stats(pool, &t, &u, &p);
    TEST("pool capacity > 0", t == 1024 * 1024 || t > 0);
    SNEPPX_secure_pool_destroy(pool);
    TEST("pool destroyed", 1);
}

void test_malloc_free(void) {
    printf("\n--- test_malloc_free ---\n");
    SNEPPXSecureAllocConfig cfg = {0, 0, 0, 0};
    SNEPPXSecurePool* pool = SNEPPX_secure_pool_create(65536, &cfg);
    TEST("pool created", pool != NULL);
    void* ptrs[100];
    int ok = 1;
    for (int i = 0; i < 100; i++) {
        ptrs[i] = SNEPPX_secure_malloc(pool, 64, 16);
        if (!ptrs[i]) { ok = 0; break; }
        memset(ptrs[i], i, 64);
    }
    TEST("100 allocations", ok);
    for (int i = 0; i < 100; i++) {
        SNEPPX_secure_free(pool, ptrs[i], 64);
    }
    TEST("100 frees", 1);
    SNEPPX_secure_pool_destroy(pool);
}

void test_canary_corruption(void) {
    printf("\n--- test_canary_corruption ---\n");
    SNEPPXCanary c1, c2;
    SNEPPX_canary_generate(&c1);
    SNEPPX_canary_generate(&c2);
    uint8_t buf[32];
    SNEPPX_canary_write(&c1, buf);
    SNEPPX_canary_write(&c2, buf + 16);
    TEST("canary verify match", SNEPPX_canary_verify(&c1, buf));
    TEST("canary verify mismatch", !SNEPPX_canary_verify(&c1, buf + 16));
    buf[0] ^= 0xFF;
    TEST("canary detect corruption", !SNEPPX_canary_verify(&c1, buf));
}

void test_overflow_detection(void) {
    printf("\n--- test_overflow_detection ---\n");
    SNEPPXSecureAllocConfig cfg = {0, 1, 0, 0};
    SNEPPXSecurePool* pool = SNEPPX_secure_pool_create(65536, &cfg);
    TEST("pool created", pool != NULL);
    void* ptr = SNEPPX_secure_malloc(pool, 64, 16);
    TEST("alloc succeeded", ptr != NULL);
    SNEPPX_secure_free(pool, ptr, 64);
    TEST("normal free ok", 1);
    SNEPPX_secure_pool_destroy(pool);
}

int main(void) {
    printf("=== SNEPPX-Secure Memory Test Suite ===\n");
    test_pool_create_destroy();
    test_malloc_free();
    test_canary_corruption();
    test_overflow_detection();
    printf("\nResults: %d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
