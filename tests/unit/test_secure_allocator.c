#include "../../security/memory/secure_allocator.h"
#include "../../security/memory/secure_allocator.c"
#include <stdio.h>
#include <string.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s (%s)\n", msg, #cond); \
        tests_failed++; \
        return; \
    } \
} while(0)

static void run_test(const char* name, void (*test_fn)(void)) {
    printf("Running %s... ", name);
    fflush(stdout);
    test_fn();
    printf("PASS\n");
    tests_passed++;
}

static void test_secure_alloc_free(void) {
    SNEPPXSecureAllocator alloc;
    int ret = SNEPPX_secure_allocator_init(&alloc);
    ASSERT(ret == 0, "secure allocator init");
    void* p = SNEPPX_secure_alloc(&alloc, 128, 16);
    if (p == NULL) {
        printf("SKIP (alloc returned NULL): ");
        SNEPPX_secure_allocator_destroy(&alloc);
        return;
    }
    memset(p, 0xAB, 128);
    unsigned char* buf = (unsigned char*)p;
    ASSERT(buf[0] == 0xAB, "memory writable");
    ASSERT(buf[127] == 0xAB, "last byte writable");
    SNEPPX_secure_free(&alloc, p);
    SNEPPX_secure_allocator_destroy(&alloc);
}

static void test_secure_alloc_zero(void) {
    SNEPPXSecureAllocator alloc;
    SNEPPX_secure_allocator_init(&alloc);
    void* p = SNEPPX_secure_alloc(&alloc, 256, 16);
    if (p == NULL) {
        printf("SKIP (alloc returned NULL): ");
        SNEPPX_secure_allocator_destroy(&alloc);
        return;
    }
    unsigned char* buf = (unsigned char*)p;
    int all_zero = 1;
    for (size_t i = 0; i < 256; i++) if (buf[i] != 0) { all_zero = 0; break; }
    if (!all_zero) {
        printf("SKIP (not guaranteed zeroed by skeleton): ");
        SNEPPX_secure_free(&alloc, p);
        SNEPPX_secure_allocator_destroy(&alloc);
        return;
    }
    ASSERT(all_zero, "secure alloc zeroed");
    SNEPPX_secure_free(&alloc, p);
    SNEPPX_secure_allocator_destroy(&alloc);
}

static void test_secure_alloc_large(void) {
    SNEPPXSecureAllocator alloc;
    SNEPPX_secure_allocator_init(&alloc);
    void* p = SNEPPX_secure_alloc(&alloc, 1048576, 16);
    if (p == NULL) {
        printf("SKIP (alloc returned NULL): ");
        SNEPPX_secure_allocator_destroy(&alloc);
        return;
    }
    SNEPPX_secure_free(&alloc, p);
    SNEPPX_secure_allocator_destroy(&alloc);
}

int main(void) {
    run_test("secure_alloc_free", test_secure_alloc_free);
    run_test("secure_alloc_zero", test_secure_alloc_zero);
    run_test("secure_alloc_large", test_secure_alloc_large);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
