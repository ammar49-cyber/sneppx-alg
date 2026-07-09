#include "secure_memory_allocator.h"
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
    void* p = SNEPPX_secure_alloc(64);
    ASSERT(p != NULL, "secure alloc 64 bytes");
    memset(p, 0xAB, 64);
    unsigned char* buf = (unsigned char*)p;
    ASSERT(buf[0] == 0xAB, "memory writable");
    ASSERT(buf[63] == 0xAB, "last byte writable");
    SNEPPX_secure_free(p, 64);
}

static void test_secure_alloc_zero(void) {
    void* p = SNEPPX_secure_alloc(128);
    ASSERT(p != NULL, "secure alloc 128 bytes");
    unsigned char* buf = (unsigned char*)p;
    int all_zero = 1;
    for (size_t i = 0; i < 128; i++) if (buf[i] != 0) { all_zero = 0; break; }
    ASSERT(all_zero, "secure alloc zeroed");
    SNEPPX_secure_free(p, 128);
}

static void test_secure_alloc_large(void) {
    void* p = SNEPPX_secure_alloc(1048576);
    ASSERT(p != NULL, "secure alloc 1MB");
    SNEPPX_secure_free(p, 1048576);
}

int main(void) {
    run_test("secure_alloc_free", test_secure_alloc_free);
    run_test("secure_alloc_zero", test_secure_alloc_zero);
    run_test("secure_alloc_large", test_secure_alloc_large);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
