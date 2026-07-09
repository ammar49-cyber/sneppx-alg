#include "address_space_randomization.h"
#include <stdio.h>

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

static void test_aslr_alloc_free(void) {
    void* p = SNEPPX_aslr_alloc(256);
    ASSERT(p != NULL, "aslr alloc 256");
    SNEPPX_aslr_free(p, 256);
}

static void test_aslr_alloc_randomized(void) {
    void* p1 = SNEPPX_aslr_alloc(64);
    void* p2 = SNEPPX_aslr_alloc(64);
    ASSERT(p1 != NULL, "aslr alloc 1");
    ASSERT(p2 != NULL, "aslr alloc 2");
    SNEPPX_aslr_free(p1, 64);
    SNEPPX_aslr_free(p2, 64);
}

static void test_aslr_mprotect(void) {
    int ret = SNEPPX_aslr_mprotect(NULL, 0, SNEPPX_ASLR_RW);
    ASSERT(ret == 0, "mprotect stub returns 0");
}

int main(void) {
    run_test("aslr_alloc_free", test_aslr_alloc_free);
    run_test("aslr_alloc_randomized", test_aslr_alloc_randomized);
    run_test("aslr_mprotect", test_aslr_mprotect);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
