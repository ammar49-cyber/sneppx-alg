#include "system_architecture_definitions.h"
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

static void test_arch_has_avx(void) {
    int has = SNEPPX_arch_has_avx();
    ASSERT(has == 0 || has == 1, "has_avx returns boolean");
}

static void test_arch_has_avx2(void) {
    int has = SNEPPX_arch_has_avx2();
    ASSERT(has == 0 || has == 1, "has_avx2 returns boolean");
}

static void test_arch_has_neon(void) {
    int has = SNEPPX_arch_has_neon();
    ASSERT(has == 0 || has == 1, "has_neon returns boolean");
}

static void test_arch_num_cores(void) {
    int cores = SNEPPX_arch_num_cores();
    ASSERT(cores > 0, "num_cores > 0");
}

static void test_arch_cache_line_size(void) {
    int sz = SNEPPX_arch_cache_line_size();
    ASSERT(sz > 0, "cache line size > 0");
}

int main(void) {
    run_test("arch_has_avx", test_arch_has_avx);
    run_test("arch_has_avx2", test_arch_has_avx2);
    run_test("arch_has_neon", test_arch_has_neon);
    run_test("arch_num_cores", test_arch_num_cores);
    run_test("arch_cache_line_size", test_arch_cache_line_size);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
