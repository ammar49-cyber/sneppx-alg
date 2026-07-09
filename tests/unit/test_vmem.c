#include "memory_management.h"
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

static void test_vmem_reserve_commit(void) {
    void* region = SNEPPX_vmem_reserve(65536);
    ASSERT(region != NULL, "vmem reserve 64KB");
    int ok = SNEPPX_vmem_commit(region, 4096);
    ASSERT(ok == 0, "vmem commit 4KB");
    SNEPPX_vmem_decommit(region, 4096);
    SNEPPX_vmem_release(region, 65536);
}

static void test_vmem_large_region(void) {
    void* region = SNEPPX_vmem_reserve(1048576);
    ASSERT(region != NULL, "vmem reserve 1MB");
    SNEPPX_vmem_release(region, 1048576);
}

int main(void) {
    run_test("vmem_reserve_commit", test_vmem_reserve_commit);
    run_test("vmem_large_region", test_vmem_large_region);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
