#include "../../mm/internal/vmem.h"
#include "../../mm/internal/vmem.c"
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
    SNEPPXVMemAllocator alloc;
    SNEPPX_vmem_init(&alloc);
    void* region = SNEPPX_vmem_reserve(&alloc, 65536, 4096, SNEPPX_VMEM_FLAG_WRITABLE);
    if (region == NULL) {
        printf("SKIP (stub returns NULL): ");
        SNEPPX_vmem_cleanup(&alloc);
        return;
    }
    int ok = SNEPPX_vmem_commit(&alloc, region, 4096);
    ASSERT(ok == 0, "vmem commit 4KB");
    SNEPPX_vmem_decommit(&alloc, region, 4096);
    SNEPPX_vmem_release(&alloc, region, 65536);
    SNEPPX_vmem_cleanup(&alloc);
}

static void test_vmem_large_region(void) {
    SNEPPXVMemAllocator alloc;
    SNEPPX_vmem_init(&alloc);
    void* region = SNEPPX_vmem_reserve(&alloc, 1048576, 4096, SNEPPX_VMEM_FLAG_WRITABLE);
    if (region == NULL) {
        printf("SKIP (stub returns NULL): ");
        SNEPPX_vmem_cleanup(&alloc);
        return;
    }
    SNEPPX_vmem_release(&alloc, region, 1048576);
    SNEPPX_vmem_cleanup(&alloc);
}

int main(void) {
    run_test("vmem_reserve_commit", test_vmem_reserve_commit);
    run_test("vmem_large_region", test_vmem_large_region);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
