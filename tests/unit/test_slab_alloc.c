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

static void test_slab_create_destroy(void) {
    SNEPPXSlabAllocator* slab = SNEPPX_slab_create(64, 16);
    ASSERT(slab != NULL, "slab allocator created");
    SNEPPX_slab_destroy(slab);
}

static void test_slab_alloc_free(void) {
    SNEPPXSlabAllocator* slab = SNEPPX_slab_create(32, 8);
    void* p = SNEPPX_slab_alloc(slab);
    ASSERT(p != NULL, "slab alloc returned memory");
    memset(p, 0x42, 32);
    unsigned char* buf = (unsigned char*)p;
    ASSERT(buf[0] == 0x42, "memory writable");
    SNEPPX_slab_free(slab, p);
    SNEPPX_slab_destroy(slab);
}

static void test_slab_multiple_allocs(void) {
    SNEPPXSlabAllocator* slab = SNEPPX_slab_create(16, 4);
    void* p1 = SNEPPX_slab_alloc(slab);
    void* p2 = SNEPPX_slab_alloc(slab);
    ASSERT(p1 != p2, "distinct allocations");
    SNEPPX_slab_free(slab, p2);
    SNEPPX_slab_free(slab, p1);
    SNEPPX_slab_destroy(slab);
}

int main(void) {
    run_test("slab_create_destroy", test_slab_create_destroy);
    run_test("slab_alloc_free", test_slab_alloc_free);
    run_test("slab_multiple_allocs", test_slab_multiple_allocs);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
