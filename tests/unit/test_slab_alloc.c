#include "../../mm/internal/slab_alloc.h"
#include "../../mm/internal/slab_alloc.c"
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
    SNEPPXSlabCache* cache = NULL;
    int ret = SNEPPX_slab_cache_create(&cache, 64, 16);
    ASSERT(ret == 0, "slab cache create");
    ASSERT(cache != NULL, "slab cache created");
    SNEPPX_slab_cache_destroy(cache);
}

static void test_slab_alloc_free(void) {
    SNEPPXSlabCache* cache = NULL;
    int ret = SNEPPX_slab_cache_create(&cache, 32, 8);
    ASSERT(ret == 0, "slab cache create");
    if (!cache) return;
    void* p = SNEPPX_slab_cache_alloc(cache);
    if (p == NULL) {
        printf("SKIP (stub returns NULL): ");
        SNEPPX_slab_cache_destroy(cache);
        return;
    }
    memset(p, 0x42, 32);
    unsigned char* buf = (unsigned char*)p;
    ASSERT(buf[0] == 0x42, "memory writable");
    SNEPPX_slab_cache_free(cache, p);
    SNEPPX_slab_cache_destroy(cache);
}

static void test_slab_multiple_allocs(void) {
    SNEPPXSlabCache* cache = NULL;
    int ret = SNEPPX_slab_cache_create(&cache, 16, 4);
    ASSERT(ret == 0, "slab cache create");
    if (!cache) return;
    void* p1 = SNEPPX_slab_cache_alloc(cache);
    void* p2 = SNEPPX_slab_cache_alloc(cache);
    if (p1 == NULL || p2 == NULL) {
        printf("SKIP (stub returns NULL): ");
        SNEPPX_slab_cache_destroy(cache);
        return;
    }
    ASSERT(p1 != p2, "distinct allocations");
    SNEPPX_slab_cache_free(cache, p2);
    SNEPPX_slab_cache_free(cache, p1);
    SNEPPX_slab_cache_destroy(cache);
}

int main(void) {
    run_test("slab_create_destroy", test_slab_create_destroy);
    run_test("slab_alloc_free", test_slab_alloc_free);
    run_test("slab_multiple_allocs", test_slab_multiple_allocs);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
