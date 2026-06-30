#include "polymorphic_memory_allocator.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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

/* ── Pool init / destroy ───────────────────────────────────── */

static void test_pool_init_destroy(void) {
    int ret = arix_mem_pool_init();
    ASSERT(ret == 0, "pool init returns 0");
    /* idempotent: calling again should succeed */
    ret = arix_mem_pool_init();
    ASSERT(ret == 0, "pool init idempotent");
    arix_mem_pool_destroy();
}

/* ── TLS cache init / destroy ──────────────────────────────── */

static void test_tls_cache_init_destroy(void) {
    arix_mem_pool_init();
    arix_tls_cache_init();
    /* second call is a no-op */
    arix_tls_cache_init();
    arix_tls_cache_destroy();
    arix_mem_pool_destroy();
}

/* ── Basic alloc / free for every size class ──────────────── */

static void test_alloc_free_each_class(void) {
    arix_mem_pool_init();
    arix_tls_cache_init();

    static const size_t sizes[] = {
        1, 16, 17, 32, 48, 64, 96, 128, 192, 256,
        384, 512, 768, 1024, 1536, 2048, 3072, 4096,
        6144, 8192
    };
    size_t nsizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t i = 0; i < nsizes; i++) {
        void* p = arix_pool_alloc(sizes[i]);
        ASSERT(p != NULL, "alloc returns non-NULL");
        memset(p, 0xAB, sizes[i]);
        arix_pool_free(p, sizes[i]);
    }

    arix_tls_cache_destroy();
    arix_mem_pool_destroy();
}

/* ── Allocated memory is zeroed ────────────────────────────── */

static void test_alloc_zeroed(void) {
    arix_mem_pool_init();
    arix_tls_cache_init();

    void* p = arix_pool_alloc(128);
    ASSERT(p != NULL, "alloc succeeds");
    unsigned char* bytes = (unsigned char*)p;
    for (int i = 0; i < 128; i++) {
        ASSERT(bytes[i] == 0, "all bytes zeroed");
    }
    arix_pool_free(p, 128);

    arix_tls_cache_destroy();
    arix_mem_pool_destroy();
}

/* ── Many iterations (stress) ──────────────────────────────── */

static void test_stress_alloc_free(void) {
    arix_mem_pool_init();
    arix_tls_cache_init();

    enum { N = 1000 };
    void** ptrs = (void**)malloc(N * sizeof(void*));
    ASSERT(ptrs != NULL, "stress malloc");

    for (int i = 0; i < N; i++) {
        size_t sz = (size_t)(1 + (rand() % 4096));
        ptrs[i] = arix_pool_alloc(sz);
        ASSERT(ptrs[i] != NULL, "stress alloc");
        memset(ptrs[i], (unsigned char)(i & 0xFF), sz);
    }
    for (int i = 0; i < N; i++) {
        arix_pool_free(ptrs[i], (size_t)(1 + (rand() % 4096)));
    }
    free(ptrs);

    arix_tls_cache_destroy();
    arix_mem_pool_destroy();
}

/* ── Size class behavior tested through public API ─────────── */

static void test_size_class_behavior(void) {
    /* Allocate at each class boundary and verify success */
    static const size_t sizes[] = {1, 16, 32, 48, 64, 128, 256, 512,
                                   1024, 2048, 4096, 8192, 8193};
    for (size_t i = 0; i < 13; i++) {
        void* p = arix_pool_alloc(sizes[i]);
        ASSERT(p != NULL, "alloc at class boundary");
        arix_pool_free(p, sizes[i]);
    }
}

/* ── Large alloc falls back to arix_malloc ─────────────────── */

static void test_large_alloc_fallback(void) {
    arix_mem_pool_init();
    arix_tls_cache_init();

    /* 1 MiB > pool max → should use arix_malloc under the hood */
    void* p = arix_pool_alloc(1024 * 1024);
    ASSERT(p != NULL, "large alloc succeeds");
    memset(p, 0xFF, 1024 * 1024);
    arix_pool_free(p, 1024 * 1024);

    arix_tls_cache_destroy();
    arix_mem_pool_destroy();
}

/* ── Null safety for pool_free ─────────────────────────────── */

static void test_null_free(void) {
    arix_mem_pool_init();
    arix_pool_free(NULL, 0);   /* must not crash */
    arix_pool_free(NULL, 128); /* must not crash */
    arix_mem_pool_destroy();
}

/* ── Stats sanity ──────────────────────────────────────────── */

static void test_stats_sanity(void) {
    arix_mem_pool_init();
    arix_tls_cache_init();

    ArixMemStats s;
    arix_mem_pool_stats(&s);
    ASSERT(s.total_chunks >= 0, "chunks >= 0");
    ASSERT(s.active_tls_caches >= 1, "at least one TLS cache");

    void* p = arix_pool_alloc(256);
    ASSERT(p != NULL, "alloc for stats");
    arix_pool_free(p, 256);

    arix_mem_pool_stats(&s);
    ASSERT(s.total_pool_allocated >= 256, "allocated bytes tracked");

    arix_mem_pool_print_stats();

    arix_tls_cache_destroy();
    arix_mem_pool_destroy();
}

/* ── Multiple pool allocs without TLS ──────────────────────── */

static void test_alloc_without_tls(void) {
    arix_mem_pool_init();
    /* Do NOT init TLS cache — should still work via global stack */

    void* p1 = arix_pool_alloc(128);
    ASSERT(p1 != NULL, "alloc without TLS works");
    void* p2 = arix_pool_alloc(128);
    ASSERT(p2 != NULL, "second alloc works");
    ASSERT(p1 != p2, "different pointers");

    arix_pool_free(p1, 128);
    arix_pool_free(p2, 128);

    arix_mem_pool_destroy();
}

/* ── Sequential destroy and re-init ────────────────────────── */

static void test_reinit(void) {
    for (int round = 0; round < 5; round++) {
        int ret = arix_mem_pool_init();
        ASSERT(ret == 0, "re-init succeeds");
        arix_tls_cache_init();
        void* p = arix_pool_alloc(512);
        ASSERT(p != NULL, "re-init alloc");
        arix_pool_free(p, 512);
        arix_tls_cache_destroy();
        arix_mem_pool_destroy();
    }
}

/* ── Mix of core and pool allocators ───────────────────────── */

static void test_mixed_allocators(void) {
    arix_mem_pool_init();

    void* a = arix_malloc(256, 64);
    ASSERT(a != NULL, "core malloc");
    void* b = arix_pool_alloc(256);
    ASSERT(b != NULL, "pool alloc");
    void* c = arix_malloc(4096, 16);
    ASSERT(c != NULL, "core malloc large");

    arix_free(a, 256);
    arix_pool_free(b, 256);
    arix_free(c, 4096);

    arix_mem_pool_destroy();
}

/* ───────────────────────────────────────────────────────────── */

int main(void) {
    run_test("pool_init_destroy",           test_pool_init_destroy);
    run_test("tls_cache_init_destroy",      test_tls_cache_init_destroy);
    run_test("alloc_free_each_class",       test_alloc_free_each_class);
    run_test("alloc_zeroed",                test_alloc_zeroed);
    run_test("stress_alloc_free",           test_stress_alloc_free);
    run_test("size_class_behavior",         test_size_class_behavior);
    run_test("large_alloc_fallback",        test_large_alloc_fallback);
    run_test("null_free",                   test_null_free);
    run_test("stats_sanity",                test_stats_sanity);
    run_test("alloc_without_tls",           test_alloc_without_tls);
    run_test("reinit",                      test_reinit);
    run_test("mixed_allocators",            test_mixed_allocators);

    printf("\n%d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
