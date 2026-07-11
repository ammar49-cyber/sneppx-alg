#include "hashtable.h"
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

static void test_hashtable_create_destroy(void) {
    SNEPPXHashTable* ht = SNEPPX_ht_create(64);
    ASSERT(ht != NULL, "hashtable created");
    ASSERT(ht->count == 0, "empty hashtable");
    SNEPPX_ht_destroy(ht);
}

static void test_hashtable_insert_lookup(void) {
    SNEPPXHashTable* ht = SNEPPX_ht_create(64);
    SNEPPX_ht_insert(ht, 1, (void*)42);
    SNEPPX_ht_insert(ht, 2, (void*)84);
    ASSERT(ht->count == 2, "two entries inserted");
    void* val = SNEPPX_ht_lookup(ht, 1);
    ASSERT(val == (void*)42, "key1 lookup");
    val = SNEPPX_ht_lookup(ht, 2);
    ASSERT(val == (void*)84, "key2 lookup");
    val = SNEPPX_ht_lookup(ht, 999);
    ASSERT(val == NULL, "missing key returns NULL");
    SNEPPX_ht_destroy(ht);
}

static void test_hashtable_remove(void) {
    SNEPPXHashTable* ht = SNEPPX_ht_create(64);
    SNEPPX_ht_insert(ht, 1, (void*)1);
    SNEPPX_ht_insert(ht, 2, (void*)2);
    SNEPPX_ht_delete(ht, 1);
    ASSERT(ht->count == 1, "one entry after remove");
    void* val = SNEPPX_ht_lookup(ht, 1);
    ASSERT(val == NULL, "removed key absent");
    SNEPPX_ht_destroy(ht);
}

static void test_hashtable_clear(void) {
    SNEPPXHashTable* ht = SNEPPX_ht_create(64);
    for (uint64_t i = 0; i < 10; i++) {
        SNEPPX_ht_insert(ht, i, (void*)(intptr_t)i);
    }
    ASSERT(ht->count == 10, "10 entries");
    SNEPPX_ht_clear(ht);
    ASSERT(ht->count == 0, "cleared");
    SNEPPX_ht_destroy(ht);
}

int main(void) {
    run_test("ht_create_destroy", test_hashtable_create_destroy);
    run_test("ht_insert_lookup", test_hashtable_insert_lookup);
    run_test("ht_remove", test_hashtable_remove);
    run_test("ht_clear", test_hashtable_clear);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}