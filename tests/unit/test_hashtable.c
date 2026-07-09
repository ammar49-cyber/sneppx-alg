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
    SNEPPXHashTable* ht = SNEPPX_hashtable_create(64);
    ASSERT(ht != NULL, "hashtable created");
    ASSERT(ht->size == 0, "empty hashtable");
    SNEPPX_hashtable_destroy(ht);
}

static void test_hashtable_insert_lookup(void) {
    SNEPPXHashTable* ht = SNEPPX_hashtable_create(64);
    SNEPPX_hashtable_insert(ht, "key1", (void*)42);
    SNEPPX_hashtable_insert(ht, "key2", (void*)84);
    ASSERT(ht->size == 2, "two entries inserted");
    void* val = SNEPPX_hashtable_lookup(ht, "key1");
    ASSERT(val == (void*)42, "key1 lookup");
    val = SNEPPX_hashtable_lookup(ht, "key2");
    ASSERT(val == (void*)84, "key2 lookup");
    val = SNEPPX_hashtable_lookup(ht, "nonexistent");
    ASSERT(val == NULL, "missing key returns NULL");
    SNEPPX_hashtable_destroy(ht);
}

static void test_hashtable_remove(void) {
    SNEPPXHashTable* ht = SNEPPX_hashtable_create(64);
    SNEPPX_hashtable_insert(ht, "a", (void*)1);
    SNEPPX_hashtable_insert(ht, "b", (void*)2);
    SNEPPX_hashtable_remove(ht, "a");
    ASSERT(ht->size == 1, "one entry after remove");
    void* val = SNEPPX_hashtable_lookup(ht, "a");
    ASSERT(val == NULL, "removed key absent");
    SNEPPX_hashtable_destroy(ht);
}

static void test_hashtable_clear(void) {
    SNEPPXHashTable* ht = SNEPPX_hashtable_create(64);
    for (int i = 0; i < 10; i++) {
        char key[8];
        sprintf(key, "k%d", i);
        SNEPPX_hashtable_insert(ht, key, (void*)(intptr_t)i);
    }
    ASSERT(ht->size == 10, "10 entries");
    SNEPPX_hashtable_clear(ht);
    ASSERT(ht->size == 0, "cleared");
    SNEPPX_hashtable_destroy(ht);
}

int main(void) {
    run_test("hashtable_create_destroy", test_hashtable_create_destroy);
    run_test("hashtable_insert_lookup", test_hashtable_insert_lookup);
    run_test("hashtable_remove", test_hashtable_remove);
    run_test("hashtable_clear", test_hashtable_clear);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
