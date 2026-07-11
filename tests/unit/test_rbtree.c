#include "rbtree.h"
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

static void test_rbtree_create_destroy(void) {
    SNEPPXRBTree* tree = SNEPPX_rbtree_create();
    ASSERT(tree != NULL, "rbtree created");
    ASSERT(tree->size == 0, "empty tree");
    SNEPPX_rbtree_destroy(tree);
}

static void test_rbtree_insert_search(void) {
    SNEPPXRBTree* tree = SNEPPX_rbtree_create();
    SNEPPX_rbtree_insert(tree, 10, (void*)100);
    SNEPPX_rbtree_insert(tree, 5, (void*)50);
    SNEPPX_rbtree_insert(tree, 15, (void*)150);
    ASSERT(tree->size == 3, "three nodes");

    void* val = SNEPPX_rbtree_search(tree, 10);
    ASSERT(val == (void*)100, "search key 10");
    val = SNEPPX_rbtree_search(tree, 5);
    ASSERT(val == (void*)50, "search key 5");
    val = SNEPPX_rbtree_search(tree, 99);
    ASSERT(val == NULL, "search missing key");
    SNEPPX_rbtree_destroy(tree);
}

static void test_rbtree_delete(void) {
    SNEPPXRBTree* tree = SNEPPX_rbtree_create();
    SNEPPX_rbtree_insert(tree, 20, (void*)200);
    SNEPPX_rbtree_insert(tree, 10, (void*)100);
    SNEPPX_rbtree_insert(tree, 30, (void*)300);
    SNEPPX_rbtree_delete(tree, 10);
    ASSERT(tree->size == 2, "size 2 after delete");
    void* val = SNEPPX_rbtree_search(tree, 10);
    ASSERT(val == NULL, "deleted key absent");
    SNEPPX_rbtree_destroy(tree);
}

static void foreach_collect(void* ctx) {
    // We can't use this easily with the current API, skip inorder test
    (void)ctx;
}

static void test_rbtree_min_max(void) {
    SNEPPXRBTree* tree = SNEPPX_rbtree_create();
    SNEPPX_rbtree_insert(tree, 20, (void*)200);
    SNEPPX_rbtree_insert(tree, 10, (void*)100);
    SNEPPX_rbtree_insert(tree, 30, (void*)300);
    
    uint64_t min = SNEPPX_rbtree_min(tree);
    ASSERT(min == 10, "min is 10");
    
    uint64_t max = SNEPPX_rbtree_max(tree);
    ASSERT(max == 30, "max is 30");
    
    SNEPPX_rbtree_destroy(tree);
}

static void test_rbtree_successor_predecessor(void) {
    SNEPPXRBTree* tree = SNEPPX_rbtree_create();
    SNEPPX_rbtree_insert(tree, 20, (void*)200);
    SNEPPX_rbtree_insert(tree, 10, (void*)100);
    SNEPPX_rbtree_insert(tree, 30, (void*)300);
    SNEPPX_rbtree_insert(tree, 15, (void*)150);
    
    uint64_t succ = SNEPPX_rbtree_successor(tree, 15);
    ASSERT(succ == 20, "successor of 15 is 20");
    
    uint64_t pred = SNEPPX_rbtree_predecessor(tree, 15);
    ASSERT(pred == 10, "predecessor of 15 is 10");
    
    SNEPPX_rbtree_destroy(tree);
}

int main(void) {
    run_test("rbtree_create_destroy", test_rbtree_create_destroy);
    run_test("rbtree_insert_search", test_rbtree_insert_search);
    run_test("rbtree_delete", test_rbtree_delete);
    run_test("rbtree_min_max", test_rbtree_min_max);
    run_test("rbtree_successor_predecessor", test_rbtree_successor_predecessor);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}