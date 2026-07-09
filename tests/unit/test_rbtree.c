#include "polymorphic_memory_allocator.h"
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
    ASSERT(SNEPPX_rbtree_size(tree) == 0, "empty tree");
    SNEPPX_rbtree_destroy(tree);
}

static void test_rbtree_insert_search(void) {
    SNEPPXRBTree* tree = SNEPPX_rbtree_create();
    SNEPPX_rbtree_insert(tree, 10, (void*)100);
    SNEPPX_rbtree_insert(tree, 5, (void*)50);
    SNEPPX_rbtree_insert(tree, 15, (void*)150);
    ASSERT(SNEPPX_rbtree_size(tree) == 3, "three nodes");

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
    ASSERT(SNEPPX_rbtree_size(tree) == 2, "size 2 after delete");
    void* val = SNEPPX_rbtree_search(tree, 10);
    ASSERT(val == NULL, "deleted key absent");
    SNEPPX_rbtree_destroy(tree);
}

static void test_rbtree_inorder(void) {
    SNEPPXRBTree* tree = SNEPPX_rbtree_create();
    SNEPPX_rbtree_insert(tree, 3, (void*)3);
    SNEPPX_rbtree_insert(tree, 1, (void*)1);
    SNEPPX_rbtree_insert(tree, 2, (void*)2);

    void* vals[3];
    SNEPPX_rbtree_inorder(tree, vals);
    ASSERT(vals[0] == (void*)1, "inorder first");
    ASSERT(vals[1] == (void*)2, "inorder second");
    ASSERT(vals[2] == (void*)3, "inorder third");
    SNEPPX_rbtree_destroy(tree);
}

int main(void) {
    run_test("rbtree_create_destroy", test_rbtree_create_destroy);
    run_test("rbtree_insert_search", test_rbtree_insert_search);
    run_test("rbtree_delete", test_rbtree_delete);
    run_test("rbtree_inorder", test_rbtree_inorder);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
