#include "pqueue.h"
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

static void test_pqueue_create_destroy(void) {
    SNEPPXPriorityQueue* pq = SNEPPX_pq_create(10);
    ASSERT(pq != NULL, "pqueue created");
    ASSERT(SNEPPX_pq_size(pq) == 0, "empty pqueue");
    SNEPPX_pq_destroy(pq);
}

static void test_pqueue_push_pop(void) {
    SNEPPXPriorityQueue* pq = SNEPPX_pq_create(10);
    SNEPPX_pq_push(pq, 3, (void*)3);
    SNEPPX_pq_push(pq, 1, (void*)1);
    SNEPPX_pq_push(pq, 2, (void*)2);
    ASSERT(SNEPPX_pq_size(pq) == 3, "three items");

    uint64_t priority;
    void* val;
    int ret = SNEPPX_pq_pop(pq, &priority, &val);
    ASSERT(ret == 0, "pop returns 0");
    ASSERT(priority == 1 && val == (void*)1, "pop lowest priority");
    ret = SNEPPX_pq_pop(pq, &priority, &val);
    ASSERT(ret == 0 && priority == 2 && val == (void*)2, "pop second lowest");
    ret = SNEPPX_pq_pop(pq, &priority, &val);
    ASSERT(ret == 0 && priority == 3 && val == (void*)3, "pop highest priority");
    ASSERT(SNEPPX_pq_size(pq) == 0, "empty after pops");
    SNEPPX_pq_destroy(pq);
}

static void test_pqueue_peek(void) {
    SNEPPXPriorityQueue* pq = SNEPPX_pq_create(10);
    SNEPPX_pq_push(pq, 5, (void*)42);
    uint64_t priority;
    void* val;
    int ret = SNEPPX_pq_peek(pq, &priority, &val);
    ASSERT(ret == 0, "peek returns 0");
    ASSERT(priority == 5 && val == (void*)42, "peek returns top");
    ASSERT(SNEPPX_pq_size(pq) == 1, "size unchanged after peek");
    SNEPPX_pq_destroy(pq);
}

int main(void) {
    run_test("pqueue_create_destroy", test_pqueue_create_destroy);
    run_test("pqueue_push_pop", test_pqueue_push_pop);
    run_test("pqueue_peek", test_pqueue_peek);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}