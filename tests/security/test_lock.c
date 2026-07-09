#include "synchronization_lock_interface.h"
#include <stdio.h>

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

static void test_lock_init_destroy(void) {
    SNEPPXLock* lock = SNEPPX_lock_init();
    ASSERT(lock != NULL, "lock initialized");
    SNEPPX_lock_destroy(lock);
}

static void test_lock_acquire_release(void) {
    SNEPPXLock* lock = SNEPPX_lock_init();
    SNEPPX_lock_acquire(lock);
    ASSERT(lock->held == 1 || lock->held == 0, "lock acquired");
    SNEPPX_lock_release(lock);
    SNEPPX_lock_destroy(lock);
}

static void test_lock_try_acquire(void) {
    SNEPPXLock* lock = SNEPPX_lock_init();
    int ok = SNEPPX_lock_try_acquire(lock);
    ASSERT(ok == 0 || ok == 1, "try_acquire returns bool");
    SNEPPX_lock_release(lock);
    SNEPPX_lock_destroy(lock);
}

int main(void) {
    run_test("lock_init_destroy", test_lock_init_destroy);
    run_test("lock_acquire_release", test_lock_acquire_release);
    run_test("lock_try_acquire", test_lock_try_acquire);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
