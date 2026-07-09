#include "stack_canary_protection.h"
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

static void test_canary_set_check(void) {
    uint64_t canary = SNEPPX_canary_set();
    ASSERT(canary != 0, "canary value non-zero");
    int ok = SNEPPX_canary_check(canary);
    ASSERT(ok, "canary check passes");
}

static void test_canary_detect_tamper(void) {
    uint64_t canary = SNEPPX_canary_set();
    canary ^= 0xFF;
    int ok = SNEPPX_canary_check(canary);
    ASSERT(!ok, "tampered canary fails check");
}

static void test_canary_refresh(void) {
    uint64_t c1 = SNEPPX_canary_set();
    uint64_t c2 = SNEPPX_canary_refresh();
    ASSERT(c2 != c1, "refreshed canary differs");
    int ok = SNEPPX_canary_check(c2);
    ASSERT(ok, "refreshed canary valid");
}

int main(void) {
    run_test("canary_set_check", test_canary_set_check);
    run_test("canary_detect_tamper", test_canary_detect_tamper);
    run_test("canary_refresh", test_canary_refresh);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
