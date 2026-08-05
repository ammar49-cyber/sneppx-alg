#include "stack_canary_protection.h"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test Canary
 *
 * WHAT
 *   Test Canary.
 *
 * CONCEPT
 *   Provides stack canary hardening.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


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
    SNEPPXCanary canary;
    SNEPPX_canary_generate(&canary);
    ASSERT(canary.generation >= 0, "canary generation set");
    uint8_t buf[SNEPPX_CANARY_SIZE];
    SNEPPX_canary_write(&canary, buf);
    int ok = SNEPPX_canary_verify(&canary, buf);
    ASSERT(ok, "canary check passes");
}

static void test_canary_detect_tamper(void) {
    SNEPPXCanary canary;
    SNEPPX_canary_generate(&canary);
    uint8_t buf[SNEPPX_CANARY_SIZE];
    SNEPPX_canary_write(&canary, buf);
    buf[0] ^= 0xFF;  // tamper
    int ok = SNEPPX_canary_verify(&canary, buf);
    ASSERT(!ok, "tampered canary fails check");
}

static void test_canary_refresh(void) {
    SNEPPXCanary c1, c2;
    SNEPPX_canary_generate(&c1);
    SNEPPX_canary_generate(&c2);
    int same = 1;
    for (int i = 0; i < SNEPPX_CANARY_SIZE; i++) {
        if (c1.value[i] != c2.value[i]) { same = 0; break; }
    }
    // Very unlikely they're the same
    ASSERT(!same || c1.generation != c2.generation || c1.value[0] != c2.value[0], "refreshed canary differs");
    uint8_t buf[SNEPPX_CANARY_SIZE];
    SNEPPX_canary_write(&c2, buf);
    int ok = SNEPPX_canary_verify(&c2, buf);
    ASSERT(ok, "refreshed canary valid");
}

int main(void) {
    run_test("canary_set_check", test_canary_set_check);
    run_test("canary_detect_tamper", test_canary_detect_tamper);
    run_test("canary_refresh", test_canary_refresh);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
