#include <stdio.h>
#include <string.h>
#include "arix_timing.h"

static int tests_passed = 0, tests_failed = 0;

#define TEST(name, expr) do { \
    if (!(expr)) { printf("FAIL: %s\n", name); tests_failed++; } \
    else { printf("PASS: %s\n", name); tests_passed++; } \
} while(0)

void test_timer_resolution(void) {
    printf("\n--- test_timer_resolution ---\n");
    uint64_t t0 = arix_timing_start();
    uint64_t t1 = arix_timing_end();
    TEST("timer produces value", t1 != t0 || t1 != 0);
    TEST("timer monotonic", t1 >= t0);
}

void test_random_delay(void) {
    printf("\n--- test_random_delay ---\n");
    uint64_t t0 = arix_timing_start();
    arix_timing_random_delay(100, 5000);
    uint64_t t1 = arix_timing_end();
    TEST("delay completed", t1 >= t0);
}

void test_safe_equal(void) {
    printf("\n--- test_safe_equal ---\n");
    uint8_t a[32] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
                     17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32};
    uint8_t b[32], c[32];
    memcpy(b, a, 32);
    memset(c, 0xFF, 32);
    uint64_t timing;
    int r1 = arix_timing_safe_equal(a, b, 32, &timing);
    TEST("safe equal same", r1);
    int r2 = arix_timing_safe_equal(a, c, 32, &timing);
    TEST("safe equal diff", !r2);
}

int main(void) {
    printf("=== ARIX-Timing Attack Prevention Test Suite ===\n");
    test_timer_resolution();
    test_random_delay();
    test_safe_equal();
    printf("\nResults: %d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
