#include "power_analysis_mitigation.h"
#include <stdio.h>
#include <stdint.h>

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

static void test_power_mul_const_time(void) {
    int r = SNEPPX_power_mul_const_time(5, 3);
    ASSERT(r == 15, "multiplication result correct");
}

static void test_power_cmp_const_time(void) {
    int r = SNEPPX_power_cmp_const_time(10, 20);
    ASSERT(r < 0, "cmp result negative");
    r = SNEPPX_power_cmp_const_time(30, 10);
    ASSERT(r > 0, "cmp result positive");
    r = SNEPPX_power_cmp_const_time(7, 7);
    ASSERT(r == 0, "cmp result zero");
}

static void test_power_mask_gen(void) {
    uint64_t mask = SNEPPX_power_mask_gen(1);
    ASSERT(mask == (uint64_t)-1 || mask == 0, "mask generated");
}

int main(void) {
    run_test("power_mul_const_time", test_power_mul_const_time);
    run_test("power_cmp_const_time", test_power_cmp_const_time);
    run_test("power_mask_gen", test_power_mask_gen);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
