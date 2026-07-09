#include "hierarchical_state_space.h"
#include <stdio.h>
#include <math.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s (%s)\n", msg, #cond); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, eps, msg) do { \
    if (fabsf((a) - (b)) > (eps)) { \
        printf("FAIL: %s (got %f, expected %f)\n", msg, (float)(a), (float)(b)); \
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

static void test_discretize_zoh(void) {
    float A[] = {-0.5f, 0.0f, 0.0f, -0.5f};
    float B[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float dt = 0.1f;
    float Ad[4], Bd[4];
    SNEPPX_hss_discretize_zoh(A, B, 2, dt, Ad, Bd);
    ASSERT(Ad[0] > 0.9f, "Ad[0] positive");
    ASSERT(Ad[3] > 0.9f, "Ad[3] positive");
    ASSERT_NEAR(Ad[1], 0.0f, 1e-6f, "Ad[1] zero");
    ASSERT_NEAR(Ad[2], 0.0f, 1e-6f, "Ad[2] zero");
}

static void test_discretize_bilinear(void) {
    float A[] = {-1.0f};
    float B[] = {1.0f};
    float dt = 0.1f;
    float Ad[1], Bd[1];
    SNEPPX_hss_discretize_bilinear(A, B, 1, dt, Ad, Bd);
    ASSERT(Ad[0] > 0.8f, "Ad bilinear positive");
    ASSERT(Bd[0] > 0.0f, "Bd bilinear positive");
}

static void test_discretize_identity(void) {
    float A[] = {0.0f};
    float B[] = {1.0f};
    float dt = 1.0f;
    float Ad[1], Bd[1];
    SNEPPX_hss_discretize_zoh(A, B, 1, dt, Ad, Bd);
    ASSERT_NEAR(Ad[0], 1.0f, 1e-4f, "Ad identity");
    ASSERT_NEAR(Bd[0], 1.0f, 1e-4f, "Bd identity");
}

int main(void) {
    run_test("discretize_zoh", test_discretize_zoh);
    run_test("discretize_bilinear", test_discretize_bilinear);
    run_test("discretize_identity", test_discretize_identity);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
