#include <stdio.h>
#include <string.h>
#include "constant_time_operations.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name, expr) do { \
    if (!(expr)) { printf("FAIL: %s\n", name); tests_failed++; } \
    else { printf("PASS: %s\n", name); tests_passed++; } \
} while(0)

#define NTIMINGS 1000

void test_equal(void) {
    printf("\n--- test_equal ---\n");
    uint8_t a[32], b[32];
    memset(a, 0x42, 32);
    memset(b, 0x42, 32);
    TEST("equal arrays return 1", SNEPPX_ct_equal(a, b, 32) == 1);

    b[15] ^= 0x01;
    TEST("different arrays return 0", SNEPPX_ct_equal(a, b, 32) == 0);

    memset(b, 0x42, 32);
    b[0] ^= 0x01;
    TEST("different first byte", SNEPPX_ct_equal(a, b, 32) == 0);

    memset(b, 0x42, 32);
    b[31] ^= 0x01;
    TEST("different last byte", SNEPPX_ct_equal(a, b, 32) == 0);
}

void test_select(void) {
    printf("\n--- test_select ---\n");
    TEST("select with 0xFF returns a", SNEPPX_ct_select(0xFF, 0xAB, 0xCD) == 0xAB);
    TEST("select with 0x00 returns b", SNEPPX_ct_select(0x00, 0xAB, 0xCD) == 0xCD);
    TEST("select with 0x42 returns mixed", SNEPPX_ct_select(0x42, 0xFF, 0x00) == 0x42);
}

void test_copy(void) {
    printf("\n--- test_copy ---\n");
    uint8_t dst[16], src[16];
    memset(dst, 0x00, 16);
    memset(src, 0xFF, 16);
    SNEPPX_ct_copy(dst, src, 16, 0xFF);
    int all_ff = 1;
    for (int i = 0; i < 16; i++) if (dst[i] != 0xFF) all_ff = 0;
    TEST("copy with 0xFF", all_ff == 1);

    memset(dst, 0x00, 16);
    SNEPPX_ct_copy(dst, src, 16, 0x00);
    int all_zero = 1;
    for (int i = 0; i < 16; i++) if (dst[i] != 0x00) all_zero = 0;
    TEST("copy with 0x00 no-op", all_zero == 1);
}

void test_is_zero(void) {
    printf("\n--- test_is_zero ---\n");
    uint8_t z[32] = {0};
    uint8_t nz[32];
    memset(nz, 0x42, 32);
    TEST("zero array", SNEPPX_ct_is_zero(z, 32) == 1);
    TEST("non-zero array", SNEPPX_ct_is_zero(nz, 32) == 0);

    nz[0] = 0;
    TEST("non-zero with first zero", SNEPPX_ct_is_zero(nz, 32) == 0);
}

void test_compare_32(void) {
    printf("\n--- test_compare_32 ---\n");
    uint32_t a[8], b[8];
    memset(a, 0, sizeof(a));
    memset(b, 0, sizeof(b));
    TEST("equal 32-bit arrays", SNEPPX_ct_compare_32(a, b, 8) == 0);
    b[7] = 1;
    TEST("different 32-bit arrays", SNEPPX_ct_compare_32(a, b, 8) != 0);
}

int main(void) {
    printf("=== SNEPPX-Constant-Time Test Suite ===\n");
    test_equal();
    test_select();
    test_copy();
    test_is_zero();
    test_compare_32();
    printf("\nResults: %d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
