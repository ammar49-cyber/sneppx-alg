#include <stdio.h>
#include <string.h>
#include "side_channel_resistant_primitives.h"

static int tests_passed = 0, tests_failed = 0;

#define TEST(name, expr) do { \
    if (!(expr)) { printf("FAIL: %s\n", name); tests_failed++; } \
    else { printf("PASS: %s\n", name); tests_passed++; } \
} while(0)

void test_select_u32(void) {
    printf("\n--- test_select_u32 ---\n");
    TEST("select true", SNEPPX_sc_select_u32(0xFFFFFFFF, 0x12345678, 0x9ABCDEF0) == 0x12345678);
    TEST("select false", SNEPPX_sc_select_u32(0x00000000, 0x12345678, 0x9ABCDEF0) == 0x9ABCDEF0);
    TEST("select all bits", SNEPPX_sc_select_u32(0xFFFFFFFF, 0xFFFFFFFF, 0x00000000) == 0xFFFFFFFF);
    TEST("select none bits", SNEPPX_sc_select_u32(0x00000000, 0xFFFFFFFF, 0x00000000) == 0x00000000);
}

void test_select_u64(void) {
    printf("\n--- test_select_u64 ---\n");
    uint64_t a = 0x123456789ABCDEF0ULL;
    uint64_t b = 0xFEDCBA9876543210ULL;
    TEST("select true 64", SNEPPX_sc_select_u64(0xFFFFFFFFFFFFFFFFULL, a, b) == a);
    TEST("select false 64", SNEPPX_sc_select_u64(0x0000000000000000ULL, a, b) == b);
}

void test_equal_u32(void) {
    printf("\n--- test_equal_u32 ---\n");
    TEST("equal same", SNEPPX_sc_equal_u32(0x12345678, 0x12345678) == 0xFFFFFFFF);
    TEST("equal diff", SNEPPX_sc_equal_u32(0x12345678, 0x87654321) == 0x00000000);
    TEST("equal zero same", SNEPPX_sc_equal_u32(0, 0) == 0xFFFFFFFF);
    TEST("equal near miss", SNEPPX_sc_equal_u32(0x10000000, 0x10000001) == 0x00000000);
}

void test_lt_u32(void) {
    printf("\n--- test_lt_u32 ---\n");
    TEST("lt simple", SNEPPX_sc_lt_u32(5, 10) == 0xFFFFFFFF);
    TEST("lt not", SNEPPX_sc_lt_u32(10, 5) == 0x00000000);
    TEST("lt equal", SNEPPX_sc_lt_u32(5, 5) == 0x00000000);
    TEST("lt zero", SNEPPX_sc_lt_u32(0, 1) == 0xFFFFFFFF);
    TEST("lt max", SNEPPX_sc_lt_u32(0xFFFFFFFF, 0) == 0x00000000);
    TEST("lt max rev", SNEPPX_sc_lt_u32(0, 0xFFFFFFFF) == 0xFFFFFFFF);
}

void test_is_zero_u32(void) {
    printf("\n--- test_is_zero_u32 ---\n");
    TEST("zero is zero", SNEPPX_sc_is_zero_u32(0) == 0xFFFFFFFF);
    TEST("non-zero not zero", SNEPPX_sc_is_zero_u32(1) == 0x00000000);
    TEST("large non-zero", SNEPPX_sc_is_zero_u32(0x80000000) == 0x00000000);
}

void test_cond_copy(void) {
    printf("\n--- test_cond_copy ---\n");
    uint8_t dst[16] = {0};
    uint8_t src[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    SNEPPX_sc_cond_copy(dst, src, 16, 0xFFFFFFFF);
    int match = 1;
    for (int i = 0; i < 16; i++) if (dst[i] != src[i]) match = 0;
    TEST("copy when true", match);
    memset(dst, 0, 16);
    SNEPPX_sc_cond_copy(dst, src, 16, 0x00000000);
    match = 1;
    for (int i = 0; i < 16; i++) if (dst[i] != 0) match = 0;
    TEST("no copy when false", match);
}

int main(void) {
    printf("=== SNEPPX-Side Channel Resistant Ops Test Suite ===\n");
    test_select_u32();
    test_select_u64();
    test_equal_u32();
    test_lt_u32();
    test_is_zero_u32();
    test_cond_copy();
    printf("\nResults: %d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
