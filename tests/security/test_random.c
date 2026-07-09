#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "cryptographic_random_generator.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name, expr) do { \
    if (!(expr)) { printf("FAIL: %s\n", name); tests_failed++; } \
    else { printf("PASS: %s\n", name); tests_passed++; } \
} while(0)

void test_uniformity(void) {
    printf("\n--- test_uniformity ---\n");
    size_t len = 1024 * 1024;
    uint8_t* buf = (uint8_t*)malloc(len);
    if (!buf) { printf("SKIP: uniformity (malloc failed)\n"); return; }
    int ret = SNEPPX_random_bytes(buf, len);
    TEST("generate 1MB random", ret == 0);
    size_t zeros = 0, ones = 0;
    for (size_t i = 0; i < len; i++) {
        for (int b = 0; b < 8; b++) {
            if ((buf[i] >> b) & 1) ones++; else zeros++;
        }
    }
    double ratio = (double)ones / (double)(ones + zeros);
    TEST("bit ratio near 0.5", ratio > 0.49 && ratio < 0.51);
    free(buf);
}

void test_no_duplicates(void) {
    printf("\n--- test_no_duplicates ---\n");
    uint32_t vals[1000];
    for (int i = 0; i < 1000; i++) vals[i] = SNEPPX_random_uint32();
    int dup = 0;
    for (int i = 0; i < 1000 && !dup; i++)
        for (int j = i + 1; j < 1000 && !dup; j++)
            if (vals[i] == vals[j]) dup = 1;
    TEST("no duplicates in 1000 values", !dup);
}

void test_uniform_bounded(void) {
    printf("\n--- test_uniform_bounded ---\n");
    uint32_t bound = 100;
    int ok = 1;
    for (int i = 0; i < 10000; i++) {
        uint32_t v = SNEPPX_random_uniform(bound);
        if (v >= bound) { ok = 0; break; }
    }
    TEST("all values in [0, bound)", ok == 1);
}

int main(void) {
    printf("=== SNEPPX-Random Test Suite ===\n");
    test_uniformity();
    test_no_duplicates();
    test_uniform_bounded();
    printf("\nResults: %d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
