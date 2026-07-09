#include "sha512_hashing_implementation.h"
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

static void test_sha512_hash(void) {
    unsigned char hash[64];
    const unsigned char* input = (const unsigned char*)"hello";
    SNEPPX_sha512_hash(input, 5, hash);
    int non_zero = 0;
    for (int i = 0; i < 64; i++) if (hash[i]) non_zero++;
    ASSERT(non_zero > 0, "sha512 hash produced non-zero output");
}

static void test_sha512_deterministic(void) {
    unsigned char h1[64], h2[64];
    const unsigned char* input = (const unsigned char*)"test";
    SNEPPX_sha512_hash(input, 4, h1);
    SNEPPX_sha512_hash(input, 4, h2);
    ASSERT(memcmp(h1, h2, 64) == 0, "sha512 deterministic");
}

static void test_sha512_diff_input(void) {
    unsigned char h1[64], h2[64];
    const unsigned char* in1 = (const unsigned char*)"abc";
    const unsigned char* in2 = (const unsigned char*)"abd";
    SNEPPX_sha512_hash(in1, 3, h1);
    SNEPPX_sha512_hash(in2, 3, h2);
    ASSERT(memcmp(h1, h2, 64) != 0, "sha512 different for diff input");
}

static void test_sha512_hmac(void) {
    unsigned char hmac[64];
    const unsigned char* key = (const unsigned char*)"key";
    const unsigned char* msg = (const unsigned char*)"message";
    SNEPPX_sha512_hmac(key, 3, msg, 7, hmac);
    int non_zero = 0;
    for (int i = 0; i < 64; i++) if (hmac[i]) non_zero++;
    ASSERT(non_zero > 0, "sha512 hmac produced output");
}

int main(void) {
    run_test("sha512_hash", test_sha512_hash);
    run_test("sha512_deterministic", test_sha512_deterministic);
    run_test("sha512_diff_input", test_sha512_diff_input);
    run_test("sha512_hmac", test_sha512_hmac);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
