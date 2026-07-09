#include "memory_management.h"
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

static void test_compress_roundtrip(void) {
    const char* input = "hello world hello world hello world";
    size_t in_len = strlen(input) + 1;
    size_t comp_len = 0;
    int ret = SNEPPX_compress(input, in_len, NULL, &comp_len);
    ASSERT(ret == 0, "get compressed size");
    ASSERT(comp_len > 0, "compressed size > 0");
}

static void test_compress_decompress(void) {
    const char* input = "AAAAABBBBBCCCCCDDDDD";
    size_t in_len = strlen(input) + 1;
    size_t comp_len = 0;
    SNEPPX_compress(input, in_len, NULL, &comp_len);
    unsigned char* compressed = (unsigned char*)malloc(comp_len);
    ASSERT(compressed != NULL, "compressed buffer");
    SNEPPX_compress(input, in_len, compressed, &comp_len);
    size_t decomp_len = in_len;
    unsigned char* decompressed = (unsigned char*)malloc(decomp_len);
    ASSERT(decompressed != NULL, "decompressed buffer");
    SNEPPX_decompress(compressed, comp_len, decompressed, &decomp_len);
    ASSERT(memcmp(input, decompressed, in_len) == 0, "roundtrip");
    free(decompressed);
    free(compressed);
}

int main(void) {
    run_test("compress_roundtrip", test_compress_roundtrip);
    run_test("compress_decompress", test_compress_decompress);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
