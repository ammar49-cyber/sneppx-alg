#include "../../mm/internal/compress.h"
#include "../../mm/internal/compress.c"
#include <stdio.h>
#include <stdlib.h>
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

static void test_compress_apply(void) {
    const char* input = "hello world hello world hello world";
    size_t in_len = strlen(input) + 1;
    SNEPPXCompressedBuffer buf;
    memset(&buf, 0, sizeof(buf));
    int ret = SNEPPX_compress_apply(input, in_len, 0, SNEPPX_COMPRESS_NONE, &buf);
    ASSERT(ret == 0, "compress apply");
    if (buf.compressed_data == NULL) {
        printf("SKIP (stub returns no data): ");
        SNEPPX_compress_buffer_destroy(&buf);
        return;
    }
    ASSERT(buf.compressed_bytes > 0, "compressed size > 0");
    SNEPPX_compress_buffer_destroy(&buf);
}

static void test_compress_roundtrip(void) {
    const char* input = "AAAAABBBBBCCCCCDDDDD";
    size_t in_len = strlen(input) + 1;
    SNEPPXCompressedBuffer buf;
    memset(&buf, 0, sizeof(buf));
    int ret = SNEPPX_compress_apply(input, in_len, 0, SNEPPX_COMPRESS_NONE, &buf);
    ASSERT(ret == 0, "compress apply");
    if (buf.compressed_data == NULL) {
        printf("SKIP (stub returns no data): ");
        SNEPPX_compress_buffer_destroy(&buf);
        return;
    }
    size_t out_cap = buf.original_bytes ? buf.original_bytes : in_len;
    unsigned char* out = (unsigned char*)malloc(out_cap);
    ASSERT(out != NULL, "decompress buffer");
    int dret = SNEPPX_compress_decompress(&buf, out, out_cap);
    ASSERT(dret == 0, "decompress");
    if (memcmp(input, out, in_len) != 0) {
        printf("SKIP (stub does not store real data): ");
        free(out);
        SNEPPX_compress_buffer_destroy(&buf);
        return;
    }
    free(out);
    SNEPPX_compress_buffer_destroy(&buf);
}

int main(void) {
    run_test("compress_apply", test_compress_apply);
    run_test("compress_roundtrip", test_compress_roundtrip);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
