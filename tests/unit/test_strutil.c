#include "strutil.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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

static void test_strutil_strlcpy(void) {
    char dst[20];
    size_t len = SNEPPX_strlcpy(dst, "hello world", sizeof(dst));
    ASSERT(len == 11, "strlcpy returns correct length");
    ASSERT(strcmp(dst, "hello world") == 0, "strlcpy copies correctly");
}

static void test_strutil_strlcat(void) {
    char dst[20] = "hello";
    size_t len = SNEPPX_strlcat(dst, " world", sizeof(dst));
    ASSERT(len == 11, "strlcat returns correct length");
    ASSERT(strcmp(dst, "hello world") == 0, "strlcat appends correctly");
}

static void test_strutil_strcmp(void) {
    ASSERT(SNEPPX_strcmp("abc", "abc") == 0, "strcmp equal strings");
    ASSERT(SNEPPX_strcmp("abc", "abd") < 0, "strcmp less than");
    ASSERT(SNEPPX_strcmp("abd", "abc") > 0, "strcmp greater than");
}

static void test_strutil_strdup(void) {
    char* dup = SNEPPX_strdup_s("hello");
    ASSERT(dup != NULL, "strdup creates copy");
    ASSERT(strcmp(dup, "hello") == 0, "strdup content matches");
    SNEPPX_free(dup, 6);
}

static void test_strutil_strsplit(void) {
    char** tokens = NULL;
    size_t count = SNEPPX_strsplit("a,b,c,d", ',', &tokens, 0);
    ASSERT(count == 4, "strsplit creates 4 tokens");
    ASSERT(strcmp(tokens[0], "a") == 0, "first token a");
    ASSERT(strcmp(tokens[3], "d") == 0, "last token d");
    for (size_t i = 0; i < count; i++) SNEPPX_free(tokens[i], 0);
    SNEPPX_free(tokens, count * sizeof(char*));
}

static void test_strutil_strjoin(void) {
    const char* parts[] = {"x", "y", "z"};
    char* joined = SNEPPX_strjoin(parts, 3, '-');
    ASSERT(joined != NULL, "strjoin creates string");
    ASSERT(strcmp(joined, "x-y-z") == 0, "strjoin joins correctly");
    SNEPPX_free(joined, 6);
}

static void test_strutil_strbuf(void) {
    SNEPPXStringBuf* sb = SNEPPX_strbuf_create(16);
    ASSERT(sb != NULL, "strbuf created");
    ASSERT(SNEPPX_strbuf_append(sb, "hello") == 0, "append hello");
    ASSERT(SNEPPX_strbuf_append(sb, " world") == 0, "append world");
    ASSERT(strcmp(sb->buf, "hello world") == 0, "strbuf content");
    SNEPPX_strbuf_destroy(sb);
}

int main(void) {
    run_test("strlcpy", test_strutil_strlcpy);
    run_test("strlcat", test_strutil_strlcat);
    run_test("strcmp", test_strutil_strcmp);
    run_test("strdup", test_strutil_strdup);
    run_test("strsplit", test_strutil_strsplit);
    run_test("strjoin", test_strutil_strjoin);
    run_test("strbuf", test_strutil_strbuf);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}