#include "polymorphic_memory_allocator.h"
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

static void test_strutil_trim(void) {
    char input[] = "  hello world  ";
    SNEPPX_strutil_trim(input);
    ASSERT(strcmp(input, "hello world") == 0, "trim spaces");
}

static void test_strutil_split(void) {
    const char* input = "a,b,c,d";
    SNEPPXStrArray* arr = SNEPPX_strutil_split(input, ',');
    ASSERT(arr != NULL, "split created");
    ASSERT(arr->count == 4, "4 parts");
    ASSERT(strcmp(arr->items[0], "a") == 0, "first part a");
    ASSERT(strcmp(arr->items[3], "d") == 0, "last part d");
    SNEPPX_strutil_array_free(arr);
}

static void test_strutil_join(void) {
    const char* parts[] = {"x", "y", "z"};
    char* joined = SNEPPX_strutil_join(parts, 3, "-");
    ASSERT(joined != NULL, "joined string");
    ASSERT(strcmp(joined, "x-y-z") == 0, "x-y-z");
    free(joined);
}

static void test_strutil_starts_ends(void) {
    ASSERT(SNEPPX_strutil_starts_with("hello world", "hello"), "starts_with true");
    ASSERT(!SNEPPX_strutil_starts_with("hello world", "world"), "starts_with false");
    ASSERT(SNEPPX_strutil_ends_with("hello world", "world"), "ends_with true");
    ASSERT(!SNEPPX_strutil_ends_with("hello world", "hello"), "ends_with false");
}

int main(void) {
    run_test("strutil_trim", test_strutil_trim);
    run_test("strutil_split", test_strutil_split);
    run_test("strutil_join", test_strutil_join);
    run_test("strutil_starts_ends", test_strutil_starts_ends);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
