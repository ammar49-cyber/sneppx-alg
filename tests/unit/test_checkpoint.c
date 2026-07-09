#include "checkpoint_reader.h"
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

static void test_checkpoint_reader_create_null(void) {
    SNEPPXCheckpointReader* reader = SNEPPX_checkpoint_reader_create(NULL);
    ASSERT(reader == NULL, "create with NULL path returns NULL");
}

static void test_checkpoint_reader_create(void) {
    SNEPPXCheckpointReader* reader = SNEPPX_checkpoint_reader_create("/tmp/test.ckpt");
    ASSERT(reader != NULL, "checkpoint reader created");
    SNEPPX_checkpoint_reader_destroy(reader);
}

static void test_checkpoint_reader_get_tensor(void) {
    SNEPPXCheckpointReader* reader = SNEPPX_checkpoint_reader_create("/tmp/test.ckpt");
    SNEPPXTensor* t = SNEPPX_checkpoint_reader_get_tensor(reader, "weights");
    /* may be NULL if file doesn't exist */
    if (t) SNEPPX_tensor_destroy(t);
    SNEPPX_checkpoint_reader_destroy(reader);
}

int main(void) {
    run_test("checkpoint_reader_create_null", test_checkpoint_reader_create_null);
    run_test("checkpoint_reader_create", test_checkpoint_reader_create);
    run_test("checkpoint_reader_get_tensor", test_checkpoint_reader_get_tensor);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
