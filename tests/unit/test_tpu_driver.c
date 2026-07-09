#include "tpu_driver.h"
#include <stdio.h>

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

static void test_tpu_init_shutdown(void) {
    int ret = SNEPPX_tpu_init();
    ASSERT(ret == 0, "tpu_init returns 0 (stub)");
    SNEPPX_tpu_shutdown();
}

static void test_tpu_device_count(void) {
    int count = SNEPPX_tpu_device_count();
    ASSERT(count >= 0, "device count >= 0");
}

static void test_tpu_malloc_free(void) {
    void* ptr = NULL;
    int ret = SNEPPX_tpu_malloc(&ptr, 2048);
    ASSERT(ret == 0, "tpu_malloc returns 0 (stub)");
    if (ptr) SNEPPX_tpu_free(ptr);
}

static void test_tpu_execute(void) {
    int ret = SNEPPX_tpu_execute(NULL, 0, NULL, 0);
    ASSERT(ret == 0, "tpu_execute stub");
}

int main(void) {
    run_test("tpu_init_shutdown", test_tpu_init_shutdown);
    run_test("tpu_device_count", test_tpu_device_count);
    run_test("tpu_malloc_free", test_tpu_malloc_free);
    run_test("tpu_execute", test_tpu_execute);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
