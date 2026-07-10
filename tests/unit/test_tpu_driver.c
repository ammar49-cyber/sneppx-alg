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

static void test_tpu_register(void) {
    int ret = SNEPPX_tpu_register_driver();
    ASSERT(ret == 0, "tpu_register_driver returns 0 (stub)");
}

static void test_tpu_device_count(void) {
    int count = 0;
    int ret = SNEPPX_tpu_get_device_count(&count);
    ASSERT(ret == 0, "tpu_get_device_count returns 0");
    ASSERT(count >= 0, "device count >= 0");
}

static void test_tpu_malloc_free(void) {
    void* ptr = NULL;
    int ret = SNEPPX_tpu_mem_alloc(&ptr, 2048, NULL);
    ASSERT(ret == 0, "tpu_mem_alloc returns 0 (stub)");
    if (ptr) SNEPPX_tpu_mem_free(ptr, NULL);
}

static void test_tpu_execute(void) {
    int ret = SNEPPX_tpu_execute(NULL, NULL, 0, NULL, 0, NULL);
    ASSERT(ret == 0, "tpu_execute stub");
}

int main(void) {
    run_test("tpu_register", test_tpu_register);
    run_test("tpu_device_count", test_tpu_device_count);
    run_test("tpu_malloc_free", test_tpu_malloc_free);
    run_test("tpu_execute", test_tpu_execute);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
