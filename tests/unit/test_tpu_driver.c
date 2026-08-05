#include "tpu_driver.h"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test Tpu Driver
 *
 * WHAT
 *   Test Tpu Driver.
 *
 * CONCEPT
 *   Provides the Test Tpu Driver.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


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
    ASSERT(ret == 0, "register returns 0");
}

static void test_tpu_device_count(void) {
    int count = 0;
    int ret = SNEPPX_tpu_get_device_count(&count);
    ASSERT(ret == 0, "get_device_count returns 0");
    ASSERT(count > 0, "at least 1 device");
}

static void test_tpu_device_props(void) {
    SNEPPXTPUDeviceProps props;
    memset(&props, 0, sizeof(props));
    int ret = SNEPPX_tpu_get_device_props(0, &props);
    ASSERT(ret == 0, "get_device_props returns 0");
    ASSERT(props.num_cores > 0, "non-zero cores");
    ASSERT(props.device_memory_bytes > 0, "non-zero memory");
}

static void test_tpu_context(void) {
    SNEPPXTPUContext* ctx = SNEPPX_tpu_create_context(0);
    ASSERT(ctx != NULL, "context created");
    ASSERT(ctx->device_id == 0, "device id matches");
    ASSERT(ctx->client != NULL, "pjrt client created");
    SNEPPX_tpu_destroy_context(ctx);
}

static void test_tpu_malloc_free(void) {
    SNEPPXTPUContext* ctx = SNEPPX_tpu_create_context(0);
    ASSERT(ctx != NULL, "context created");

    void* ptr = NULL;
    int ret = SNEPPX_tpu_mem_alloc(&ptr, 2048, ctx);
    ASSERT(ret == 0, "mem_alloc returns 0");
    ASSERT(ptr != NULL, "pointer allocated");

    memset(ptr, 0xAB, 256);
    ret = SNEPPX_tpu_mem_free(ptr, ctx);
    ASSERT(ret == 0, "mem_free returns 0");

    SNEPPX_tpu_destroy_context(ctx);
}

static void test_tpu_memcpy(void) {
    SNEPPXTPUContext* ctx = SNEPPX_tpu_create_context(0);
    float host_src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float host_dst[4] = {0};
    void* device = NULL;
    SNEPPX_tpu_mem_alloc(&device, 16, ctx);

    int ret = SNEPPX_tpu_mem_htod(device, host_src, 16, ctx);
    ASSERT(ret == 0, "htod returns 0");

    ret = SNEPPX_tpu_mem_dtoh(host_dst, device, 16, ctx);
    ASSERT(ret == 0, "dtoh returns 0");
    ASSERT(host_dst[0] == 1.0f && host_dst[3] == 4.0f, "data integrity");

    SNEPPX_tpu_mem_free(device, ctx);
    SNEPPX_tpu_destroy_context(ctx);
}

static void test_tpu_compile_execute(void) {
    SNEPPXTPUContext* ctx = SNEPPX_tpu_create_context(0);
    const char* hlo = "HloModule test";
    SNEPPXTPUExecutable* exec = SNEPPX_tpu_compile(hlo, strlen(hlo), ctx);
    ASSERT(exec != NULL, "executable created");

    SNEPPX_tpu_executable_destroy(exec);
    SNEPPX_tpu_destroy_context(ctx);
}

int main(void) {
    run_test("tpu_register", test_tpu_register);
    run_test("tpu_device_count", test_tpu_device_count);
    run_test("tpu_device_props", test_tpu_device_props);
    run_test("tpu_context", test_tpu_context);
    run_test("tpu_malloc_free", test_tpu_malloc_free);
    run_test("tpu_memcpy", test_tpu_memcpy);
    run_test("tpu_compile_execute", test_tpu_compile_execute);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
