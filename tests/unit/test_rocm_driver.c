#include "rocm_driver.h"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test Rocm Driver
 *
 * WHAT
 *   Test Rocm Driver.
 *
 * CONCEPT
 *   Provides the Test Rocm Driver.
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

static void test_rocm_register_driver(void) {
    int ret = SNEPPX_rocm_register_driver();
    (void)ret;
}

static void test_rocm_device_count(void) {
    int count = -1;
    int ret = SNEPPX_rocm_get_device_count(&count);
    ASSERT(ret == 0, "rocm_get_device_count returns 0");
    ASSERT(count >= 0, "device count >= 0");
}

static void test_rocm_device_props(void) {
    SNEPPXROCmDeviceProps props;
    memset(&props, 0, sizeof(props));
    int ret = SNEPPX_rocm_get_device_props(0, &props);
    ASSERT(ret == 0, "get_device_props returns 0");
    ASSERT(props.global_mem_bytes > 0, "non-zero memory");
    ASSERT(props.max_threads_per_workgroup > 0, "non-zero threads");
}

static void test_rocm_context(void) {
    SNEPPXROCmContext* ctx = SNEPPX_rocm_create_context(0);
    ASSERT(ctx != NULL, "context created");
    ASSERT(ctx->device_id == 0, "device id matches");
    SNEPPX_rocm_destroy_context(ctx);
}

static void test_rocm_malloc_free(void) {
    void* ptr = NULL;
    int ret = SNEPPX_rocm_mem_alloc(&ptr, 1024);
    ASSERT(ret == 0, "rocm_mem_alloc returns 0");
    ASSERT(ptr != NULL, "pointer allocated");
    if (ptr) SNEPPX_rocm_mem_free(ptr);
}

static void test_rocm_memcpy(void) {
    float host_src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float host_dst[4] = {0};
    void* device = NULL;
    SNEPPX_rocm_mem_alloc(&device, 16);
    ASSERT(device != NULL, "device mem allocated");

    int ret = SNEPPX_rocm_mem_htod(device, host_src, 16);
    ASSERT(ret == 0, "memcpy host to device");

    ret = SNEPPX_rocm_mem_dtoh(host_dst, device, 16);
    ASSERT(ret == 0, "memcpy device to host");
    ASSERT(host_dst[0] == 1.0f && host_dst[3] == 4.0f, "data integrity");

    if (device) SNEPPX_rocm_mem_free(device);
}

static void test_rocm_stream_event(void) {
    SNEPPXROCmStream* stream = NULL;
    int ret = SNEPPX_rocm_stream_create(&stream);
    ASSERT(ret == 0, "stream created");
    ASSERT(stream != NULL, "stream non-null");

    SNEPPXROCmEvent* event = NULL;
    ret = SNEPPX_rocm_event_create(&event);
    ASSERT(ret == 0, "event created");

    ret = SNEPPX_rocm_event_record(event, stream);
    ASSERT(ret == 0, "event recorded");

    ret = SNEPPX_rocm_stream_synchronize(stream);
    ASSERT(ret == 0, "stream synced");

    ret = SNEPPX_rocm_event_synchronize(event);
    ASSERT(ret == 0, "event synced");

    SNEPPX_rocm_event_destroy(event);
    SNEPPX_rocm_stream_destroy(stream);
}

int main(void) {
    run_test("rocm_register_driver", test_rocm_register_driver);
    run_test("rocm_device_count", test_rocm_device_count);
    run_test("rocm_device_props", test_rocm_device_props);
    run_test("rocm_context", test_rocm_context);
    run_test("rocm_malloc_free", test_rocm_malloc_free);
    run_test("rocm_memcpy", test_rocm_memcpy);
    run_test("rocm_stream_event", test_rocm_stream_event);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
