#include "neural_core/drivers/driver_status.h"
#include "npu_driver.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

/*
 * SNEPPX - Test Backend Npu
 *
 * WHAT
 *   Test Backend Npu.
 *
 * CONCEPT
 *   Provides the Test Backend Npu.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static int pass = 0, fail = 0;
#define CHECK(c, m) do { if (!(c)) { printf("FAIL: %s\n", m); fail++; } else { pass++; } } while (0)

int main(void) {
    int r = SNEPPX_npu_register_driver();
    if (r == SNEPPX_DRIVER_OK) {
        int cnt = 0;
        CHECK(SNEPPX_npu_get_device_count(&cnt) == SNEPPX_DRIVER_OK && cnt >= 1, "npu device count");
        void* ctx = SNEPPX_npu_create_context(0);
        CHECK(ctx != NULL, "npu create context");
        void *a = NULL, *b = NULL;
        CHECK(SNEPPX_npu_mem_alloc(&a, 16, ctx) == SNEPPX_DRIVER_OK && a != NULL, "npu mem alloc");
        CHECK(SNEPPX_npu_mem_alloc(&b, 16, ctx) == SNEPPX_DRIVER_OK, "npu mem alloc b");
        float A[4] = {1,2,3,4};
        CHECK(SNEPPX_npu_mem_htod(a, A, 16, ctx) == SNEPPX_DRIVER_OK, "npu htod");
        float B[4] = {0};
        CHECK(SNEPPX_npu_mem_dtoh(B, a, 16, ctx) == SNEPPX_DRIVER_OK, "npu dtoh");
        CHECK(B[0] == 1.0f && B[3] == 4.0f, "npu memcpy roundtrip");
        void* inputs[1] = {a};
        void* outputs[1] = {b};
        CHECK(SNEPPX_npu_execute(NULL, inputs, 1, outputs, 1, ctx) == SNEPPX_DRIVER_OK, "npu execute");
        SNEPPX_npu_mem_free(a, ctx); SNEPPX_npu_mem_free(b, ctx);
        SNEPPX_npu_destroy_context(ctx);
    } else {
        CHECK(r == SNEPPX_DRIVER_UNSUPPORTED, "npu reports unsupported");
    }
    printf("%d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
