#include "neural_core/drivers/driver_status.h"
#include "amd_driver.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

/*
 * SNEPPX - Test Backend Amd
 *
 * WHAT
 *   Test Backend Amd.
 *
 * CONCEPT
 *   Provides the Test Backend Amd.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static int pass = 0, fail = 0;
#define CHECK(c, m) do { if (!(c)) { printf("FAIL: %s\n", m); fail++; } else { pass++; } } while (0)
static int nearf(float a, float b) { return fabsf(a - b) < 0.05f; }

int main(void) {
    int r = SNEPPX_amd_register();
    if (r == SNEPPX_DRIVER_OK) {
        int cnt = 0;
        CHECK(SNEPPX_amd_get_device_count(&cnt) == SNEPPX_DRIVER_OK && cnt >= 1, "amd device count");
        char name[64]; size_t mem = 0; int cu = 0;
        CHECK(SNEPPX_amd_get_device_props(0, name, sizeof(name), &mem, &cu) == SNEPPX_DRIVER_OK, "amd device props");
        void* ctx = SNEPPX_amd_create_context(0);
        CHECK(ctx != NULL, "amd create context");
        void *a = NULL, *b = NULL, *cptr = NULL;
        CHECK(SNEPPX_amd_mem_alloc(&a, 16) == SNEPPX_DRIVER_OK && a != NULL, "amd mem alloc a");
        CHECK(SNEPPX_amd_mem_alloc(&b, 16) == SNEPPX_DRIVER_OK, "amd mem alloc b");
        CHECK(SNEPPX_amd_mem_alloc(&cptr, 16) == SNEPPX_DRIVER_OK, "amd mem alloc c");
        float A[4] = {1,2,3,4}, B[4] = {5,6,7,8};
        CHECK(SNEPPX_amd_memcpy_htod(a, A, 16) == SNEPPX_DRIVER_OK, "amd htod");
        CHECK(SNEPPX_amd_memcpy_htod(b, B, 16) == SNEPPX_DRIVER_OK, "amd htod b");
        void* args[3] = {a, b, cptr};
        CHECK(SNEPPX_amd_launch_kernel("gemm", ctx, args, 3, 2, 1) == SNEPPX_DRIVER_OK, "amd launch gemm");
        float C[4] = {0};
        CHECK(SNEPPX_amd_memcpy_dtoh(C, cptr, 16) == SNEPPX_DRIVER_OK, "amd dtoh");
        CHECK(nearf(C[0],19.0f) && nearf(C[3],50.0f), "amd gemm result");
        SNEPPX_amd_mem_free(a); SNEPPX_amd_mem_free(b); SNEPPX_amd_mem_free(cptr);
        SNEPPX_amd_destroy_context(ctx);
    } else {
        CHECK(r == SNEPPX_DRIVER_UNSUPPORTED, "amd reports unsupported");
    }
    printf("%d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
