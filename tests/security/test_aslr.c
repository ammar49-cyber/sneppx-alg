#include "address_space_randomization.h"
#include <stdio.h>

/*
 * SNEPPX - Test Aslr
 *
 * WHAT
 *   Test Aslr.
 *
 * CONCEPT
 *   Provides address space layout randomization.
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

static void test_aslr_random_offset(void) {
    size_t off = SNEPPX_aslr_random_offset(4096);
    ASSERT(off < 4096, "random offset within range");
    ASSERT((off & 4095) == 0, "random offset page-aligned");
}

static void test_aslr_random_offset_varied(void) {
    size_t off1 = SNEPPX_aslr_random_offset(65536);
    size_t off2 = SNEPPX_aslr_random_offset(65536);
    ASSERT(off1 < 65536, "offset1 within range");
    ASSERT(off2 < 65536, "offset2 within range");
}

static void test_aslr_apply(void) {
    char buf[256];
    void* ptr = (void*)buf;
    size_t sz = sizeof(buf);
    SNEPPX_aslr_apply(&ptr, &sz, 128);
    ASSERT(sz <= sizeof(buf), "size not increased");
    // offset might be 0, so we just check it doesn't crash
    ASSERT(ptr != NULL, "ptr valid after apply");
}

int main(void) {
    run_test("aslr_random_offset", test_aslr_random_offset);
    run_test("aslr_random_offset_varied", test_aslr_random_offset_varied);
    run_test("aslr_apply", test_aslr_apply);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
