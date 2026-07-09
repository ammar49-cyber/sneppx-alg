#include "signed_update.h"
#include "s7_extensions.h"
#include <stdio.h>
#include <string.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s (%s)\n", msg, #cond); tests_failed++; return; } \
} while(0)

static void run_test(const char* name, void (*test_fn)(void)) {
    printf("Running %s... ", name); fflush(stdout);
    test_fn(); printf("PASS\n"); tests_passed++;
}

static void test_update_verifier_init(void) {
    SNEPPXUpdateVerifier uv;
    ASSERT(SNEPPX_update_verifier_init(&uv) == 0, "init");
    ASSERT(uv.rollback_protection_enabled == 1, "rollback protection");
    SNEPPX_update_verifier_destroy(&uv);
}

static void test_update_verifier_rollback_detect(void) {
    SNEPPXUpdateVerifier uv;
    SNEPPX_update_verifier_init(&uv);
    uint32_t older[] = {0, 9, 0};
    ASSERT(SNEPPX_update_verifier_rollback_check(&uv, older) == 1, "rollback detected");
    uint32_t same[] = {1, 0, 0};
    ASSERT(SNEPPX_update_verifier_rollback_check(&uv, same) == 0, "same version ok");
    uint32_t newer[] = {2, 0, 0};
    ASSERT(SNEPPX_update_verifier_rollback_check(&uv, newer) == 0, "newer version ok");
    SNEPPX_update_verifier_destroy(&uv);
}

static void test_update_verifier_min_version(void) {
    SNEPPXUpdateVerifier uv;
    SNEPPX_update_verifier_init(&uv);
    ASSERT(SNEPPX_update_verifier_set_min_version(&uv, 1, 5, 0) == 0, "set min version");
    uint32_t below_min[] = {1, 2, 0};
    ASSERT(SNEPPX_update_verifier_rollback_check(&uv, below_min) == 1, "below min rejected");
    SNEPPX_update_verifier_destroy(&uv);
}

static void test_signed_update_apply(void) {
    SNEPPXUpdateVerifier uv;
    SNEPPX_update_verifier_init(&uv);
    SNEPPXSignedUpdate update;
    memset(&update, 0, sizeof(update));
    update.version_major = 2;
    update.version_minor = 0;
    update.version_patch = 0;
    ASSERT(SNEPPX_update_verifier_apply(&uv, &update, NULL, 0) == 0, "apply update");
    ASSERT(uv.current_version[0] == 2, "version updated");
    SNEPPX_update_verifier_destroy(&uv);
}

static void test_tuf_init(void) {
    SNEPPXTUFMetadata tuf;
    ASSERT(SNEPPX_tuf_init(&tuf) == 0, "tuf init");
    ASSERT(tuf.initialized == 1, "tuf initialized");
}

static void test_ab_partition(void) {
    SNEPPXABPartition ab;
    ASSERT(SNEPPX_ab_partition_init(&ab) == 0, "ab init");
    ASSERT(SNEPPX_ab_partition_mark_good(&ab, 0) == 0, "mark slot 0 good");
    ASSERT(SNEPPX_ab_partition_swap(&ab) == 1, "swap ready");
}

static void test_canary_rollout(void) {
    SNEPPXCanaryRollout cr;
    ASSERT(SNEPPX_canary_rollout_init(&cr, 100, 5) == 0, "canary init");
    ASSERT(cr.total_nodes == 100, "total nodes");
    ASSERT(cr.canary_nodes == 5, "canary nodes");
    ASSERT(SNEPPX_canary_rollout_promote(&cr) == 0, "promote");
    ASSERT(cr.promoted == 1, "promoted");
}

static void test_dep_resolver(void) {
    SNEPPXDepResolver dr;
    ASSERT(SNEPPX_dep_resolver_init(&dr) == 0, "dep init");
    ASSERT(SNEPPX_dep_resolver_add_dep(&dr, "libfoo", 1, 2, 3) == 0, "add dep");
    ASSERT(SNEPPX_dep_resolver_resolve(&dr) == 0, "resolve");
    ASSERT(dr.resolved == 1, "resolved");
}

int main(void) {
    run_test("update_verifier_init", test_update_verifier_init);
    run_test("update_verifier_rollback_detect", test_update_verifier_rollback_detect);
    run_test("update_verifier_min_version", test_update_verifier_min_version);
    run_test("signed_update_apply", test_signed_update_apply);
    run_test("tuf_init", test_tuf_init);
    run_test("ab_partition", test_ab_partition);
    run_test("canary_rollout", test_canary_rollout);
    run_test("dep_resolver", test_dep_resolver);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
