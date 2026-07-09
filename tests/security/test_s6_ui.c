#include "key_vault.h"
#include "audit_logger.h"
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

static void test_key_vault_init(void) {
    SNEPPXKeyVault vault;
    ASSERT(SNEPPX_key_vault_init(&vault) == 0, "init");
    ASSERT(vault.is_locked == 1, "locked by default");
    SNEPPX_key_vault_destroy(&vault);
}

static void test_key_vault_generate_get(void) {
    SNEPPXKeyVault vault;
    SNEPPX_key_vault_init(&vault);
    SNEPPX_key_vault_unlock(&vault, NULL);
    uint8_t key_id[16];
    ASSERT(SNEPPX_key_vault_generate_key(&vault, key_id, 3600) >= 0, "generate key");
    uint8_t key_out[32];
    ASSERT(SNEPPX_key_vault_get_key(&vault, key_id, key_out) == 0, "get key");
    SNEPPX_key_vault_destroy(&vault);
}

static void test_key_vault_rotate_revoke(void) {
    SNEPPXKeyVault vault;
    SNEPPX_key_vault_init(&vault);
    SNEPPX_key_vault_unlock(&vault, NULL);
    uint8_t key_id[16];
    SNEPPX_key_vault_generate_key(&vault, key_id, 3600);
    uint8_t old_key[32];
    SNEPPX_key_vault_get_key(&vault, key_id, old_key);
    ASSERT(SNEPPX_key_vault_rotate_key(&vault, key_id) == 0, "rotate");
    uint8_t new_key[32];
    SNEPPX_key_vault_get_key(&vault, key_id, new_key);
    int diff = 0;
    for (int i = 0; i < 32; i++) if (old_key[i] != new_key[i]) diff = 1;
    ASSERT(diff, "key changed");
    ASSERT(SNEPPX_key_vault_revoke_key(&vault, key_id) == 0, "revoke");
    ASSERT(SNEPPX_key_vault_get_key(&vault, key_id, new_key) != 0, "revoked key fails");
    SNEPPX_key_vault_destroy(&vault);
}

static void test_audit_logger(void) {
    SNEPPXAuditLogger audit;
    ASSERT(SNEPPX_audit_init(&audit, NULL) == 0, "audit init");
    ASSERT(SNEPPX_audit_log(&audit, 1, "test event", 0x42) == 0, "log entry");
    ASSERT(audit.entry_count == 1, "1 entry");
    ASSERT(SNEPPX_audit_verify_chain(&audit) == 1, "chain valid");
    SNEPPX_audit_shutdown(&audit);
}

static void test_audit_search(void) {
    SNEPPXAuditLogger audit;
    SNEPPX_audit_init(&audit, NULL);
    SNEPPX_audit_log(&audit, 1, "first event", 0);
    SNEPPX_audit_log(&audit, 2, "second event", 0);
    SNEPPX_audit_log(&audit, 1, "third event", 0);
    SNEPPXAuditEntry results[10];
    int found = SNEPPX_audit_search(&audit, 1, results, 10);
    ASSERT(found == 2, "found 2 type-1 events");
    SNEPPX_audit_shutdown(&audit);
}

int main(void) {
    run_test("key_vault_init", test_key_vault_init);
    run_test("key_vault_generate_get", test_key_vault_generate_get);
    run_test("key_vault_rotate_revoke", test_key_vault_rotate_revoke);
    run_test("audit_logger", test_audit_logger);
    run_test("audit_search", test_audit_search);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
