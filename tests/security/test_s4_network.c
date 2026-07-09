#include "transport_security.h"
#include "identity_management.h"
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

static void test_transport_init(void) {
    SNEPPXTransportSecurity ts;
    ASSERT(SNEPPX_transport_init(&ts) == 0, "init");
    ASSERT(ts.enabled == 1, "enabled by default");
    SNEPPX_transport_shutdown(&ts);
}

static void test_transport_session(void) {
    SNEPPXTransportSecurity ts;
    SNEPPX_transport_init(&ts);
    int sid = SNEPPX_transport_new_session(&ts, NULL, 0);
    ASSERT(sid >= 0, "new session");
    int sid2 = SNEPPX_transport_new_session(&ts, NULL, 0);
    ASSERT(sid2 != sid, "distinct sessions");
    ASSERT(SNEPPX_transport_close_session(&ts, sid) == 0, "close session");
    SNEPPX_transport_shutdown(&ts);
}

static void test_transport_encrypt_decrypt(void) {
    SNEPPXTransportSecurity ts;
    SNEPPX_transport_init(&ts);
    int sid = SNEPPX_transport_new_session(&ts, NULL, 0);
    uint8_t plaintext[] = "hello transport security!";
    uint8_t ciphertext[256];
    uint8_t nonce[12];
    ASSERT(SNEPPX_transport_encrypt(&ts, sid, plaintext, strlen((char*)plaintext)+1, ciphertext, nonce) == 0, "encrypt");
    uint8_t decrypted[256];
    ASSERT(SNEPPX_transport_decrypt(&ts, sid, ciphertext, strlen((char*)plaintext)+1, nonce, decrypted) == 0, "decrypt");
    ASSERT(strcmp((char*)plaintext, (char*)decrypted) == 0, "roundtrip");
    SNEPPX_transport_shutdown(&ts);
}

static void test_transport_noise_handshake(void) {
    SNEPPXTransportSecurity ts;
    SNEPPX_transport_init(&ts);
    uint8_t msg[64];
    size_t msg_len = sizeof(msg);
    ASSERT(SNEPPX_transport_noise_handshake(&ts, NULL, 0, msg, &msg_len) == 0, "noise handshake");
    ASSERT(msg_len == 48, "handshake msg 48 bytes");
    SNEPPX_transport_shutdown(&ts);
}

static void test_identity_init(void) {
    SNEPPXIdentityManager mgr;
    ASSERT(SNEPPX_identity_init(&mgr) == 0, "identity init");
    ASSERT(mgr.ddos_protection_enabled == 1, "ddos enabled");
    SNEPPX_identity_shutdown(&mgr);
}

static void test_identity_pin_verify(void) {
    SNEPPXIdentityManager mgr;
    SNEPPX_identity_init(&mgr);
    uint8_t fp[32];
    memset(fp, 0xAB, 32);
    ASSERT(SNEPPX_identity_pin_cert(&mgr, fp, "test.SNEPPX.local", 0) >= 0, "pin cert");
    ASSERT(SNEPPX_identity_verify_cert(&mgr, fp) == 1, "verify pinned cert");
    uint8_t bad_fp[32];
    memset(bad_fp, 0, 32);
    ASSERT(SNEPPX_identity_verify_cert(&mgr, bad_fp) == 0, "unpinned cert rejected");
    ASSERT(SNEPPX_identity_unpin_cert(&mgr, fp) == 0, "unpin cert");
    ASSERT(SNEPPX_identity_verify_cert(&mgr, fp) == 0, "unpinned cert not verified");
    SNEPPX_identity_shutdown(&mgr);
}

static void test_ddos_protection(void) {
    SNEPPXIdentityManager mgr;
    SNEPPX_identity_init(&mgr);
    mgr.ddos_request_limit = 5;
    for (int i = 0; i < 5; i++) ASSERT(SNEPPX_identity_ddos_check(&mgr) == 0, "under limit");
    ASSERT(SNEPPX_identity_ddos_check(&mgr) == 1, "over limit");
    SNEPPX_identity_ddos_reset(&mgr);
    ASSERT(SNEPPX_identity_ddos_check(&mgr) == 0, "after reset");
    SNEPPX_identity_shutdown(&mgr);
}

int main(void) {
    run_test("transport_init", test_transport_init);
    run_test("transport_session", test_transport_session);
    run_test("transport_encrypt_decrypt", test_transport_encrypt_decrypt);
    run_test("transport_noise_handshake", test_transport_noise_handshake);
    run_test("identity_init", test_identity_init);
    run_test("identity_pin_verify", test_identity_pin_verify);
    run_test("ddos_protection", test_ddos_protection);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
