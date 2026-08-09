#include <stdio.h>
#include "test_gtest.h"
#include <string.h>
#include <stdlib.h>
#include "ed25519_signature_verification.h"
#include "cryptographic_random_generator.h"

/*
 * SNEPPX - Test Ed25519
 *
 * WHAT
 *   Test Ed25519.
 *
 * CONCEPT
 *   Provides Ed25519 signatures (RFC 8032).
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */




void test_keypair_generate(void) {
    printf("\n--- test_keypair_generate ---\n");
    SNEPPXEd25519Keypair kps[100];
    for (int i = 0; i < 100; i++) {
        int ret = SNEPPX_ed25519_keypair_generate(&kps[i]);
        SX_TEST("keypair generate success", ret == 0);
        if (i > 0) {
            int dup = memcmp(kps[i].public_key, kps[i-1].public_key, 32) == 0;
            SX_TEST("no duplicate keypairs", !dup);
        }
        int non_zero = 0;
        for (int j = 0; j < 32; j++) non_zero |= kps[i].public_key[j];
        SX_TEST("public key non-zero", non_zero != 0);
    }
}

void test_sign_verify(void) {
    printf("\n--- test_sign_verify ---\n");
    SNEPPXEd25519Keypair kp;
    SNEPPX_ed25519_keypair_generate(&kp);

    uint8_t msg[] = "Hello, SNEPPX-Algo Ed25519!";
    SNEPPXEd25519Signature sig;
    int ret = SNEPPX_ed25519_sign(&kp, msg, sizeof(msg), &sig);
    SX_TEST("sign success", ret == 0);

    ret = SNEPPX_ed25519_verify(kp.public_key, msg, sizeof(msg), &sig);
    SX_TEST("verify success", ret == 1);

    SNEPPXEd25519Keypair wrong_kp;
    SNEPPX_ed25519_keypair_generate(&wrong_kp);
    ret = SNEPPX_ed25519_verify(wrong_kp.public_key, msg, sizeof(msg), &sig);
    SX_TEST("verify wrong key fails", ret == 0 || ret == -1);
}

void test_signature_malleability(void) {
    printf("\n--- test_signature_malleability ---\n");
    SNEPPXEd25519Keypair kp;
    SNEPPX_ed25519_keypair_generate(&kp);
    uint8_t msg[] = "Test message";
    SNEPPXEd25519Signature sig, modified;
    SNEPPX_ed25519_sign(&kp, msg, sizeof(msg), &sig);
    memcpy(&modified, &sig, sizeof(sig));

    modified.data[0] ^= 1;
    int ret = SNEPPX_ed25519_verify(kp.public_key, msg, sizeof(msg), &modified);
    SX_TEST("modified S fails", ret == 0 || ret == -1);

    memcpy(&modified, &sig, sizeof(sig));
    modified.data[10] ^= 1;
    ret = SNEPPX_ed25519_verify(kp.public_key, msg, sizeof(msg), &modified);
    SX_TEST("modified R fails", ret == 0 || ret == -1);
}

void test_large_message(void) {
    printf("\n--- test_large_message ---\n");
    SNEPPXEd25519Keypair kp;
    SNEPPX_ed25519_keypair_generate(&kp);
    size_t len = 1024 * 1024;
    uint8_t* msg = (uint8_t*)malloc(len);
    if (!msg) { printf("SKIP: large_message (malloc failed)\n"); return; }
    memset(msg, 'A', len);
    SNEPPXEd25519Signature sig;
    int ret = SNEPPX_ed25519_sign(&kp, msg, len, &sig);
    SX_TEST("sign 1MB", ret == 0);
    ret = SNEPPX_ed25519_verify(kp.public_key, msg, len, &sig);
    SX_TEST("verify 1MB", ret == 1);
    free(msg);
}


TEST(test_ed25519, test_keypair_generate) { test_keypair_generate(); }
TEST(test_ed25519, test_sign_verify) { test_sign_verify(); }
TEST(test_ed25519, test_signature_malleability) { test_signature_malleability(); }
TEST(test_ed25519, test_large_message) { test_large_message(); }
