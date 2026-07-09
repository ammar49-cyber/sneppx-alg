#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "ed25519_signature_verification.h"
#include "cryptographic_random_generator.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name, expr) do { \
    if (!(expr)) { printf("FAIL: %s\n", name); tests_failed++; } \
    else { printf("PASS: %s\n", name); tests_passed++; } \
} while(0)

void test_keypair_generate(void) {
    printf("\n--- test_keypair_generate ---\n");
    SNEPPXEd25519Keypair kps[100];
    for (int i = 0; i < 100; i++) {
        int ret = SNEPPX_ed25519_keypair_generate(&kps[i]);
        TEST("keypair generate success", ret == 0);
        if (i > 0) {
            int dup = memcmp(kps[i].public_key, kps[i-1].public_key, 32) == 0;
            TEST("no duplicate keypairs", !dup);
        }
        int non_zero = 0;
        for (int j = 0; j < 32; j++) non_zero |= kps[i].public_key[j];
        TEST("public key non-zero", non_zero != 0);
    }
}

void test_sign_verify(void) {
    printf("\n--- test_sign_verify ---\n");
    SNEPPXEd25519Keypair kp;
    SNEPPX_ed25519_keypair_generate(&kp);

    uint8_t msg[] = "Hello, SNEPPX-Algo Ed25519!";
    SNEPPXEd25519Signature sig;
    int ret = SNEPPX_ed25519_sign(&kp, msg, sizeof(msg), &sig);
    TEST("sign success", ret == 0);

    ret = SNEPPX_ed25519_verify(kp.public_key, msg, sizeof(msg), &sig);
    TEST("verify success", ret == 1);

    SNEPPXEd25519Keypair wrong_kp;
    SNEPPX_ed25519_keypair_generate(&wrong_kp);
    ret = SNEPPX_ed25519_verify(wrong_kp.public_key, msg, sizeof(msg), &sig);
    TEST("verify wrong key fails", ret == 0 || ret == -1);
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
    TEST("modified S fails", ret == 0 || ret == -1);

    memcpy(&modified, &sig, sizeof(sig));
    modified.data[10] ^= 1;
    ret = SNEPPX_ed25519_verify(kp.public_key, msg, sizeof(msg), &modified);
    TEST("modified R fails", ret == 0 || ret == -1);
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
    TEST("sign 1MB", ret == 0);
    ret = SNEPPX_ed25519_verify(kp.public_key, msg, len, &sig);
    TEST("verify 1MB", ret == 1);
    free(msg);
}

int main(void) {
    printf("=== SNEPPX-Ed25519 Test Suite ===\n");
    test_keypair_generate();
    test_sign_verify();
    test_signature_malleability();
    test_large_message();
    printf("\nResults: %d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
