#include "sphincsplus.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test Sphincsplus
 *
 * WHAT
 *   Test Sphincsplus.
 *
 * CONCEPT
 *   Provides SPHINCS+ stateless hash signatures (FIPS 205).
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


TEST(test_sphincsplus, suite) {
    uint8_t pk[SPHINCS_PUBLICKEYBYTES], sk[SPHINCS_SECRETKEYBYTES];
    uint8_t sig[SPHINCS_SIGBYTES];
    size_t siglen;
    uint8_t msg[] = "SNEPPX-Algo SPHINCS+ test message";
    if (SNEPPX_sphincs_keygen(pk, sk, 128) != 0) { printf("FAIL: keygen\n"); FAIL() << "early exit (legacy return 1)"; }
    if (SNEPPX_sphincs_sign(sig, &siglen, msg, sizeof(msg), sk, 128) != 0) { printf("FAIL: sign\n"); FAIL() << "early exit (legacy return 1)"; }
    if (SNEPPX_sphincs_verify(sig, siglen, msg, sizeof(msg), pk, 128) != 0) { printf("FAIL: verify\n"); FAIL() << "early exit (legacy return 1)"; }
    printf("PASS: SPHINCS+ sign/verify OK\n");
    return;
}
