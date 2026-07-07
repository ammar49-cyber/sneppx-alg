#include "kyber.h"
#include "cryptographic_random_generator.h"
#include <stdio.h>
#include <string.h>

int main() {
    uint8_t pk[KYBER_PUBLICKEYBYTES], sk[KYBER_SECRETKEYBYTES];
    uint8_t ct[KYBER_CIPHERTEXTBYTES], ss1[KYBER_SSBYTES], ss2[KYBER_SSBYTES];
    if (arix_kyber_keygen(pk, sk, 3) != 0) { printf("FAIL: keygen\n"); return 1; }
    if (arix_kyber_encaps(ct, ss1, pk, 3) != 0) { printf("FAIL: encaps\n"); return 1; }
    if (arix_kyber_decaps(ss2, ct, sk, 3) != 0) { printf("FAIL: decaps\n"); return 1; }
    if (memcmp(ss1, ss2, KYBER_SSBYTES) != 0) { printf("FAIL: shared secret mismatch\n"); return 1; }
    printf("PASS: Kyber KEM round-trip OK\n");
    return 0;
}
