#include "sphincsplus.h"
#include <stdio.h>
#include <string.h>

int main() {
    uint8_t pk[SPHINCS_PUBLICKEYBYTES], sk[SPHINCS_SECRETKEYBYTES];
    uint8_t sig[SPHINCS_SIGBYTES];
    size_t siglen;
    uint8_t msg[] = "SNEPPX-Algo SPHINCS+ test message";
    if (SNEPPX_sphincs_keygen(pk, sk, 128) != 0) { printf("FAIL: keygen\n"); return 1; }
    if (SNEPPX_sphincs_sign(sig, &siglen, msg, sizeof(msg), sk, 128) != 0) { printf("FAIL: sign\n"); return 1; }
    if (SNEPPX_sphincs_verify(sig, siglen, msg, sizeof(msg), pk, 128) != 0) { printf("FAIL: verify\n"); return 1; }
    printf("PASS: SPHINCS+ sign/verify OK\n");
    return 0;
}
