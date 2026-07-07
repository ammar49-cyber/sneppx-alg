#ifndef ARIX_DILITHIUM_H
#define ARIX_DILITHIUM_H

#include <stdint.h>
#include <stddef.h>

#define DILITHIUM_PUBLICKEYBYTES 1312
#define DILITHIUM_SECRETKEYBYTES 2560
#define DILITHIUM_SIGBYTES 3368

int arix_dilithium_keygen(uint8_t *pk, uint8_t *sk, int variant);
int arix_dilithium_sign(uint8_t *sig, size_t *siglen, const uint8_t *m, size_t mlen, const uint8_t *sk, int variant);
int arix_dilithium_verify(const uint8_t *sig, size_t siglen, const uint8_t *m, size_t mlen, const uint8_t *pk, int variant);

#endif
