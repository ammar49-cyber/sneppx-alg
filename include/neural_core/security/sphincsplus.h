#ifndef ARIX_SPHINXPLUS_H
#define ARIX_SPHINXPLUS_H

#include <stdint.h>
#include <stddef.h>

#define SPHINCS_PUBLICKEYBYTES 32
#define SPHINCS_SECRETKEYBYTES 64
#define SPHINCS_SIGBYTES 8080

int arix_sphincs_keygen(uint8_t *pk, uint8_t *sk, int variant);
int arix_sphincs_sign(uint8_t *sig, size_t *siglen, const uint8_t *m, size_t mlen, const uint8_t *sk, int variant);
int arix_sphincs_verify(const uint8_t *sig, size_t siglen, const uint8_t *m, size_t mlen, const uint8_t *pk, int variant);

#endif
