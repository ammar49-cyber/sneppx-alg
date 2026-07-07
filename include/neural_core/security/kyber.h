#ifndef ARIX_KYBER_H
#define ARIX_KYBER_H

#include <stdint.h>
#include <stddef.h>

#define KYBER_PUBLICKEYBYTES 1184
#define KYBER_SECRETKEYBYTES 1632
#define KYBER_CIPHERTEXTBYTES 1088
#define KYBER_SSBYTES 32
#define KYBER_K 3

int arix_kyber_keygen(uint8_t *pk, uint8_t *sk, int variant);
int arix_kyber_encaps(uint8_t *ct, uint8_t *ss, const uint8_t *pk, int variant);
int arix_kyber_decaps(uint8_t *ss, const uint8_t *ct, const uint8_t *sk, int variant);

#endif
