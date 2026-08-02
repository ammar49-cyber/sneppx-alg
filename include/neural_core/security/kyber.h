#ifndef SNEPPX_KYBER_H
#define SNEPPX_KYBER_H

#include <stdint.h>
#include <stddef.h>

#define KYBER_PUBLICKEYBYTES 1184
#define KYBER_SECRETKEYBYTES 1632
#define KYBER_CIPHERTEXTBYTES 1088
#define KYBER_SSBYTES 32
#define KYBER_K 3

#define KYBER_VARIANT_512 2
#define KYBER_VARIANT_768 3
#define KYBER_VARIANT_1024 4
/*
 * SNEPPX - Kyber ML-KEM (Post-Quantum Key Encapsulation)
 *
 * WHAT
 *   Kyber ML-KEM (Post-Quantum Key Encapsulation).
 *
 * CONCEPT
 *   Module-lattice (ML-KEM) key generation, encapsulation, and decapsulation for FIPS 203 Kyber-512/768/1024.
 *
 * ROLE
 *   Layer S0 post-quantum crypto of the S0-S9 security stack.
 *
 * REFERENCES
 *   FIPS 203 (Kyber), NIST PQC Round 3 finalist.
 */



int SNEPPX_kyber_keygen(uint8_t *pk, uint8_t *sk, int variant);
int SNEPPX_kyber_encaps(uint8_t *ct, uint8_t *ss, const uint8_t *pk, int variant);
int SNEPPX_kyber_decaps(uint8_t *ss, const uint8_t *ct, const uint8_t *sk, int variant);

#endif
