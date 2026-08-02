#ifndef SNEPPX_SPHINCS_PLUS_H
#define SNEPPX_SPHINCS_PLUS_H

#include <stdint.h>
#include <stddef.h>

#define SPHINCS_PUBLICKEYBYTES 32
#define SPHINCS_SECRETKEYBYTES 64
#define SPHINCS_SIGBYTES 8080
/*
 * SNEPPX - SPHINCS+ Stateless Hash-Based Signatures
 *
 * WHAT
 *   SPHINCS+ Stateless Hash-Based Signatures.
 *
 * CONCEPT
 *   SPHINCS+ hash-based signature scheme for FIPS 205 quantum-safe signatures.
 *
 * ROLE
 *   Layer S0 post-quantum crypto for long-lived signatures.
 *
 * REFERENCES
 *   FIPS 205 (SPHINCS+), NIST PQC Round 3 finalist.
 */



int SNEPPX_sphincs_keygen(uint8_t *pk, uint8_t *sk, int variant);
int SNEPPX_sphincs_sign(uint8_t *sig, size_t *siglen, const uint8_t *m, size_t mlen, const uint8_t *sk, int variant);
int SNEPPX_sphincs_verify(const uint8_t *sig, size_t siglen, const uint8_t *m, size_t mlen, const uint8_t *pk, int variant);

#endif
