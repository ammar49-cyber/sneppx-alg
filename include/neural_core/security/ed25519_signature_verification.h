#ifndef SNEPPX_ED25519_H
#define SNEPPX_ED25519_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_ED25519_PUBLIC_KEY_LEN 32
#define SNEPPX_ED25519_PRIVATE_KEY_LEN 64
#define SNEPPX_ED25519_SIGNATURE_LEN 64
#define SNEPPX_ED25519_SEED_LEN 32

typedef struct {
    uint8_t public_key[SNEPPX_ED25519_PUBLIC_KEY_LEN];
    uint8_t private_key[SNEPPX_ED25519_PRIVATE_KEY_LEN];
} SNEPPXEd25519Keypair;

typedef struct {
    uint8_t data[SNEPPX_ED25519_SIGNATURE_LEN];
} SNEPPXEd25519Signature;

int SNEPPX_ed25519_keypair_generate(SNEPPXEd25519Keypair* kp);
int SNEPPX_ed25519_secret_key_expand(uint8_t* expanded_sk, const uint8_t* seed);
int SNEPPX_ed25519_sign(const SNEPPXEd25519Keypair* kp, const uint8_t* message, size_t msg_len, SNEPPXEd25519Signature* sig);
int SNEPPX_ed25519_verify(const uint8_t* public_key, const uint8_t* message, size_t msg_len, const SNEPPXEd25519Signature* sig);
int SNEPPX_ed25519_scalar_multiply(uint8_t* result, const uint8_t* scalar, const uint8_t* point);

#endif
