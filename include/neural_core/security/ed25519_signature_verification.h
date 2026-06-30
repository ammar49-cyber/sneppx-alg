#ifndef ARIX_ED25519_H
#define ARIX_ED25519_H

#include <stddef.h>
#include <stdint.h>

#define ARIX_ED25519_PUBLIC_KEY_LEN 32
#define ARIX_ED25519_PRIVATE_KEY_LEN 64
#define ARIX_ED25519_SIGNATURE_LEN 64
#define ARIX_ED25519_SEED_LEN 32

typedef struct {
    uint8_t public_key[ARIX_ED25519_PUBLIC_KEY_LEN];
    uint8_t private_key[ARIX_ED25519_PRIVATE_KEY_LEN];
} ArixEd25519Keypair;

typedef struct {
    uint8_t data[ARIX_ED25519_SIGNATURE_LEN];
} ArixEd25519Signature;

int arix_ed25519_keypair_generate(ArixEd25519Keypair* kp);
int arix_ed25519_secret_key_expand(uint8_t* expanded_sk, const uint8_t* seed);
int arix_ed25519_sign(const ArixEd25519Keypair* kp, const uint8_t* message, size_t msg_len, ArixEd25519Signature* sig);
int arix_ed25519_verify(const uint8_t* public_key, const uint8_t* message, size_t msg_len, const ArixEd25519Signature* sig);
int arix_ed25519_scalar_multiply(uint8_t* result, const uint8_t* scalar, const uint8_t* point);

#endif
