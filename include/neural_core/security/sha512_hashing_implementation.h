#ifndef ARIX_SHA512_H
#define ARIX_SHA512_H

#include <stddef.h>
#include <stdint.h>

#define ARIX_SHA512_BLOCK_SIZE 128
#define ARIX_SHA512_DIGEST_SIZE 64

typedef struct {
    uint64_t state[8];
    uint64_t count[2];
    uint8_t buffer[ARIX_SHA512_BLOCK_SIZE];
    unsigned int buflen;
} ArixSHA512Context;

void arix_sha512_init(ArixSHA512Context* ctx);
void arix_sha512_update(ArixSHA512Context* ctx, const uint8_t* data, size_t len);
void arix_sha512_finish(ArixSHA512Context* ctx, uint8_t digest[ARIX_SHA512_DIGEST_SIZE]);
void arix_sha512(const uint8_t* data, size_t len, uint8_t digest[ARIX_SHA512_DIGEST_SIZE]);

#endif
