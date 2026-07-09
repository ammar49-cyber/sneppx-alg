#ifndef SNEPPX_SHA512_H
#define SNEPPX_SHA512_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_SHA512_BLOCK_SIZE 128
#define SNEPPX_SHA512_DIGEST_SIZE 64

typedef struct {
    uint64_t state[8];
    uint64_t count[2];
    uint8_t buffer[SNEPPX_SHA512_BLOCK_SIZE];
    unsigned int buflen;
} SNEPPXSHA512Context;

void SNEPPX_sha512_init(SNEPPXSHA512Context* ctx);
void SNEPPX_sha512_update(SNEPPXSHA512Context* ctx, const uint8_t* data, size_t len);
void SNEPPX_sha512_finish(SNEPPXSHA512Context* ctx, uint8_t digest[SNEPPX_SHA512_DIGEST_SIZE]);
void SNEPPX_sha512(const uint8_t* data, size_t len, uint8_t digest[SNEPPX_SHA512_DIGEST_SIZE]);

#endif
