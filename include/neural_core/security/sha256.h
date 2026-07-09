#ifndef SNEPPX_SHA256_H
#define SNEPPX_SHA256_H

#include <stdint.h>
#include <stddef.h>

void SNEPPX_sha256(uint8_t out[32], const uint8_t *in, size_t inlen);

#endif
