#ifndef ARIX_SHA256_H
#define ARIX_SHA256_H

#include <stdint.h>
#include <stddef.h>

void arix_sha256(uint8_t out[32], const uint8_t *in, size_t inlen);

#endif
