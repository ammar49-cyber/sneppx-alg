#ifndef SNEPPX_ARGON2_H
#define SNEPPX_ARGON2_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t memory_kb;
    size_t iterations;
    size_t parallelism;
    size_t hash_len;
} SNEPPXArgon2Config;

#define SNEPPX_ARGON2_DEFAULT_MEMORY 65536
#define SNEPPX_ARGON2_DEFAULT_ITERATIONS 3
#define SNEPPX_ARGON2_DEFAULT_PARALLELISM 4
#define SNEPPX_ARGON2_DEFAULT_HASH_LEN 32

int SNEPPX_argon2id(const uint8_t* password, size_t password_len, const uint8_t* salt, size_t salt_len,
                  const SNEPPXArgon2Config* config, uint8_t* hash);
int SNEPPX_argon2id_verify(const uint8_t* password, size_t password_len, const uint8_t* salt, size_t salt_len,
                         const SNEPPXArgon2Config* config, const uint8_t* expected_hash);

#endif
