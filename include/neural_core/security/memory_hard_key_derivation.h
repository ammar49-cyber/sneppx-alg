#ifndef ARIX_ARGON2_H
#define ARIX_ARGON2_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t memory_kb;
    size_t iterations;
    size_t parallelism;
    size_t hash_len;
} ArixArgon2Config;

#define ARIX_ARGON2_DEFAULT_MEMORY 65536
#define ARIX_ARGON2_DEFAULT_ITERATIONS 3
#define ARIX_ARGON2_DEFAULT_PARALLELISM 4
#define ARIX_ARGON2_DEFAULT_HASH_LEN 32

int arix_argon2id(const uint8_t* password, size_t password_len, const uint8_t* salt, size_t salt_len,
                  const ArixArgon2Config* config, uint8_t* hash);
int arix_argon2id_verify(const uint8_t* password, size_t password_len, const uint8_t* salt, size_t salt_len,
                         const ArixArgon2Config* config, const uint8_t* expected_hash);

#endif
