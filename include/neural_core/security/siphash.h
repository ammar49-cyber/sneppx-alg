#ifndef SNEPPX_SIPHASH_H
#define SNEPPX_SIPHASH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_SIPHASH_KEY_SIZE 16
#define SNEPPX_SIPHASH_OUT_SIZE 8

typedef struct {
    uint64_t v0, v1, v2, v3;
    uint64_t k0, k1;
    int c_rounds;
    int d_rounds;
} SNEPPXSipHash;

void SNEPPX_siphash_init(SNEPPXSipHash* sh, const uint8_t key[SNEPPX_SIPHASH_KEY_SIZE]);
void SNEPPX_siphash_update(SNEPPXSipHash* sh, const uint8_t* data, size_t len);
uint64_t SNEPPX_siphash_finalize(SNEPPXSipHash* sh);
uint64_t SNEPPX_siphash(const uint8_t key[SNEPPX_SIPHASH_KEY_SIZE], const uint8_t* data, size_t len);

void SNEPPX_siphash_24_init(SNEPPXSipHash* sh, const uint8_t key[SNEPPX_SIPHASH_KEY_SIZE]);
uint64_t SNEPPX_siphash_24(const uint8_t key[SNEPPX_SIPHASH_KEY_SIZE], const uint8_t* data, size_t len);

#ifdef __cplusplus
}
#endif
#endif
