#ifndef SNEPPX_DRBG_H
#define SNEPPX_DRBG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_DRBG_MAX_OUTPUT 65536
#define SNEPPX_DRBG_SEED_SIZE 48

typedef struct {
    uint8_t v[SNEPPX_DRBG_SEED_SIZE];
    uint8_t c[SNEPPX_DRBG_SEED_SIZE];
    uint64_t reseed_counter;
    int security_strength;
    int initialized;
} SNEPPXHashDRBG;

typedef struct {
    SNEPPXHashDRBG hb;
    int use_hmac;
} SNEPPXDRBG;

int  SNEPPX_drbg_init(SNEPPXDRBG* ctx, const uint8_t* entropy, size_t entropy_len, const uint8_t* nonce, size_t nonce_len);
int  SNEPPX_drbg_reseed(SNEPPXDRBG* ctx, const uint8_t* entropy, size_t entropy_len);
int  SNEPPX_drbg_generate(SNEPPXDRBG* ctx, uint8_t* out, size_t out_len);
void SNEPPX_drbg_destroy(SNEPPXDRBG* ctx);
int  SNEPPX_drbg_self_test(void);

#ifdef __cplusplus
}
#endif
#endif
