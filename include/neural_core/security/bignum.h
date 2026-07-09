#ifndef SNEPPX_BIGNUM_H
#define SNEPPX_BIGNUM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_BN_MAX_WORDS 128
#define SNEPPX_BN_WORD uint64_t
#define SNEPPX_BN_HALF_WORD uint32_t

typedef struct {
    SNEPPX_BN_WORD words[SNEPPX_BN_MAX_WORDS];
    int used;
    int sign;
} SNEPPXBigNum;

void SNEPPX_bn_init(SNEPPXBigNum* bn);
void SNEPPX_bn_zero(SNEPPXBigNum* bn);
int  SNEPPX_bn_set_word(SNEPPXBigNum* bn, SNEPPX_BN_WORD val);
int  SNEPPX_bn_set_array(SNEPPXBigNum* bn, const uint8_t* bytes, size_t len);
int  SNEPPX_bn_from_hex(SNEPPXBigNum* bn, const char* hex);
void SNEPPX_bn_to_array(const SNEPPXBigNum* bn, uint8_t* out, size_t* out_len);

int  SNEPPX_bn_copy(SNEPPXBigNum* dst, const SNEPPXBigNum* src);
int  SNEPPX_bn_is_zero(const SNEPPXBigNum* bn);
int  SNEPPX_bn_is_one(const SNEPPXBigNum* bn);
int  SNEPPX_bn_cmp(const SNEPPXBigNum* a, const SNEPPXBigNum* b);
int  SNEPPX_bn_cmp_word(const SNEPPXBigNum* a, SNEPPX_BN_WORD b);

int  SNEPPX_bn_add(SNEPPXBigNum* r, const SNEPPXBigNum* a, const SNEPPXBigNum* b);
int  SNEPPX_bn_sub(SNEPPXBigNum* r, const SNEPPXBigNum* a, const SNEPPXBigNum* b);
int  SNEPPX_bn_mul(SNEPPXBigNum* r, const SNEPPXBigNum* a, const SNEPPXBigNum* b);
int  SNEPPX_bn_div(SNEPPXBigNum* q, SNEPPXBigNum* rem, const SNEPPXBigNum* a, const SNEPPXBigNum* b);
int  SNEPPX_bn_mod(SNEPPXBigNum* r, const SNEPPXBigNum* a, const SNEPPXBigNum* m);
int  SNEPPX_bn_exp_mod(SNEPPXBigNum* r, const SNEPPXBigNum* base, const SNEPPXBigNum* exp, const SNEPPXBigNum* mod);

int  SNEPPX_bn_gcd(SNEPPXBigNum* r, const SNEPPXBigNum* a, const SNEPPXBigNum* b);
int  SNEPPX_bn_inv_mod(SNEPPXBigNum* r, const SNEPPXBigNum* a, const SNEPPXBigNum* m);
int  SNEPPX_bn_is_prime(const SNEPPXBigNum* bn);
void SNEPPX_bn_print(const SNEPPXBigNum* bn);

#ifdef __cplusplus
}
#endif
#endif
