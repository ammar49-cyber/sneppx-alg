#include "dilithium.h"
#include "cryptographic_random_generator.h"
#include "drbg.h"
#include "sha512_hashing_implementation.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>

#define DILITHIUM_N 256
#define DILITHIUM_Q 8380417
#define DILITHIUM_ROOT 1753

static int16_t dilithium_zetas[256];
static int init_ntt_d = 0;

static int32_t dilithium_mont_reduce(int64_t a) {
    int64_t t = (int64_t)((int32_t)a * 58728449);
    t = (a - t * 8380417) >> 32;
    return (int32_t)t;
}

static int32_t dilithium_reduce(int32_t a) {
    int32_t t = (a + (1 << 22)) >> 23;
    t = a - t * 8380417;
    return t;
}

static void dilithium_ntt(int32_t a[256]) {
    if (!init_ntt_d) {
        int32_t z = DILITHIUM_ROOT, z2 = (z * z) % DILITHIUM_Q;
        dilithium_zetas[0] = 1;
        for (int i = 1; i < 256; i++) dilithium_zetas[i] = (dilithium_zetas[i-1] * z2) % DILITHIUM_Q;
        init_ntt_d = 1;
    }
    int len = 128, k = 0;
    while (len >= 2) {
        int start = 0;
        while (start < 256) {
            int32_t zeta = dilithium_zetas[++k];
            for (int j = start; j < start + len; j++) {
                int32_t t = dilithium_mont_reduce((int64_t)zeta * a[j + len]);
                a[j + len] = a[j] - t;
                a[j] = a[j] + t;
                if (a[j] >= DILITHIUM_Q) a[j] -= DILITHIUM_Q;
                if (a[j + len] >= DILITHIUM_Q) a[j + len] -= DILITHIUM_Q;
            }
            start += len * 2;
        }
        len >>= 1;
    }
}

static void dilithium_inv_ntt(int32_t a[256]) {
    int len = 2, k = 255;
    while (len <= 128) {
        int start = 0;
        while (start < 256) {
            int32_t zeta = -dilithium_zetas[--k];
            if (zeta < 0) zeta += DILITHIUM_Q;
            for (int j = start; j < start + len; j++) {
                int32_t t = a[j + len];
                a[j + len] = dilithium_reduce(a[j] - t);
                if (a[j + len] < 0) a[j + len] += DILITHIUM_Q;
                a[j] = dilithium_reduce(a[j] + t);
                if (a[j] < 0) a[j] += DILITHIUM_Q;
                int32_t tmp = dilithium_mont_reduce((int64_t)zeta * a[j + len]);
                a[j + len] = tmp;
            }
            start += len * 2;
        }
        len <<= 1;
    }
    int32_t inv_n = 8347681;
    for (int j = 0; j < 256; j++) a[j] = dilithium_mont_reduce((int64_t)a[j] * inv_n);
}

static void dilithium_poly_add(int32_t r[256], const int32_t a[256], const int32_t b[256]) {
    for (int i = 0; i < 256; i++) { r[i] = a[i] + b[i]; if (r[i] >= DILITHIUM_Q) r[i] -= DILITHIUM_Q; }
}

static void dilithium_poly_sub(int32_t r[256], const int32_t a[256], const int32_t b[256]) {
    for (int i = 0; i < 256; i++) { r[i] = a[i] - b[i]; if (r[i] < 0) r[i] += DILITHIUM_Q; }
}

static void dilithium_poly_mul(int32_t r[256], const int32_t a[256], const int32_t b[256]) {
    int32_t ta[256], tb[256];
    memcpy(ta, a, sizeof(ta)); dilithium_ntt(ta);
    memcpy(tb, b, sizeof(tb)); dilithium_ntt(tb);
    for (int i = 0; i < 256; i++) ta[i] = dilithium_mont_reduce((int64_t)ta[i] * tb[i]);
    dilithium_inv_ntt(ta);
    memcpy(r, ta, sizeof(ta));
}

static void dilithium_poly_mul_by_const(int32_t r[256], const int32_t a[256], int32_t c) {
    for (int i = 0; i < 256; i++) r[i] = (int32_t)(((int64_t)a[i] * c) % DILITHIUM_Q);
}

static void dilithium_poly_tobytes(uint8_t *out, const int32_t a[256]) {
    for (int i = 0; i < 64; i++) {
        int32_t t0 = a[4*i], t1 = a[4*i+1], t2 = a[4*i+2], t3 = a[4*i+3];
        out[13*i] = t0 & 0xff;
        out[13*i+1] = (t0 >> 8) & 0xff;
        out[13*i+2] = (t0 >> 16) | ((t1 & 0x1f) << 5);
        out[13*i+3] = (t1 >> 5) & 0xff;
        out[13*i+4] = (t1 >> 13) & 0xff;
        out[13*i+5] = (t1 >> 21) | ((t2 & 0x03) << 3);
        out[13*i+6] = (t2 >> 2) & 0xff;
        out[13*i+7] = (t2 >> 10) & 0xff;
        out[13*i+8] = (t2 >> 18) | ((t3 & 0x0f) << 2);
        out[13*i+9] = (t3 >> 2) & 0xff;
        out[13*i+10] = (t3 >> 10) & 0xff;
        out[13*i+11] = (t3 >> 18) & 0xff;
        out[13*i+12] = (t3 >> 26) & 0xff;
    }
}

static int dilithium_poly_frombytes(int32_t r[256], const uint8_t *in) {
    for (int i = 0; i < 64; i++) {
        r[4*i] = in[13*i] | ((int32_t)in[13*i+1] << 8) | ((int32_t)(in[13*i+2] & 0x1f) << 16);
        r[4*i+1] = (in[13*i+2] >> 5) | ((int32_t)in[13*i+3] << 3) | ((int32_t)in[13*i+4] << 11) | ((int32_t)(in[13*i+5] & 0x03) << 19);
        r[4*i+2] = (in[13*i+5] >> 2) | ((int32_t)in[13*i+6] << 6) | ((int32_t)in[13*i+7] << 14) | ((int32_t)(in[13*i+8] & 0x0f) << 22);
        r[4*i+3] = (in[13*i+8] >> 4) | ((int32_t)in[13*i+9] << 4) | ((int32_t)in[13*i+10] << 12) | ((int32_t)in[13*i+11] << 20) | ((int32_t)in[13*i+12] << 28);
        for (int j = 0; j < 4; j++) if (r[4*i+j] >= DILITHIUM_Q) r[4*i+j] -= DILITHIUM_Q;
    }
    return 0;
}

static void pack_10bit(uint8_t *out, const int32_t in[256]) {
    uint16_t buf[4];
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 4; j++) buf[j] = (uint16_t)(in[4*i + j] & 0x3FF);
        uint64_t v = 0;
        for (int j = 0; j < 4; j++) v |= (uint64_t)buf[j] << (10 * j);
        for (int j = 0; j < 5; j++) out[5*i + j] = (v >> (8 * j)) & 0xFF;
    }
}

static void unpack_10bit(int32_t out[256], const uint8_t *in) {
    for (int i = 0; i < 64; i++) {
        uint64_t v = 0;
        for (int j = 0; j < 5; j++) v |= (uint64_t)in[5*i + j] << (8 * j);
        for (int j = 0; j < 4; j++) out[4*i + j] = (int32_t)((v >> (10 * j)) & 0x3FF);
    }
}

static void pack_3bit_signed(uint8_t *out, const int32_t in[256]) {
    uint64_t buf = 0;
    int bits = 0, pos = 0;
    for (int i = 0; i < 256; i++) {
        int val = in[i] + 2;
        buf |= (uint64_t)(val & 7) << bits;
        bits += 3;
        while (bits >= 8) { out[pos++] = buf & 0xFF; buf >>= 8; bits -= 8; }
    }
    while (bits > 0) { out[pos++] = buf & 0xFF; buf >>= 8; bits -= 8; }
}

static void unpack_3bit_signed(int32_t out[256], const uint8_t *in) {
    uint64_t buf = 0;
    int bits = 0, pos = 0;
    for (int i = 0; i < 256; i++) {
        while (bits < 3) { buf |= (uint64_t)in[pos++] << bits; bits += 8; }
        out[i] = (int32_t)(buf & 7) - 2;
        buf >>= 3;
        bits -= 3;
    }
}

static void pack_13bit_signed(uint8_t *out, const int32_t in[256]) {
    uint64_t buf = 0;
    int bits = 0, pos = 0;
    for (int i = 0; i < 256; i++) {
        int val = in[i] & 0x1FFF;
        buf |= (uint64_t)val << bits;
        bits += 13;
        while (bits >= 8) { out[pos++] = buf & 0xFF; buf >>= 8; bits -= 8; }
    }
    while (bits > 0) { out[pos++] = buf & 0xFF; buf >>= 8; bits -= 8; }
}

static void unpack_13bit_signed(int32_t out[256], const uint8_t *in) {
    uint64_t buf = 0;
    int bits = 0, pos = 0;
    for (int i = 0; i < 256; i++) {
        while (bits < 13) { buf |= (uint64_t)in[pos++] << bits; bits += 8; }
        out[i] = (int32_t)(buf & 0x1FFF);
        if (out[i] >= 4096) out[i] -= 8192;
        buf >>= 13;
        bits -= 13;
    }
}

static void pack_z(uint8_t *out, const int32_t in[256], int gamma1) {
    int bound = 2 * gamma1;
    int bits = 0;
    bits = 0;
    while ((1 << bits) < bound) bits++;
    uint64_t buf = 0;
    int bitpos = 0, pos = 0;
    for (int i = 0; i < 256; i++) {
        int val = in[i] + gamma1;
        buf |= (uint64_t)(val & ((1 << bits) - 1)) << bitpos;
        bitpos += bits;
        while (bitpos >= 8) { out[pos++] = buf & 0xFF; buf >>= 8; bitpos -= 8; }
    }
    while (bitpos > 0) { out[pos++] = buf & 0xFF; buf >>= 8; bitpos -= 8; }
}

static void unpack_z(int32_t out[256], const uint8_t *in, int gamma1) {
    int bound = 2 * gamma1;
    int bits = 0;
    while ((1 << bits) < bound) bits++;
    uint64_t buf = 0;
    int bitpos = 0, pos = 0;
    for (int i = 0; i < 256; i++) {
        while (bitpos < bits) { buf |= (uint64_t)in[pos++] << bitpos; bitpos += 8; }
        int val = buf & ((1 << bits) - 1);
        out[i] = val - gamma1;
        buf >>= bits;
        bitpos -= bits;
    }
}

static void dilithium_poly_power2round(int32_t r1[256], int32_t r0[256], const int32_t a[256]) {
    int32_t half_q = DILITHIUM_Q / 2;
    for (int i = 0; i < 256; i++) {
        int32_t a1 = (a[i] + half_q) >> 13;
        r1[i] = a1;
        r0[i] = a[i] - (a1 << 13);
    }
}

static int32_t dilithium_gamma2(int variant) {
    return (variant == 2) ? (DILITHIUM_Q - 1) / 88 : (variant == 3) ? (DILITHIUM_Q - 1) / 32 : (DILITHIUM_Q - 1) / 22;
}

static void dilithium_poly_decompose(int32_t r1[256], int32_t r0[256], const int32_t a[256], int32_t alpha) {
    int32_t half_alpha = alpha / 2;
    for (int i = 0; i < 256; i++) {
        int32_t a1 = (a[i] + half_alpha) / alpha;
        if (a1 >= 22) a1 -= 43;
        r1[i] = a1;
        r0[i] = a[i] - a1 * alpha;
    }
}

static int dilithium_poly_make_hint(int32_t h[256], const int32_t r0[256], const int32_t r1[256], int32_t alpha) {
    int cnt = 0;
    int32_t half_alpha = alpha / 2;
    for (int i = 0; i < 256; i++) {
        int32_t t;
        if (r0[i] > half_alpha) t = 1;
        else if (r0[i] < -half_alpha) t = 1;
        else if (r1[i] == 0) t = 0;
        else if (r1[i] == 21) t = 0;
        else t = 1;
        h[i] = t;
        cnt += t;
    }
    return cnt;
}

static void dilithium_poly_use_hint(int32_t r[256], const int32_t a[256], const int32_t h[256], int32_t alpha) {
    int32_t half_alpha = alpha / 2;
    for (int i = 0; i < 256; i++) {
        if (h[i]) {
            int32_t a1 = (a[i] + half_alpha) / alpha;
            if (a1 >= 22) a1 -= 43;
            int32_t r0v = a[i] - a1 * alpha;
            int32_t r1v;
            if (r0v > 0) r1v = (a1 == 21) ? 0 : a1 + 1;
            else r1v = (a1 == 0) ? 21 : a1 - 1;
            r[i] = r1v;
        } else {
            int32_t a1 = (a[i] + half_alpha) / alpha;
            if (a1 >= 22) a1 -= 43;
            r[i] = a1;
        }
    }
}

static void cbd_eta2(int32_t r[256], const uint8_t seed[32], uint8_t nonce) {
    uint8_t buf[128];
    SNEPPXDRBG drbg;
    SNEPPX_drbg_init(&drbg, seed, 32, &nonce, 1);
    SNEPPX_drbg_generate(&drbg, buf, 128);
    SNEPPX_drbg_destroy(&drbg);
    for (int i = 0; i < 256; i++) {
        int a = (buf[i/4] >> (2*(i%4))) & 3;
        int b = (buf[i/4 + 64] >> (2*(i%4))) & 3;
        int a_bits = (a & 1) + ((a >> 1) & 1);
        int b_bits = (b & 1) + ((b >> 1) & 1);
        r[i] = a_bits - b_bits;
    }
}

static void cbd_eta4(int32_t r[256], const uint8_t seed[32], uint8_t nonce) {
    uint8_t buf[256];
    SNEPPXDRBG drbg;
    SNEPPX_drbg_init(&drbg, seed, 32, &nonce, 1);
    SNEPPX_drbg_generate(&drbg, buf, 256);
    SNEPPX_drbg_destroy(&drbg);
    for (int i = 0; i < 256; i++) {
        int a = (buf[i/2] >> (4*(i%2))) & 15;
        int b = (buf[i/2 + 128] >> (4*(i%2))) & 15;
        int a_bits = (a & 1) + ((a>>1)&1) + ((a>>2)&1) + ((a>>3)&1);
        int b_bits = (b & 1) + ((b>>1)&1) + ((b>>2)&1) + ((b>>3)&1);
        r[i] = a_bits - b_bits;
    }
}

static void dilithium_expand_a(int32_t *a, const uint8_t rho[32], int k) {
    int total = k * k * 256;
    int bytes_needed = total * 3;
    uint8_t *buf = (uint8_t*)malloc(bytes_needed);
    if (!buf) return;
    SNEPPXDRBG drbg;
    SNEPPX_drbg_init(&drbg, rho, 32, NULL, 0);
    SNEPPX_drbg_generate(&drbg, buf, bytes_needed);
    SNEPPX_drbg_destroy(&drbg);
    for (int i = 0; i < total; i++) {
        a[i] = (int32_t)(buf[3*i] | ((uint32_t)buf[3*i+1] << 8) | ((uint32_t)buf[3*i+2] << 16));
        if (a[i] >= DILITHIUM_Q) a[i] %= DILITHIUM_Q;
    }
    free(buf);
}

static void dilithium_poly_challenge(int32_t c[256], const uint8_t seed[32]) {
    uint8_t buf[640];
    SNEPPXDRBG drbg;
    SNEPPX_drbg_init(&drbg, seed, 32, NULL, 0);
    SNEPPX_drbg_generate(&drbg, buf, sizeof(buf));
    SNEPPX_drbg_destroy(&drbg);
    memset(c, 0, 256 * sizeof(int32_t));
    int pos = 0, found = 0;
    while (found < 60 && pos + 2 <= (int)sizeof(buf)) {
        int idx = buf[pos] % 256;
        int sign = (buf[pos + 1] & 1) ? 1 : -1;
        pos += 2;
        if (c[idx] == 0) { c[idx] = sign; found++; }
    }
}

static void dilithium_mvp(int32_t *out, const int32_t *a, const int32_t *vec, int k) {
    memset(out, 0, k * 256 * sizeof(int32_t));
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            int32_t term[256];
            dilithium_poly_mul(term, a + (i * k + j) * 256, vec + j * 256);
            dilithium_poly_add(out + i * 256, out + i * 256, term);
        }
    }
}

int SNEPPX_dilithium_keygen(uint8_t *pk, uint8_t *sk, int variant) {
    if (!pk || !sk) return -1;
    int k = (variant == 2) ? 4 : (variant == 3) ? 6 : 8;
    int eta = (variant == 2) ? 2 : 4;
    uint8_t rho[32], rho_prime[32];
    SNEPPX_random_bytes(rho, 32);
    SNEPPX_random_bytes(rho_prime, 32);
    int32_t *a = (int32_t*)calloc((size_t)k * k * 256, sizeof(int32_t));
    int32_t *s1 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *s2 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *t = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *t1 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *t0 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    if (!a || !s1 || !s2 || !t || !t1 || !t0) {
        free(a); free(s1); free(s2); free(t); free(t1); free(t0);
        return -1;
    }
    dilithium_expand_a(a, rho, k);
    for (int i = 0; i < k; i++) {
        if (eta == 2) {
            cbd_eta2(s1 + i * 256, rho_prime, i);
            cbd_eta2(s2 + i * 256, rho_prime, k + i);
        } else {
            cbd_eta4(s1 + i * 256, rho_prime, i);
            cbd_eta4(s2 + i * 256, rho_prime, k + i);
        }
    }
    dilithium_mvp(t, a, s1, k);
    for (int i = 0; i < k; i++) dilithium_poly_add(t + i * 256, t + i * 256, s2 + i * 256);
    for (int i = 0; i < k; i++) {
        dilithium_poly_power2round(t1 + i * 256, t0 + i * 256, t + i * 256);
    }
    memcpy(pk, rho, 32);
    for (int i = 0; i < k; i++) pack_10bit(pk + 32 + i * 320, t1 + i * 256);
    int sk_pos = 0;
    memcpy(sk + sk_pos, rho, 32); sk_pos += 32;
    memcpy(sk + sk_pos, rho_prime, 32); sk_pos += 32;
    memset(sk + sk_pos, 0, 32); sk_pos += 32;
    for (int i = 0; i < k; i++) pack_3bit_signed(sk + sk_pos + i * 96, s1 + i * 256);
    sk_pos += k * 96;
    for (int i = 0; i < k; i++) pack_3bit_signed(sk + sk_pos + i * 96, s2 + i * 256);
    sk_pos += k * 96;
    for (int i = 0; i < k; i++) pack_13bit_signed(sk + sk_pos + i * 416, t0 + i * 256);
    sk_pos += k * 416;

    free(a); free(s1); free(s2); free(t); free(t1); free(t0);
    return 0;
}

int SNEPPX_dilithium_sign(uint8_t *sig, size_t *siglen, const uint8_t *m, size_t mlen, const uint8_t *sk, int variant) {
    if (!sig || !siglen || !m || !sk) return -1;
    int k = (variant == 2) ? 4 : (variant == 3) ? 6 : 8;
    int eta = (variant == 2) ? 2 : 4;
    int gamma1 = (variant == 2) ? 1 << 17 : 1 << 19;
    int gamma2 = dilithium_gamma2(variant);
    int alpha = 2 * gamma2;
    int beta = 60 * eta;
    uint8_t rho[32], rho_prime[32];
    memcpy(rho, sk, 32);
    memcpy(rho_prime, sk + 32, 32);
    int32_t *a = (int32_t*)calloc((size_t)k * k * 256, sizeof(int32_t));
    int32_t *s1 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *s2 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *t0 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *y = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *w = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *w1 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *w0 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *z = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *cs1 = (int32_t*)calloc(256, sizeof(int32_t));
    int32_t *cs2 = (int32_t*)calloc(256, sizeof(int32_t));
    int32_t *ct0 = (int32_t*)calloc(256, sizeof(int32_t));
    int32_t *h = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *hint_in = (int32_t*)calloc(256, sizeof(int32_t));
    if (!a || !s1 || !s2 || !t0 || !y || !w || !w1 || !w0 || !z || !cs1 || !cs2 || !ct0 || !h || !hint_in) {
        free(a); free(s1); free(s2); free(t0); free(y); free(w); free(w1); free(w0); free(z);
        free(cs1); free(cs2); free(ct0); free(h); free(hint_in);
        return -1;
    }
    int sk_pos = 96;
    for (int i = 0; i < k; i++) unpack_3bit_signed(s1 + i * 256, sk + sk_pos + i * 96);
    sk_pos += k * 96;
    for (int i = 0; i < k; i++) unpack_3bit_signed(s2 + i * 256, sk + sk_pos + i * 96);
    sk_pos += k * 96;
    for (int i = 0; i < k; i++) unpack_13bit_signed(t0 + i * 256, sk + sk_pos + i * 416);
    dilithium_expand_a(a, rho, k);
    uint8_t c_seed[32];
    int32_t c_poly[256];
    int reject;
    int max_attempts = 100;
    for (int attempt = 0; attempt < max_attempts; attempt++) {
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < 256; j++) {
                uint32_t r;
                SNEPPX_random_bytes((uint8_t*)&r, 4);
                r %= (uint32_t)(2 * gamma1);
                y[i * 256 + j] = (int32_t)r - gamma1;
            }
        }
        dilithium_mvp(w, a, y, k);
        memset(w1, 0, k * 256 * sizeof(int32_t));
        memset(w0, 0, k * 256 * sizeof(int32_t));
        for (int i = 0; i < k; i++)
            dilithium_poly_decompose(w1 + i * 256, w0 + i * 256, w + i * 256, alpha);
        {
            uint8_t *w1_enc = (uint8_t*)calloc((size_t)k * 416, 1);
        if (!w1_enc) { reject = 1; continue; }
            for (int i = 0; i < k; i++) dilithium_poly_tobytes(w1_enc + i * 416, w1 + i * 256);
            uint8_t hash_buf[64];
            SNEPPXSHA512Context ctx;
            SNEPPX_sha512_init(&ctx);
            SNEPPX_sha512_update(&ctx, m, mlen);
            SNEPPX_sha512_update(&ctx, w1_enc, (size_t)k * 416);
            SNEPPX_sha512_finish(&ctx, hash_buf);
            memcpy(c_seed, hash_buf, 32);
            free(w1_enc);
        }
        dilithium_poly_challenge(c_poly, c_seed);
        reject = 0;
        for (int i = 0; i < k; i++) {
            memset(cs1, 0, 256 * sizeof(int32_t));
            dilithium_poly_mul(cs1, c_poly, s1 + i * 256);
            for (int j = 0; j < 256; j++) {
                z[i * 256 + j] = y[i * 256 + j] + cs1[j];
                if (z[i * 256 + j] > gamma1 - beta || z[i * 256 + j] < -(gamma1 - beta))
                    reject = 1;
            }
        }
        if (reject) continue;
        int w0_reject = 0;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < 256; j++) {
                if (w0[i * 256 + j] > alpha / 2 || w0[i * 256 + j] < -(alpha / 2))
                    w0_reject = 1;
            }
        }
        if (w0_reject) continue;
        int r0_reject = 0;
        for (int i = 0; i < k; i++) {
            memset(cs2, 0, 256 * sizeof(int32_t));
            dilithium_poly_mul(cs2, c_poly, s2 + i * 256);
            int32_t r0poly[256];
            for (int j = 0; j < 256; j++) r0poly[j] = w0[i * 256 + j] - cs2[j];
            for (int j = 0; j < 256; j++) {
                if (r0poly[j] > alpha / 2 || r0poly[j] < -(alpha / 2))
                    r0_reject = 1;
            }
        }
        if (r0_reject) continue;
        for (int i = 0; i < k; i++) {
            memset(ct0, 0, 256 * sizeof(int32_t));
            dilithium_poly_mul(ct0, c_poly, t0 + i * 256);
            for (int j = 0; j < 256; j++) hint_in[j] = -ct0[j];
            memset(cs2, 0, 256 * sizeof(int32_t));
            dilithium_poly_mul(cs2, c_poly, s2 + i * 256);
            int32_t r1poly[256];
            for (int j = 0; j < 256; j++) r1poly[j] = w0[i * 256 + j] - cs2[j] + ct0[j];
            dilithium_poly_make_hint(h + i * 256, hint_in, r1poly, alpha);
        }
        size_t pos = 0;
        for (int i = 0; i < k; i++) {
            pack_z(sig + pos, z + i * 256, gamma1);
            int z_bits = 0; while ((1 << z_bits) < 2 * gamma1) z_bits++;
            pos += (256 * z_bits + 7) / 8;
        }
        for (int i = 0; i < k; i++) {
            uint8_t hint_byte = 0;
            int hbits = 0;
            for (int j = 0; j < 256; j++) {
                hint_byte |= (uint8_t)(h[i * 256 + j] & 1) << hbits;
                hbits++;
                if (hbits == 8) { sig[pos++] = hint_byte; hint_byte = 0; hbits = 0; }
            }
            if (hbits > 0) sig[pos++] = hint_byte;
        }
        memcpy(sig + pos, c_seed, 32);
        pos += 32;
        *siglen = pos;
        free(a); free(s1); free(s2); free(t0); free(y); free(w); free(w1); free(w0); free(z);
        free(cs1); free(cs2); free(ct0); free(h); free(hint_in);
        return 0;
    }
    free(a); free(s1); free(s2); free(t0); free(y); free(w); free(w1); free(w0); free(z);
    free(cs1); free(cs2); free(ct0); free(h); free(hint_in);
    return -1;
}

int SNEPPX_dilithium_verify(const uint8_t *sig, size_t siglen, const uint8_t *m, size_t mlen, const uint8_t *pk, int variant) {
    if (!sig || !pk) return -1;
    int k = (variant == 2) ? 4 : (variant == 3) ? 6 : 8;
    int eta = (variant == 2) ? 2 : 4;
    int gamma1 = (variant == 2) ? 1 << 17 : 1 << 19;
    int gamma2 = dilithium_gamma2(variant);
    int alpha = 2 * gamma2;
    int beta = 60 * eta;
    int z_bits = 0;
    while ((1 << z_bits) < 2 * gamma1) z_bits++;
    size_t z_bytes = (256 * z_bits + 7) / 8;
    size_t hint_bytes = (k * 256 + 7) / 8;
    size_t expected_siglen = k * z_bytes + hint_bytes + 32;
    if (siglen < expected_siglen) return -1;
    uint8_t rho[32];
    memcpy(rho, pk, 32);
    int32_t *a = (int32_t*)calloc((size_t)k * k * 256, sizeof(int32_t));
    int32_t *t1 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *zvec = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *h = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *az = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *w1 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *ct1_shifted = (int32_t*)calloc(256, sizeof(int32_t));
    if (!a || !t1 || !zvec || !h || !az || !w1 || !ct1_shifted) {
        free(a); free(t1); free(zvec); free(h); free(az); free(w1); free(ct1_shifted);
        return -1;
    }
    dilithium_expand_a(a, rho, k);
    for (int i = 0; i < k; i++) unpack_10bit(t1 + i * 256, pk + 32 + i * 320);
    size_t pos = 0;
    for (int i = 0; i < k; i++) {
        unpack_z(zvec + i * 256, sig + pos, gamma1);
        pos += z_bytes;
    }
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < 256; j++) {
            int byte_idx = (int)(pos + (i * 256 + j) / 8);
            int bit_idx = (int)((i * 256 + j) % 8);
            if ((size_t)byte_idx < siglen)
                h[i * 256 + j] = (sig[byte_idx] >> bit_idx) & 1;
        }
    }
    pos += hint_bytes;
    uint8_t c_seed[32];
    memcpy(c_seed, sig + pos, 32);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < 256; j++) {
            int32_t c = zvec[i * 256 + j];
            if (c > gamma1 - beta || c < -(gamma1 - beta)) {
                free(a); free(t1); free(zvec); free(h); free(az); free(w1); free(ct1_shifted);
                return -1;
            }
        }
    }
    int32_t c_poly[256];
    dilithium_poly_challenge(c_poly, c_seed);
    memset(az, 0, k * 256 * sizeof(int32_t));
    dilithium_mvp(az, a, zvec, k);
    for (int i = 0; i < k; i++) {
        memset(ct1_shifted, 0, 256 * sizeof(int32_t));
        dilithium_poly_mul(ct1_shifted, c_poly, t1 + i * 256);
        for (int j = 0; j < 256; j++) ct1_shifted[j] = (int32_t)(((int64_t)ct1_shifted[j] * 8192) % DILITHIUM_Q);
        int32_t diff[256];
        for (int j = 0; j < 256; j++) {
            diff[j] = az[i * 256 + j] - ct1_shifted[j];
            diff[j] = (diff[j] % DILITHIUM_Q + DILITHIUM_Q) % DILITHIUM_Q;
        }
        dilithium_poly_use_hint(w1 + i * 256, diff, h + i * 256, alpha);
    }
    uint8_t *w1_enc = (uint8_t*)calloc((size_t)k * 416, 1);
    if (!w1_enc) { free(a); free(t1); free(zvec); free(h); free(az); free(w1); free(ct1_shifted); return -1; }
    for (int i = 0; i < k; i++) dilithium_poly_tobytes(w1_enc + i * 416, w1 + i * 256);
    uint8_t hash_buf[64];
    SNEPPXSHA512Context ctx;
    SNEPPX_sha512_init(&ctx);
    SNEPPX_sha512_update(&ctx, m, mlen);
    SNEPPX_sha512_update(&ctx, w1_enc, (size_t)k * 416);
    SNEPPX_sha512_finish(&ctx, hash_buf);
    free(w1_enc);
    if (memcmp(c_seed, hash_buf, 32) != 0) {
        free(a); free(t1); free(zvec); free(h); free(az); free(w1); free(ct1_shifted);
        return -1;
    }
    free(a); free(t1); free(zvec); free(h); free(az); free(w1); free(ct1_shifted);
    return 0;
}
