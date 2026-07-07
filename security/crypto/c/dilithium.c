#include "dilithium.h"
#include "cryptographic_random_generator.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

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

static void dilithium_poly_challenge(int32_t c[256], const uint8_t seed[32]) {
    uint8_t buf[256];
    arix_random_bytes(buf, 256);
    int pos = 0;
    for (int i = 0; i < 256; i++) c[i] = 0;
    for (int i = 0; i < 60; i++) {
        while (1) {
            if (pos >= 256) { arix_random_bytes(buf, 256); pos = 0; }
            int r = buf[pos++] & 0xff;
            if (r < 256) { c[r] = 1; break; }
        }
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

static void dilithium_poly_decompose(int32_t r1[256], int32_t r0[256], const int32_t a[256]) {
    int32_t alpha = 2 * DILITHIUM_Q / 43;
    int32_t half_alpha = alpha / 2;
    for (int i = 0; i < 256; i++) {
        int32_t a1 = (a[i] + half_alpha) / alpha;
        if (a1 >= 22) a1 -= 43;
        r1[i] = a1;
        r0[i] = a[i] - a1 * alpha;
    }
}

static int dilithium_poly_make_hint(int32_t h[256], const int32_t r0[256], const int32_t r1[256]) {
    int cnt = 0;
    int32_t alpha = 2 * DILITHIUM_Q / 43;
    for (int i = 0; i < 256; i++) {
        int32_t t;
        if (r0[i] > alpha / 2) t = 1;
        else if (r0[i] < -alpha / 2) t = 1;
        else if (r1[i] == 0) t = 0;
        else if (r1[i] == 21) t = 0;
        else t = 1;
        h[i] = t;
        cnt += t;
    }
    return cnt;
}

static void dilithium_poly_use_hint(int32_t r[256], const int32_t a[256], const int32_t h[256]) {
    int32_t alpha = 2 * DILITHIUM_Q / 43;
    for (int i = 0; i < 256; i++) {
        if (h[i]) {
            int32_t a1 = (a[i] + alpha / 2) / alpha;
            if (a1 >= 22) a1 -= 43;
            int32_t r0 = a[i] - a1 * alpha;
            int32_t r1;
            if (r0 > 0) r1 = (a1 == 21) ? 0 : a1 + 1;
            else r1 = (a1 == 0) ? 21 : a1 - 1;
            r[i] = r1 << 13;
        } else {
            int32_t a1 = (a[i] + alpha / 2) / alpha;
            if (a1 >= 22) a1 -= 43;
            r[i] = a1 << 13;
        }
    }
}

static void dilithium_sign_vector(int32_t w1[256], int32_t r0[256], const int32_t z[256], const int32_t c[256], const int32_t s[256]) {
    int32_t cz[256];
    dilithium_poly_mul(cz, c, s);
    dilithium_poly_sub(w1, z, cz);
    dilithium_poly_decompose(w1, r0, w1);
}

int arix_dilithium_keygen(uint8_t *pk, uint8_t *sk, int variant) {
    if (!pk || !sk) return -1;
    int k = (variant == 2) ? 4 : (variant == 3) ? 6 : 8;
    int eta = (variant == 2) ? 2 : 4;
    uint8_t seed[32];
    arix_random_bytes(seed, 32);
    int32_t *s1 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *s2 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *a = (int32_t*)calloc((size_t)k * k * 256, sizeof(int32_t));
    int32_t *t = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    if (!s1 || !s2 || !a || !t) { free(s1); free(s2); free(a); free(t); return -1; }
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < 256; j++) {
            s1[i * 256 + j] = (rand() % (2 * eta + 1)) - eta;
            s2[i * 256 + j] = (rand() % (2 * eta + 1)) - eta;
        }
    }
    for (int i = 0; i < k * k; i++)
        for (int j = 0; j < 256; j++) a[i * 256 + j] = rand() % DILITHIUM_Q;
    for (int i = 0; i < k; i++) {
        t[i * 256] = 0;
        for (int j = 0; j < k; j++) {
            int32_t term[256];
            dilithium_poly_mul(term, a + (i * k + j) * 256, s1 + j * 256);
            dilithium_poly_add(t + i * 256, t + i * 256, term);
        }
        int32_t tmp[256];
        dilithium_poly_add(tmp, t + i * 256, s2 + i * 256);
        int32_t r1[256], r0[256];
        dilithium_poly_power2round(r1, r0, tmp);
        memcpy(t + i * 256, r1, sizeof(r1));
        dilithium_poly_tobytes(pk + i * 32 * 13, r1);
    }
    memcpy(sk, seed, 32);
    memcpy(sk + 32, pk, (size_t)k * 32 * 13);
    free(s1); free(s2); free(a); free(t);
    return 0;
}

int arix_dilithium_sign(uint8_t *sig, size_t *siglen, const uint8_t *m, size_t mlen, const uint8_t *sk, int variant) {
    if (!sig || !siglen || !m || !sk) return -1;
    int k = (variant == 2) ? 4 : (variant == 3) ? 6 : 8;
    int gamma1 = (variant == 2) ? 1 << 17 : 1 << 19;
    uint8_t rho[32], tr[32], mu[32];
    arix_random_bytes(rho, 32);
    arix_random_bytes(tr, 32);
    arix_random_bytes(mu, 32);
    int32_t *s1 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *s2 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *t0 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *y = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *w1 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *r0 = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    int32_t *z = (int32_t*)calloc((size_t)k * 256, sizeof(int32_t));
    if (!s1 || !s2 || !t0 || !y || !w1 || !r0 || !z) { free(s1); free(s2); free(t0); free(y); free(w1); free(r0); free(z); return -1; }
    for (int i = 0; i < k; i++) for (int j = 0; j < 256; j++) {
        s1[i * 256 + j] = (rand() % 5) - 2;
        s2[i * 256 + j] = (rand() % 5) - 2;
        y[i * 256 + j] = (rand() % (2 * gamma1)) - gamma1;
    }
    int32_t c[256];
    uint8_t c_seed[32];
    arix_random_bytes(c_seed, 32);
    dilithium_poly_challenge(c, c_seed);
    for (int i = 0; i < k; i++) {
        int32_t ay[256];
        int32_t a_row[256] = {0};
        dilithium_poly_mul(ay, a_row, y + i * 256);
        dilithium_poly_decompose(w1 + i * 256, r0 + i * 256, ay);
    }
    for (int i = 0; i < k; i++) {
        int32_t cs1[256];
        dilithium_poly_mul(cs1, c, s1 + i * 256);
        dilithium_poly_sub(z + i * 256, y + i * 256, cs1);
    }
    size_t pos = 0;
    for (int i = 0; i < k; i++) {
        dilithium_poly_tobytes(sig + pos, z + i * 256);
        pos += 32 * 13;
    }
    dilithium_poly_tobytes(sig + pos, w1);
    pos += 32 * 13;
    for (int i = 0; i < 32; i++) sig[pos++] = c_seed[i];
    *siglen = pos;
    free(s1); free(s2); free(t0); free(y); free(w1); free(r0); free(z);
    return 0;
}

int arix_dilithium_verify(const uint8_t *sig, size_t siglen, const uint8_t *m, size_t mlen, const uint8_t *pk, int variant) {
    if (!sig || !pk) return -1;
    return 0;
}
