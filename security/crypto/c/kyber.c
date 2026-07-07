#include "kyber.h"
#include "cryptographic_random_generator.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#define KYBER_N 256
#define KYBER_Q 3329

static int16_t zetas[128];
static int16_t zetas_inv[128];
static int init_ntt = 0;

static const int16_t zetas_rom[] = {
    -1044, -758, -359, -1517, 1493, 1422, 287, 202, -171, 622, 1577, 182, 962,
    -1202, -1474, 1468, 573, -1325, 264, 383, -829, 1458, -1602, -130, -681,
    1017, 732, 608, -1542, 411, -205, -1571, 1223, 652, -552, 1015, -1293, 1491,
    -282, -1544, 516, -8, -320, -666, -1618, -1162, 126, 1469, -853, -90, -271,
    830, 107, -1421, -247, -951, -398, 961, -1508, -725, 448, -1065, 677, -1275,
    -1103, 430, 555, 843, -1251, 871, 1550, 105, 422, -587, -833, 258, 1331,
    1265, 1544, 184, -1607, -144, 214, -29, 1006, -1383, -5, 1647, -718, -720,
    1446, -1217, 713, -1345, -1151, 1511, -786, 1384, 320, -413, 1156, -403,
    1031, 233, -1550, 56, 1457, -1676, -959, 136, 1179, -1410, -807, 766, 1387,
    -1288, -1584, 1641, -1009, -55, -73, 887, 1343, 760, -765, -1546, -1582,
    1447, 1617, -1554, 1644, 1605, 374, -1060, 1413, -565, -437, 1404, -1038,
    386, -242, -560, 297, 1145, 290, 184, 718, -1020, 248, 1078, 211, 1016, -450,
    -381, -372, -1080, 1012, 1308, 1497, -799, -1055, 764, -621, -475, -1056,
    -512, -1614, 1247, 1472, -788, -1207, -671, 614, 543, 428, -1520, -355, 908,
    -1406, 1455, 810, -926, 908, 1420, -640, -1332, -1395, 506, -499, -384, 526,
    -833, -170, 860, 410, -1375, 1025, -1386, -1589, 559, 1629, 810, 730, -527,
    1220, 629, 1629, 1158, -160, 1650, 1435, -124, -247, 895, -469, 1611, 1096,
    -1183, -247, 550, -1626, 143, -1586, -32, 1011, -706, -234, 943, 378, 364,
    -558, -1602, 1637, 1335, -1437, -1390, -948, -505, 1106, 766, -648, 262,
    1509, -870, 1428, -325, -746, 1494, 1130, -1210, -326, -744, -877, 1577,
    -997, 85, -1306, 1456, 1593, 1621, 431, 832, -691, -1376, 1509, -135, 1040,
    -1443, -1319, 1267, 1314, -176, -1605, 895, 1379, 1377, 472, -1356, -1050,
    728, 112, -1437, -1593, -843, -888, 1568, 1295, -1096, -119, -399, 230, 1420,
    1427, -583, 649, 869, -422, 1117, 787, 869, 1507, -675, 745, 1079, -1269,
    706, 1122, -426, -829, -1373, -621, -981, 1558, 1601, 216, -1542, 380, 1489,
    -317, -992, 966, -1358, 52, 320, -128, 471, -254, -196, 250, -1261, 850,
    -353, 1481, -1394, 119, 1334
};

static int kyber_init_ntt(void) {
    if (!init_ntt) {
        for (int i = 0; i < 128; i++) zetas[i] = zetas_rom[i];
        for (int i = 0; i < 128; i++) zetas_inv[i] = zetas_rom[127 - i];
        init_ntt = 1;
    }
    return 1;
}

static int16_t fq_reduce(int32_t a) {
    int16_t t = (int16_t)((a + 3329 * 128) % 3329);
    return t >= 3329 ? (int16_t)(t - 3329) : t;
}

static int16_t mont_reduce(int32_t a) {
    int16_t u = (int16_t)(a * 20159);
    int32_t t = (u * 3329) + a;
    return (int16_t)(t >> 16);
}

static void ntt(int16_t r[256]) {
    int len = 128, k = 0;
    while (len >= 2) {
        int start = 0;
        while (start < 256) {
            int16_t zeta = zetas[++k];
            for (int j = start; j < start + len; j++) {
                int16_t t = mont_reduce(zeta * r[j + len]);
                r[j + len] = r[j] - t;
                r[j] = r[j] + t;
            }
            start += len * 2;
        }
        len >>= 1;
    }
}

static void inv_ntt(int16_t r[256]) {
    int len = 2, k = 127;
    while (len <= 128) {
        int start = 0;
        while (start < 256) {
            int16_t zeta = zetas_inv[--k];
            for (int j = start; j < start + len; j++) {
                int16_t t = r[j + len];
                r[j + len] = fq_reduce(r[j] - t);
                r[j] = fq_reduce(r[j] + t);
                r[j + len] = mont_reduce(zeta * r[j + len]);
            }
            start += len * 2;
        }
        len <<= 1;
    }
    for (int j = 0; j < 256; j++) r[j] = fq_reduce(r[j] * 3303);
}

static void poly_add(int16_t r[256], const int16_t a[256], const int16_t b[256]) {
    for (int i = 0; i < 256; i++) r[i] = fq_reduce(a[i] + b[i]);
}

static void poly_sub(int16_t r[256], const int16_t a[256], const int16_t b[256]) {
    for (int i = 0; i < 256; i++) r[i] = fq_reduce(a[i] - b[i]);
}

static void poly_mul(int16_t r[256], const int16_t a[256], const int16_t b[256]) {
    int16_t tmp[256];
    memcpy(tmp, a, sizeof(tmp));
    ntt(tmp);
    int16_t tmp2[256];
    memcpy(tmp2, b, sizeof(tmp2));
    ntt(tmp2);
    for (int i = 0; i < 256; i++) tmp[i] = mont_reduce(tmp[i] * tmp2[i]);
    inv_ntt(tmp);
    memcpy(r, tmp, sizeof(tmp));
}

static void poly_tobytes(uint8_t *out, const int16_t a[256]) {
    for (int i = 0; i < 128; i++) {
        int16_t t0 = a[2 * i], t1 = a[2 * i + 1];
        out[3 * i] = t0 & 0xff;
        out[3 * i + 1] = (t0 >> 8) | ((t1 & 0xf) << 4);
        out[3 * i + 2] = t1 >> 4;
    }
}

static void poly_frombytes(int16_t r[256], const uint8_t *in) {
    for (int i = 0; i < 128; i++) {
        r[2 * i] = ((in[3 * i + 1] & 0x0f) << 8) | in[3 * i];
        r[2 * i + 1] = (in[3 * i + 2] << 4) | ((in[3 * i + 1] >> 4) & 0x0f);
        if (r[2 * i] >= KYBER_Q) r[2 * i] -= KYBER_Q;
        if (r[2 * i + 1] >= KYBER_Q) r[2 * i + 1] -= KYBER_Q;
    }
}

static void poly_compress(uint8_t *out, const int16_t a[256], int d) {
    int32_t f = (1 << 12) / KYBER_Q;
    for (int i = 0; i < 256; i++) {
        int32_t t = (a[i] * f + (1 << 11)) >> 12;
        t &= (1 << d) - 1;
        int bit = i * d;
        for (int b = 0; b < d; b++)
            if (t & (1 << b)) out[bit / 8] |= (1 << (bit % 8 + b));
    }
}

static void poly_decompress(int16_t r[256], const uint8_t *in, int d) {
    int32_t f = (1 << d) / 2;
    for (int i = 0; i < 256; i++) {
        int t = 0, bit = i * d;
        for (int b = 0; b < d; b++)
            if (in[(bit + b) / 8] & (1 << ((bit + b) % 8))) t |= (1 << b);
        r[i] = (t * KYBER_Q + f) >> d;
    }
}

static void poly_frommsg(int16_t r[256], const uint8_t msg[32]) {
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 8; j++) {
            int bit = (msg[i] >> j) & 1;
            r[8 * i + j] = bit * (KYBER_Q / 2);
        }
}

static void poly_tomsg(uint8_t msg[32], const int16_t a[256]) {
    for (int i = 0; i < 32; i++) {
        msg[i] = 0;
        for (int j = 0; j < 8; j++) {
            int16_t t = a[8 * i + j];
            int bit = 1;
            if (t < KYBER_Q / 4 || t > 3 * KYBER_Q / 4) bit = 0;
            msg[i] |= bit << j;
        }
    }
}

static void poly_getnoise(int16_t r[256], const uint8_t *seed, uint8_t nonce) {
    uint8_t buf[256 * 2];
    uint8_t ctx[33];
    memcpy(ctx, seed, 32);
    ctx[32] = nonce;
    arix_random_bytes(buf, sizeof(buf));
    for (int i = 0; i < 256; i++) r[i] = (buf[2 * i] | ((uint16_t)buf[2 * i + 1] << 8)) & 0x1fff;
    for (int i = 0; i < 256; i++) if (r[i] >= KYBER_Q) r[i] = 0;
}

static void cpa_pke_keygen(uint8_t pk[KYBER_PUBLICKEYBYTES], uint8_t sk[KYBER_SECRETKEYBYTES]) {
    kyber_init_ntt();
    uint8_t seed[32];
    arix_random_bytes(seed, 32);
    int16_t a[KYBER_K * KYBER_K * 256], s[KYBER_K * 256], e[KYBER_K * 256];
    for (int i = 0; i < KYBER_K; i++) {
        poly_getnoise(s + i * 256, seed, i);
        poly_getnoise(e + i * 256, seed, KYBER_K + i);
    }
    for (int i = 0; i < KYBER_K * KYBER_K; i++)
        for (int j = 0; j < 256; j++) a[i * 256 + j] = (int16_t)(rand() % KYBER_Q);
    for (int i = 0; i < KYBER_K; i++) {
        int16_t t[256] = {0};
        for (int j = 0; j < KYBER_K; j++)
            poly_mul(t, a + (i * KYBER_K + j) * 256, s + j * 256);
        poly_sub(pk + i * 384, t, e + i * 256);
        poly_tobytes(pk + i * 384, pk + i * 384);
    }
    memcpy(sk, seed, 32);
    memcpy(sk + 32, pk, KYBER_PUBLICKEYBYTES);
}

int arix_kyber_keygen(uint8_t *pk, uint8_t *sk, int variant) {
    if (!pk || !sk) return -1;
    cpa_pke_keygen(pk, sk);
    return 0;
}

int arix_kyber_encaps(uint8_t *ct, uint8_t *ss, const uint8_t *pk, int variant) {
    if (!ct || !ss || !pk) return -1;
    uint8_t coin[32], m[32];
    arix_random_bytes(coin, 32);
    arix_random_bytes(m, 32);
    int16_t mp[256];
    poly_frommsg(mp, m);
    int16_t sp[KYBER_K * 256], ep[KYBER_K * 256], epp[256];
    for (int i = 0; i < KYBER_K; i++) {
        poly_getnoise(sp + i * 256, coin, i);
        poly_getnoise(ep + i * 256, coin, KYBER_K + i);
    }
    poly_getnoise(epp, coin, 2 * KYBER_K);
    int16_t at[KYBER_K * KYBER_K * 256];
    for (int i = 0; i < KYBER_K * KYBER_K; i++)
        for (int j = 0; j < 256; j++) at[i * 256 + j] = rand() % KYBER_Q;
    int16_t u[KYBER_K * 256], v[256];
    for (int i = 0; i < KYBER_K; i++) {
        int16_t t[256] = {0};
        for (int j = 0; j < KYBER_K; j++)
            poly_mul(t, at + (j * KYBER_K + i) * 256, sp + j * 256);
        poly_add(u + i * 256, t, ep + i * 256);
    }
    int16_t pkpoly[KYBER_K * 256];
    for (int i = 0; i < KYBER_K; i++)
        poly_frombytes(pkpoly + i * 256, pk + i * 384);
    for (int i = 0; i < KYBER_K; i++) {
        int16_t t[256] = {0};
        poly_mul(t, pkpoly + i * 256, sp + i * 256);
        poly_add(v, v, t);
    }
    poly_add(v, v, epp);
    poly_add(v, v, mp);
    for (int i = 0; i < KYBER_K; i++)
        poly_tobytes(ct + i * 384, u + i * 256);
    poly_compress(ct + KYBER_K * 384, v, 4);
    memcpy(ss, m, 32);
    return 0;
}

int arix_kyber_decaps(uint8_t *ss, const uint8_t *ct, const uint8_t *sk, int variant) {
    if (!ss || !ct || !sk) return -1;
    uint8_t pk[KYBER_PUBLICKEYBYTES];
    memcpy(pk, sk + 32, KYBER_PUBLICKEYBYTES);
    int16_t u[KYBER_K * 256], v[256];
    for (int i = 0; i < KYBER_K; i++)
        poly_frombytes(u + i * 256, ct + i * 384);
    poly_decompress(v, ct + KYBER_K * 384, 4);
    int16_t s[KYBER_K * 256];
    uint8_t seed[32];
    memcpy(seed, sk, 32);
    for (int i = 0; i < KYBER_K; i++)
        poly_getnoise(s + i * 256, seed, i);
    int16_t m[256] = {0};
    for (int i = 0; i < KYBER_K; i++) {
        int16_t t[256] = {0};
        poly_mul(t, s + i * 256, u + i * 256);
        poly_sub(m, m, t);
    }
    poly_add(m, m, v);
    poly_tomsg(ss, m);
    return 0;
}
