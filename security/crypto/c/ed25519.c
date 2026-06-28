#include "arix_ed25519.h"
#include "arix_sha512.h"
#include "arix_ct.h"
#include <string.h>

/* GF(2^255-19) field element: 5 limbs of 51 bits */
typedef struct { uint64_t v[5]; } field;

static const field D = {{
    0x0000000000000019ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL
}};
static const field D2 = {{
    0x0000000000000032ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL
}};
static const field SQRTM1 = {{
    0x1d5dc8c5feba2c79ULL, 0x499183e80ec73e91ULL, 0x5b596a684f25bcf9ULL,
    0x36f5067f552c7f08ULL, 0x0d113c3c86578acaULL
}};

static uint64_t mask51(uint64_t x) { return x & 0x7ffffffffffffULL; }
static uint64_t reduce51(uint64_t x) { return (x >> 51) + (x & 0x7ffffffffffffULL); }

static void fe_add(field* r, const field* a, const field* b) {
    for (int i = 0; i < 5; i++) r->v[i] = a->v[i] + b->v[i];
}

static void fe_sub(field* r, const field* a, const field* b) {
    for (int i = 0; i < 5; i++) r->v[i] = a->v[i] - b->v[i] + (uint64_t)0xfffffffffffdaULL;
}

static void fe_mul(field* r, const field* a, const field* b) {
    uint64_t t[9] = {0};
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++) {
            int k = i + j;
            if (k < 9) t[k] += a->v[i] * b->v[j];
        }
    for (int i = 8; i >= 5; i--) {
        t[i - 5] += 19 * (t[i] >> 51);
        t[i - 4] += 19 * (t[i] & 0x7ffffffffffffULL);
    }
    uint64_t c0 = t[0] >> 51; t[1] += c0; t[0] = mask51(t[0]);
    uint64_t c1 = t[1] >> 51; t[2] += c1; t[1] = mask51(t[1]);
    uint64_t c2 = t[2] >> 51; t[3] += c2; t[2] = mask51(t[2]);
    uint64_t c3 = t[3] >> 51; t[4] += c3; t[3] = mask51(t[3]);
    uint64_t c4 = t[4] >> 51; t[0] += 19 * c4; t[4] = mask51(t[4]);
    c0 = t[0] >> 51; t[1] += c0; t[0] = mask51(t[0]);
    for (int i = 0; i < 5; i++) r->v[i] = t[i];
}

static void fe_sq(field* r, const field* a) { fe_mul(r, a, a); }

static void fe_inv(field* r, const field* a) {
    field t1, t2, t3;
    fe_sq(&t1, a); fe_mul(&t2, &t1, a); fe_sq(&t1, &t2); fe_sq(&t1, &t1); fe_mul(&t2, &t1, a);
    for (int i = 0; i < 5; i++) { fe_sq(&t1, &t2); fe_mul(&t2, &t1, a); }
    fe_sq(&t1, &t2); fe_mul(&t3, &t1, a);
    for (int i = 0; i < 9; i++) fe_sq(&t1, &t3);
    fe_mul(&t2, &t1, &t3); fe_sq(&t1, &t2);
    for (int i = 0; i < 6; i++) { fe_sq(&t1, &t1); fe_mul(&t1, &t1, a); }
    fe_sq(&t1, &t1); fe_mul(r, &t1, &t2);
}

static void fe_from_bytes(field* r, const uint8_t b[32]) {
    r->v[0] = (uint64_t)b[0] | (uint64_t)b[1]<<8 | (uint64_t)b[2]<<16 | (uint64_t)b[3]<<24 |
              (uint64_t)b[4]<<32 | (uint64_t)b[5]<<40 | ((uint64_t)b[6]&7)<<48;
    r->v[1] = (uint64_t)b[6]>>3 | (uint64_t)b[7]<<5 | (uint64_t)b[8]<<13 | (uint64_t)b[9]<<21 |
              (uint64_t)b[10]<<29 | (uint64_t)b[11]<<37 | ((uint64_t)b[12]&63)<<45;
    r->v[2] = (uint64_t)b[12]>>6 | (uint64_t)b[13]<<2 | (uint64_t)b[14]<<10 | (uint64_t)b[15]<<18 |
              (uint64_t)b[16]<<26 | (uint64_t)b[17]<<34 | ((uint64_t)b[18]&31)<<42;
    r->v[3] = (uint64_t)b[18]>>5 | (uint64_t)b[19]<<3 | (uint64_t)b[20]<<11 | (uint64_t)b[21]<<19 |
              (uint64_t)b[22]<<27 | (uint64_t)b[23]<<35 | ((uint64_t)b[24]&15)<<43;
    r->v[4] = (uint64_t)b[24]>>4 | (uint64_t)b[25]<<4 | (uint64_t)b[26]<<12 | (uint64_t)b[27]<<20 |
              (uint64_t)b[28]<<28 | (uint64_t)b[29]<<36 | ((uint64_t)b[30]&7)<<44;
}

static void fe_to_bytes(uint8_t b[32], const field* r) {
    uint64_t t[5];
    memcpy(t, r->v, sizeof(t));
    uint64_t c = 19 * (t[4] >> 51);
    t[0] += c; t[4] = mask51(t[4]);
    uint64_t c0 = t[0] >> 51; t[1] += c0; t[0] = mask51(t[0]);
    uint64_t c1 = t[1] >> 51; t[2] += c1; t[1] = mask51(t[1]);
    uint64_t c2 = t[2] >> 51; t[3] += c2; t[2] = mask51(t[2]);
    uint64_t c3 = t[3] >> 51; t[4] += c3; t[3] = mask51(t[3]);
    uint64_t c4 = t[4] >> 51; t[0] += 19 * c4; t[4] = mask51(t[4]);
    c0 = t[0] >> 51; t[1] += c0; t[0] = mask51(t[0]);
    b[0] = (uint8_t)t[0]; b[1] = (uint8_t)(t[0] >> 8); b[2] = (uint8_t)(t[0] >> 16); b[3] = (uint8_t)(t[0] >> 24);
    b[4] = (uint8_t)(t[0] >> 32); b[5] = (uint8_t)(t[0] >> 40); b[6] = (uint8_t)(t[1] << 3) | (uint8_t)(t[0] >> 48);
    b[7] = (uint8_t)(t[1] >> 5); b[8] = (uint8_t)(t[1] >> 13); b[9] = (uint8_t)(t[1] >> 21);
    b[10] = (uint8_t)(t[1] >> 29); b[11] = (uint8_t)(t[1] >> 37); b[12] = (uint8_t)(t[2] << 6) | (uint8_t)(t[1] >> 45);
    b[13] = (uint8_t)(t[2] >> 2); b[14] = (uint8_t)(t[2] >> 10); b[15] = (uint8_t)(t[2] >> 18);
    b[16] = (uint8_t)(t[2] >> 26); b[17] = (uint8_t)(t[2] >> 34); b[18] = (uint8_t)(t[3] << 5) | (uint8_t)(t[2] >> 42);
    b[19] = (uint8_t)(t[3] >> 3); b[20] = (uint8_t)(t[3] >> 11); b[21] = (uint8_t)(t[3] >> 19);
    b[22] = (uint8_t)(t[3] >> 27); b[23] = (uint8_t)(t[3] >> 35); b[24] = (uint8_t)(t[4] << 4) | (uint8_t)(t[3] >> 43);
    b[25] = (uint8_t)(t[4] >> 4); b[26] = (uint8_t)(t[4] >> 12); b[27] = (uint8_t)(t[4] >> 20);
    b[28] = (uint8_t)(t[4] >> 28); b[29] = (uint8_t)(t[4] >> 36); b[30] = (uint8_t)(t[4] >> 44);
    b[31] = 0;
}

/* Point in extended coordinates (X, Y, Z, T) where x=X/Z, y=Y/Z, x*y=T/Z */
typedef struct { field X, Y, Z, T; } point;

static void point_set_neutral(point* p) {
    memset(p, 0, sizeof(point));
    p->Y.v[0] = 1; p->Z.v[0] = 1;
}

static void point_add(point* r, const point* p, const point* q) {
    field a, b, c, d, e, f, g, h;
    fe_sub(&a, &p->Y, &p->X); fe_sub(&b, &q->Y, &q->X); fe_mul(&a, &a, &b);
    fe_add(&c, &p->Y, &p->X); fe_add(&d, &q->Y, &q->X); fe_mul(&c, &c, &d);
    fe_mul(&e, &p->T, &q->T); fe_mul(&e, &e, &D2);
    fe_mul(&f, &p->Z, &q->Z); fe_add(&f, &f, &f);
    fe_add(&g, &c, &a); fe_sub(&h, &c, &a);
    fe_sub(&c, &f, &e); fe_add(&a, &f, &e);
    fe_mul(&r->X, &g, &c); fe_mul(&r->Y, &h, &a);
    fe_mul(&r->T, &g, &a); fe_mul(&r->Z, &h, &c);
}

static void point_double(point* r, const point* p) {
    field a, b, c, d, e, f;
    fe_sq(&a, &p->X); fe_sq(&b, &p->Y); fe_sq(&c, &p->Z);
    fe_add(&c, &c, &c); fe_add(&d, &p->X, &p->Y); fe_sq(&d, &d);
    fe_sub(&d, &d, &a); fe_sub(&d, &d, &b); fe_add(&e, &b, &a);
    fe_sub(&f, &d, &c); fe_mul(&r->X, &d, &f);
    fe_mul(&r->Y, &e, &b); fe_mul(&r->Y, &r->Y, &a);
    fe_mul(&r->T, &d, &e); fe_mul(&r->T, &r->T, &f);
    fe_mul(&r->Z, &c, &a); fe_mul(&r->Z, &r->Z, &b);
}

static point B;

static void init_base_point(void) {
    static int initialized = 0;
    if (initialized) return;
    B.X.v[0] = 0x08b999a3a49a0951ULL; B.X.v[1] = 0x088ec5e36ca3884fULL;
    B.X.v[2] = 0x34ceec1b86aa6a17ULL; B.X.v[3] = 0x48a365bd0c9d71a8ULL;
    B.X.v[4] = 0x28c8f376b8cfa895ULL;
    B.Y.v[0] = 0x60e0c40eed77c1deULL; B.Y.v[1] = 0x79918bbaa3dded63ULL;
    B.Y.v[2] = 0x48666e0e8d82a5e1ULL; B.Y.v[3] = 0x17db6d21b11bfbb5ULL;
    B.Y.v[4] = 0x62733c88ec799a2cULL;
    B.Z.v[0] = 1;
    B.T.v[0] = 0x7134d2ce4e2a1b62ULL; B.T.v[1] = 0x1d26a62b9b39e54dULL;
    B.T.v[2] = 0x489ee74d3b1fac69ULL; B.T.v[3] = 0x44bd81cf1ccfe980ULL;
    B.T.v[4] = 0x3a5c2bae045f1a87ULL;
    initialized = 1;
}

static int point_is_on_curve(const point* p) {
    field u, v;
    fe_sq(&u, &p->Y); fe_sq(&v, &p->X); fe_mul(&v, &v, &D); fe_add(&u, &u, &v);
    fe_mul(&v, &p->Z, &p->Z); fe_add(&u, &u, &v);
    fe_sub(&u, &u, &v); fe_mul(&v, &p->T, &p->Z); fe_sq(&v, &v);
    fe_sub(&u, &u, &v); fe_sq(&v, &p->X); fe_mul(&v, &v, &p->Y); fe_sq(&v, &v);
    uint8_t b1[32], b2[32]; fe_to_bytes(b1, &u); fe_to_bytes(b2, &v);
    return arix_ct_equal(b1, b2, 32);
}

static void point_scalar_mult(point* r, const uint8_t* scalar, size_t sc_len, const point* base) {
    point_set_neutral(r);
    for (int i = (int)sc_len * 8 - 1; i >= 0; i--) {
        point_double(r, r);
        uint8_t bit = (scalar[i / 8] >> (i % 8)) & 1;
        point t; memcpy(&t, r, sizeof(t));
        point_add(&t, &t, base);
        uint8_t mask = (uint8_t)(-(int)bit);
        for (int j = 0; j < 5; j++) {
            r->X.v[j] ^= mask & (r->X.v[j] ^ t.X.v[j]);
            r->Y.v[j] ^= mask & (r->Y.v[j] ^ t.Y.v[j]);
            r->Z.v[j] ^= mask & (r->Z.v[j] ^ t.Z.v[j]);
            r->T.v[j] ^= mask & (r->T.v[j] ^ t.T.v[j]);
        }
    }
}

static void clamp_scalar(uint8_t* s) {
    s[0] &= 248; s[31] &= 127; s[31] |= 64;
}

int arix_ed25519_secret_key_expand(uint8_t* expanded_sk, const uint8_t* seed) {
    if (!seed || !expanded_sk) return -1;
    uint8_t hash[64];
    arix_sha512(seed, 32, hash);
    memcpy(expanded_sk, hash, 64);
    clamp_scalar(expanded_sk);
    return 0;
}

int arix_ed25519_keypair_generate(ArixEd25519Keypair* kp) {
    if (!kp) return -1;
    init_base_point();
    memset(kp, 0, sizeof(ArixEd25519Keypair));
    uint8_t seed[32];
    extern int arix_random_bytes(uint8_t* buffer, size_t len);
    if (arix_random_bytes(seed, 32) != 0) return -1;
    if (arix_ed25519_secret_key_expand(kp->private_key, seed) != 0) return -1;
    point pub; point_scalar_mult(&pub, kp->private_key, 32, &B);
    fe_to_bytes(kp->public_key, &pub.Y);
    kp->public_key[31] |= (uint8_t)((pub.X.v[0] & 1) << 7);
    return 0;
}

int arix_ed25519_sign(const ArixEd25519Keypair* kp, const uint8_t* message, size_t msg_len, ArixEd25519Signature* sig) {
    if (!kp || !message || !sig) return -1;
    init_base_point();
    uint8_t hash[64], r[64];
    arix_sha512(kp->private_key + 32, 32, hash);
    memcpy(r, hash, 64);
    arix_sha512(r, 64, hash);
    point R; point_scalar_mult(&R, hash, 64, &B);
    fe_to_bytes(sig->data, &R.Y);
    sig->data[31] |= (uint8_t)((R.X.v[0] & 1) << 7);
    ArixSHA512Context ctx;
    arix_sha512_init(&ctx);
    arix_sha512_update(&ctx, sig->data, 32);
    arix_sha512_update(&ctx, kp->public_key, 32);
    arix_sha512_update(&ctx, message, msg_len);
    arix_sha512_finish(&ctx, hash);
    uint8_t S[32]; memcpy(S, hash, 32);
    uint8_t a[32]; memcpy(a, kp->private_key, 32);
    uint32_t carry = 0;
    for (int i = 0; i < 32; i++) {
        carry = (uint32_t)kp->private_key[i] + (uint32_t)S[i] + carry;
        sig->data[32 + i] = (uint8_t)(carry & 0xFF);
        carry >>= 8;
    }
    return 0;
}

int arix_ed25519_verify(const uint8_t* public_key, const uint8_t* message, size_t msg_len, const ArixEd25519Signature* sig) {
    if (!public_key || !message || !sig) return -1;
    if (sig->data[31] & 224) return -1;
    init_base_point();
    uint8_t hram[64];
    ArixSHA512Context ctx;
    arix_sha512_init(&ctx);
    arix_sha512_update(&ctx, sig->data, 32);
    arix_sha512_update(&ctx, public_key, 32);
    arix_sha512_update(&ctx, message, msg_len);
    arix_sha512_finish(&ctx, hram);
    point A; fe_from_bytes(&A.Y, public_key);
    A.X.v[0] = (uint64_t)(public_key[31] >> 7);
    A.Z.v[0] = 1;
    fe_sq(&A.T, &A.Y); fe_mul(&A.T, &A.T, &D); fe_add(&A.T, &A.T, &A.Z);
    fe_sub(&A.T, &A.T, &A.Z); fe_mul(&A.T, &A.T, &A.Y);
    point R; fe_from_bytes(&R.Y, sig->data);
    R.X.v[0] = (uint64_t)(sig->data[31] >> 7);
    R.Z.v[0] = 1;
    fe_sq(&R.T, &R.Y); fe_mul(&R.T, &R.T, &D); fe_add(&R.T, &R.T, &R.Z);
    fe_sub(&R.T, &R.T, &R.Z); fe_mul(&R.T, &R.T, &R.Y);
    uint8_t h_scalar[32]; memcpy(h_scalar, hram, 32);
    {
        static const uint8_t sc_l[32] = {
            0xed,0xd3,0xf5,0x5c,0x1a,0x63,0x12,0x58,
            0xd6,0x9c,0xf7,0xa2,0xde,0xf9,0xde,0x14,
            0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
            0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x10
        };
        for (int iter = 0; iter < 16; iter++) {
            uint32_t borrow = 0;
            for (int i = 0; i < 32; i++) {
                uint16_t w = (uint16_t)h_scalar[i] - sc_l[i] - borrow;
                borrow = (w >> 8) & 1;
                h_scalar[i] = (uint8_t)w;
            }
            if (borrow) { for (int i = 0; i < 32; i++) { uint16_t w = (uint16_t)h_scalar[i] + sc_l[i]; h_scalar[i] = (uint8_t)w; } break; }
        }
    }
    point hA; point_scalar_mult(&hA, h_scalar, 32, &A);
    point sB; point_scalar_mult(&sB, sig->data + 32, 32, &B);
    point R_plus_hA; point_add(&R_plus_hA, &R, &hA);
    field lhs, rhs;
    fe_mul(&lhs, &R_plus_hA.X, &sB.Z);
    fe_mul(&rhs, &sB.X, &R_plus_hA.Z);
    uint8_t lb1[32], lb2[32];
    fe_to_bytes(lb1, &lhs); fe_to_bytes(lb2, &rhs);
    return arix_ct_equal(lb1, lb2, 32);
}

int arix_ed25519_scalar_multiply(uint8_t* result, const uint8_t* scalar, const uint8_t* point_bytes) {
    if (!result || !scalar || !point_bytes) return -1;
    point p; fe_from_bytes(&p.Y, point_bytes);
    p.X.v[0] = (uint64_t)(point_bytes[31] >> 7);
    p.Z.v[0] = 1;
    fe_sq(&p.T, &p.Y); fe_mul(&p.T, &p.T, &D); fe_add(&p.T, &p.T, &p.Z);
    fe_sub(&p.T, &p.T, &p.Z); fe_mul(&p.T, &p.T, &p.Y);
    point r; point_scalar_mult(&r, scalar, 32, &p);
    fe_to_bytes(result, &r.Y);
    result[31] |= (uint8_t)((r.X.v[0] & 1) << 7);
    return 0;
}
