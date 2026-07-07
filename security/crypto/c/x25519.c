#include "x25519.h"
#include "cryptographic_random_generator.h"
#include <string.h>

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    #define NO_UINT128
#endif

static const uint8_t basepoint[32] = {9};

#ifndef NO_UINT128
static void mul_sub(uint64_t* r, const uint64_t* a, const uint64_t* b, uint64_t* t) {
    uint64_t product[16];
    memset(product,0,sizeof(product));
    for (int i=0;i<4;i++) {
        uint64_t carry=0;
        for (int j=0;j<4;j++) {
            __uint128_t p=(__uint128_t)a[i]*b[j]+product[i+j]+carry;
            product[i+j]=(uint64_t)p;
            carry=(uint64_t)(p>>64);
        }
        product[i+4]+=carry;
    }
    memcpy(t,product,64);
}
#else
static void mul_sub(uint64_t* r, const uint64_t* a, const uint64_t* b, uint64_t* t) {
    (void)r; (void)a; (void)b; (void)t;
}
#endif

static void reduce(uint64_t* r, const uint64_t* t) {
    uint64_t t2[8];
    memcpy(t2,t,64);
    uint64_t mask=(t2[3]>>63)-1;
    t2[3]&=0x7FFFFFFFFFFFFFFFULL;
    for (int i=0;i<4;i++) {
        r[i]=t2[i];
        r[i+4]=0;
    }
    for (int i=0;i<3;i++) {
        uint64_t q=r[3]>>32;
        r[2]+=q*38; r[3]&=0xFFFFFFFF;
        if (r[2]<q*38) r[3]+=1;
        r[0]+=(r[3]>>32)*38; r[3]&=0xFFFFFFFF;
    }
    r[0] = (r[0]>>32)*38 + (r[0]&0xFFFFFFFF) + (r[1]>>32);
    r[1] = (r[1]&0xFFFFFFFF) + (r[2]>>32);
    r[2] = (r[2]&0xFFFFFFFF) + (r[3]>>32);
    r[3] = r[3]&0xFFFFFFFF;
}

static void fe_add(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    for (int i=0;i<4;i++) r[i]=a[i]+b[i];
}

static void fe_sub(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    for (int i=0;i<4;i++) r[i]=a[i]+(0xFFFFFFFFFFFFFFFFULL-b[i]);
}

#ifndef NO_UINT128
static void fe_mul(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    uint64_t t[8];
    mul_sub(t,a,b,(uint64_t(*)[4])t);
    reduce(r,t);
}

static void fe_invert(uint64_t* r, const uint64_t* a) {
    uint64_t t[4]; memcpy(t,a,32);
    for (int i=0;i<253;i++) fe_mul(t,t,t);
    memcpy(r,t,32);
}
#endif

void arix_x25519_clamp(uint8_t scalar[32]) {
    scalar[0]&=248; scalar[31]&=127; scalar[31]|=64;
}

#ifndef NO_UINT128
void arix_x25519_scalar_mult(uint8_t out[32], const uint8_t scalar[32], const uint8_t point[32]) {
    uint64_t x[4],z[4],x1[4],z1[4],a[4],b[4],c[4],d[4],e[4],f[4];
    uint8_t e_[32]; memcpy(e_,scalar,32); arix_x25519_clamp(e_);

    memset(x,0,32); memset(z,0,32); x[0]=1;
    for (int i=0;i<4;i++) { x1[i]=((uint64_t*)point)[i]; z1[i]=0; } z1[0]=1;

    for (int i=254;i>=0;i--) {
        int bit=(e_[i/8]>>(i%8))&1;
        fe_add(a,x1,z1); fe_sub(b,x1,z1);
        if (bit) {
            fe_sub(c,x,z); fe_add(d,x,z);
            fe_mul(e,a,c);  fe_mul(f,b,d);
        } else {
            fe_sub(c,x1,z1); fe_add(d,x1,z1);
            fe_mul(e,a,c);   fe_mul(f,b,d);
        }
    }
    fe_invert(z1,z);
    fe_mul(x,x1,z1);
    uint8_t* p=(uint8_t*)x;
    for (int i=0;i<32;i++) out[31-i]=p[i];
}
#else
void arix_x25519_scalar_mult(uint8_t out[32], const uint8_t scalar[32], const uint8_t point[32]) {
    (void)out; (void)scalar; (void)point;
}
#endif

void arix_x25519_keygen(uint8_t public_key[32], uint8_t secret_key[32]) {
    arix_random_bytes(secret_key,32);
    arix_x25519_clamp(secret_key);
    arix_x25519_scalar_mult(public_key,secret_key,basepoint);
}

int arix_x25519_shared_secret(uint8_t shared[32], const uint8_t secret_key[32], const uint8_t public_key[32]) {
    if (!shared||!secret_key||!public_key) return -1;
    arix_x25519_scalar_mult(shared,secret_key,public_key);
    return 0;
}

int arix_x25519_scalar_valid(const uint8_t scalar[32]) {
    if (!scalar) return 0;
    for (int i=0;i<32;i++) if (scalar[i]) return 1;
    return 0;
}
