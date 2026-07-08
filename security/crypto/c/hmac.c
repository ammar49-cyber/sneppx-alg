#include "hmac.h"
#include "sha512_hashing_implementation.h"
#include "constant_time_operations.h"
#include <string.h>

static void sha256_hash(const uint8_t* data, size_t len, uint8_t out[32]) {
    ArixSHA512Context ctx;
    arix_sha512_init(&ctx);
    arix_sha512_update(&ctx,data,len);
    uint8_t full[64];
    arix_sha512_finish(&ctx,full);
    memcpy(out,full,32);
}

static void hmac_compute(const uint8_t* key, size_t key_len, const uint8_t* data, size_t data_len, uint8_t* out, size_t out_len, int use_sha512) {
    uint8_t k[128],ipad[128],opad[128];
    size_t block_size=use_sha512?128:64;
    size_t hash_size=use_sha512?64:32;
    memset(k,0,block_size);
    if (key_len>block_size) {
        if (use_sha512) { ArixSHA512Context ctx; arix_sha512_init(&ctx); arix_sha512_update(&ctx,key,key_len); uint8_t h[64]; arix_sha512_finish(&ctx,h); memcpy(k,h,hash_size); }
        else sha256_hash(key,key_len,k);
    } else memcpy(k,key,key_len);
    for (size_t i=0;i<block_size;i++) { ipad[i]=k[i]^0x36; opad[i]=k[i]^0x5C; }
    uint8_t inner[256];
    memcpy(inner,ipad,block_size);
    memcpy(inner+block_size,data,data_len);
    uint8_t inner_hash[64];
    if (use_sha512) { ArixSHA512Context ctx; arix_sha512_init(&ctx); arix_sha512_update(&ctx,inner,block_size+data_len); arix_sha512_finish(&ctx,inner_hash); }
    else sha256_hash(inner,block_size+data_len,inner_hash);
    uint8_t outer[256];
    memcpy(outer,opad,block_size);
    memcpy(outer+block_size,inner_hash,hash_size);
    if (use_sha512) { ArixSHA512Context ctx; arix_sha512_init(&ctx); arix_sha512_update(&ctx,outer,block_size+hash_size); arix_sha512_finish(&ctx,out); }
    else sha256_hash(outer,block_size+hash_size,out);
    memset(k,0,sizeof(k)); memset(ipad,0,sizeof(ipad)); memset(opad,0,sizeof(opad));
}

int arix_hmac_init(ArixHMAC* ctx, const uint8_t* key, size_t key_len, int hash_type) {
    if (!ctx||!key) return -1;
    ctx->key_len=key_len<ARIX_HMAC_MAX_KEY?key_len:ARIX_HMAC_MAX_KEY;
    memcpy(ctx->key,key,ctx->key_len);
    ctx->hash_type=hash_type;
    return 0;
}

int arix_hmac_compute(ArixHMAC* ctx, const uint8_t* data, size_t data_len, uint8_t* out, size_t* out_len) {
    if (!ctx||!data||!out||!out_len) return -1;
    size_t hash_size=(ctx->hash_type==0)?32:64;
    if (*out_len<hash_size) return -1;
    hmac_compute(ctx->key,ctx->key_len,data,data_len,out,hash_size,ctx->hash_type!=0);
    *out_len=hash_size;
    return 0;
}

int arix_hmac_sha256(const uint8_t* key, size_t key_len, const uint8_t* data, size_t data_len, uint8_t out[32]) {
    if (!key||!data||!out) return -1;
    hmac_compute(key,key_len,data,data_len,out,32,0);
    return 0;
}

int arix_hmac_sha512(const uint8_t* key, size_t key_len, const uint8_t* data, size_t data_len, uint8_t out[64]) {
    if (!key||!data||!out) return -1;
    hmac_compute(key,key_len,data,data_len,out,64,1);
    return 0;
}
