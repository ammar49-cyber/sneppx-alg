#include "pbkdf2.h"
#include "hmac.h"
#include <string.h>

int SNEPPX_pbkdf2_hmac_sha256(const uint8_t* pwd, size_t pwd_len, const uint8_t* salt, size_t salt_len, uint32_t iter, uint8_t* out, size_t out_len) {
    if (!pwd||!salt||!out) return -1;
    size_t hlen=32;
    uint32_t blocks=(uint32_t)((out_len+hlen-1)/hlen);
    for (uint32_t b=1;b<=blocks;b++) {
        uint8_t u[32],t[32],sblock[256];
        size_t sl=0;
        if (salt) { memcpy(sblock,salt,salt_len); sl=salt_len; }
        sblock[sl++]=(uint8_t)(b>>24); sblock[sl++]=(uint8_t)(b>>16);
        sblock[sl++]=(uint8_t)(b>>8);  sblock[sl++]=(uint8_t)b;
        SNEPPX_hmac_sha256(pwd,pwd_len,sblock,sl,u);
        memcpy(t,u,hlen);
        for (uint32_t i=1;i<iter;i++) {
            SNEPPX_hmac_sha256(pwd,pwd_len,u,hlen,u);
            for (size_t j=0;j<hlen;j++) t[j]^=u[j];
        }
        size_t copy=(out_len-(b-1)*hlen<hlen)?out_len-(b-1)*hlen:hlen;
        memcpy(out+(b-1)*hlen,t,copy);
    }
    return 0;
}

int SNEPPX_pbkdf2_hmac_sha512(const uint8_t* pwd, size_t pwd_len, const uint8_t* salt, size_t salt_len, uint32_t iter, uint8_t* out, size_t out_len) {
    if (!pwd||!salt||!out) return -1;
    size_t hlen=64;
    uint32_t blocks=(uint32_t)((out_len+hlen-1)/hlen);
    for (uint32_t b=1;b<=blocks;b++) {
        uint8_t u[64],t[64],sblock[256];
        size_t sl=0;
        if (salt) { memcpy(sblock,salt,salt_len); sl=salt_len; }
        sblock[sl++]=(uint8_t)(b>>24); sblock[sl++]=(uint8_t)(b>>16);
        sblock[sl++]=(uint8_t)(b>>8);  sblock[sl++]=(uint8_t)b;
        SNEPPX_hmac_sha512(pwd,pwd_len,sblock,sl,u);
        memcpy(t,u,hlen);
        for (uint32_t i=1;i<iter;i++) {
            SNEPPX_hmac_sha512(pwd,pwd_len,u,hlen,u);
            for (size_t j=0;j<hlen;j++) t[j]^=u[j];
        }
        size_t copy=(out_len-(b-1)*hlen<hlen)?out_len-(b-1)*hlen:hlen;
        memcpy(out+(b-1)*hlen,t,copy);
    }
    return 0;
}
