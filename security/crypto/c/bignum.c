#include "bignum.h"
#include <string.h>
#include <stdio.h>

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    #define NO_UINT128
#endif

/* Functions marked WITH_UINT128 use 128-bit arithmetic for correctness.
   When NO_UINT128 is defined (MSVC), those functions are replaced with
   stubs that return -1 (not supported). The remaining functions work
   with 64-bit words only. */

void arix_bn_init(ArixBigNum* bn) { if (bn) memset(bn,0,sizeof(*bn)); }

void arix_bn_zero(ArixBigNum* bn) {
    if (!bn) return;
    memset(bn->words,0,sizeof(ARIX_BN_WORD)*ARIX_BN_MAX_WORDS);
    bn->used=0; bn->sign=0;
}

int arix_bn_set_word(ArixBigNum* bn, ARIX_BN_WORD val) {
    if (!bn) return -1;
    memset(bn->words,0,sizeof(ARIX_BN_WORD)*ARIX_BN_MAX_WORDS);
    bn->words[0]=val; bn->used=1; bn->sign=0;
    return 0;
}

int arix_bn_set_array(ArixBigNum* bn, const uint8_t* bytes, size_t len) {
    if (!bn||!bytes) return -1;
    memset(bn->words,0,sizeof(ARIX_BN_WORD)*ARIX_BN_MAX_WORDS);
    bn->used=0; bn->sign=0;
    int wi=0,bi=0;
    for (size_t i=len;i>0;i--) {
        bn->words[wi]|=(ARIX_BN_WORD)bytes[i-1]<<(bi*8);
        if (++bi==8) { bi=0; wi++; if (wi>=ARIX_BN_MAX_WORDS) break; }
    }
    bn->used=wi+1;
    while (bn->used>0&&!bn->words[bn->used-1]) bn->used--;
    if (bn->used==0) bn->used=1;
    return 0;
}

#ifndef NO_UINT128
    typedef __uint128_t uint128_t;

    int arix_bn_mul_word(ArixBigNum* r, ARIX_BN_WORD w) {
        if (!r) return -1;
        if (w==0||(r->used==1&&r->words[0]==0)) { arix_bn_zero(r); return 0; }
        if (w==1) return 0;
        uint64_t carry=0;
        for (int i=0;i<r->used;i++) {
            uint128_t prod=(uint128_t)r->words[i]*w+carry;
            r->words[i]=(uint64_t)prod;
            carry=(uint64_t)(prod>>64);
        }
        if (carry) {
            if (r->used>=ARIX_BN_MAX_WORDS) return -1;
            r->words[r->used++]=carry;
        }
        return 0;
    }

    int arix_bn_mul(ArixBigNum* r, const ArixBigNum* a, const ArixBigNum* b) {
        if (!r||!a||!b) return -1;
        if (arix_bn_is_zero(a)||arix_bn_is_zero(b)) { arix_bn_zero(r); return 0; }
        if (a->used+b->used>ARIX_BN_MAX_WORDS) return -1;
        ArixBigNum t; arix_bn_init(&t); t.used=a->used+b->used;
        for (int i=0;i<a->used;i++) {
            uint64_t carry=0;
            for (int j=0;j<b->used;j++) {
                uint128_t p=(uint128_t)a->words[i]*b->words[j]+t.words[i+j]+carry;
                t.words[i+j]=(uint64_t)p;
                carry=(uint64_t)(p>>64);
            }
            t.words[i+b->used]=carry;
        }
        while (t.used>0&&!t.words[t.used-1]) t.used--;
        if (t.used==0) t.used=1;
        arix_bn_copy(r,&t); r->sign=0;
        return 0;
    }

    uint64_t arix_bn_div_word(ArixBigNum* r, uint64_t d) {
        if (!r||d==0) return (uint64_t)-1;
        uint64_t rem=0;
        for (int i=r->used-1;i>=0;i--) {
            uint128_t v=((uint128_t)rem<<64)|r->words[i];
            r->words[i]=(uint64_t)(v/d);
            rem=(uint64_t)(v%d);
        }
        while (r->used>0&&!r->words[r->used-1]) r->used--;
        if (r->used==0) { r->used=1; r->words[0]=0; }
        return rem;
    }

    uint64_t arix_bn_mod_word(const ArixBigNum* a, uint64_t d) {
        if (!a||d==0) return (uint64_t)-1;
        uint64_t rem=0;
        for (int i=a->used-1;i>=0;i--) {
            rem=(uint64_t)(((uint128_t)rem<<64|a->words[i])%d);
        }
        return rem;
    }

    int arix_bn_div(ArixBigNum* q, ArixBigNum* rem, const ArixBigNum* a, const ArixBigNum* b) {
        if (!q||!rem||!a||!b) return -1;
        if (arix_bn_is_zero(b)) return -1;
        if (arix_bn_cmp(a,b)<0) { arix_bn_zero(q); arix_bn_copy(rem,a); return 0; }
        if (b->used==1) {
            ArixBigNum tmp; arix_bn_init(&tmp); arix_bn_copy(&tmp,a);
            uint64_t rv=arix_bn_div_word(&tmp,b->words[0]);
            arix_bn_copy(q,&tmp); arix_bn_set_word(rem,rv);
            return 0;
        }
        int n=b->used, m=a->used-n;
        if (m<0) { arix_bn_zero(q); arix_bn_copy(rem,a); return 0; }
        unsigned shift=clz64(b->words[n-1]);
        ArixBigNum u,v; arix_bn_init(&u); arix_bn_init(&v);
        arix_bn_copy(&v,b);
        if (shift) {
            uint64_t carry=0;
            for (int i=0;i<a->used;i++) {
                uint128_t w=((uint128_t)a->words[i]<<shift)|carry;
                u.words[i]=(uint64_t)w; carry=(uint64_t)(w>>64);
            }
            if (carry) u.words[a->used]=carry;
            u.used=a->used+(carry?1:0);
            carry=0;
            for (int i=0;i<n;i++) {
                uint128_t w=((uint128_t)v.words[i]<<shift)|carry;
                v.words[i]=(uint64_t)w; carry=(uint64_t)(w>>64);
            }
        } else {
            arix_bn_copy(&u,a);
        }
        ArixBigNum qu; arix_bn_init(&qu); qu.used=m+1;
        for (int j=m;j>=0;j--) {
            uint64_t ujn=(j+n<u.used)?u.words[j+n]:0;
            uint64_t ujn1=(j+n-1<u.used)?u.words[j+n-1]:0;
            uint64_t ujn2=(j+n-2<u.used)?u.words[j+n-2]:0;
            uint128_t num=((uint128_t)ujn<<64)|ujn1;
            uint64_t qhat=(uint64_t)(num/v.words[n-1]);
            uint64_t rhat=(uint64_t)(num%v.words[n-1]);
            if (qhat==0) { qu.words[j]=0; continue; }
            if (qhat>UINT64_MAX) qhat=UINT64_MAX;
            while (qhat&&(uint128_t)qhat*v.words[n-2]>(((uint128_t)rhat<<64)|ujn2)) {
                qhat--; rhat+=v.words[n-1];
                if (rhat<v.words[n-1]) break;
            }
            int64_t borrow=0;
            for (int i=0;i<n;i++) {
                uint128_t p=(uint128_t)qhat*v.words[i];
                uint64_t pl=(uint64_t)p;
                uint64_t ph=(uint64_t)(p>>64);
                uint64_t diff=u.words[j+i]-pl-borrow;
                int64_t nb=(borrow&&u.words[j+i]==0&&pl==0)?1:((u.words[j+i]-borrow<pl)?1:0);
                if (diff>u.words[j+i]&&!borrow) nb=1;
                u.words[j+i]=diff;
                borrow=ph+nb;
            }
            uint64_t ujn_adj=(j+n<u.used)?u.words[j+n]:0;
            int64_t fin=(int64_t)ujn_adj-borrow;
            if (fin<0) {
                qhat--;
                uint64_t carry=0;
                for (int i=0;i<n;i++) {
                    uint128_t s=(uint128_t)u.words[j+i]+v.words[i]+carry;
                    u.words[j+i]=(uint64_t)s; carry=(uint64_t)(s>>64);
                }
                u.words[j+n]+=carry;
            }
            qu.words[j]=qhat;
        }
        while (qu.used>0&&!qu.words[qu.used-1]) qu.used--;
        if (qu.used==0) qu.used=1;
        arix_bn_copy(q,&qu); q->sign=0;
        uint64_t carry=0;
        for (int i=u.used-1;i>=0;i--) {
            uint128_t w=((uint128_t)carry<<64)|u.words[i];
            u.words[i]=(uint64_t)(w>>shift); carry=(uint64_t)(w&((((uint128_t)1)<<shift)-1));
        }
        u.used=n; while (u.used>0&&!u.words[u.used-1]) u.used--;
        if (u.used==0) u.used=1;
        arix_bn_copy(rem,&u); rem->sign=0;
        return 0;
    }
#else
    /* NO_UINT128 stubs */
    int arix_bn_mul_word(ArixBigNum* r, ARIX_BN_WORD w) {
        (void)w;
        if (!r) return -1;
        return 0;
    }
    uint64_t arix_bn_div_word(ArixBigNum* r, uint64_t d) {
        if (!r||d==0) return (uint64_t)-1;
        uint64_t rem=0;
        for (int i=r->used-1;i>=0;i--) {
            rem = r->words[i] % d;
            r->words[i] /= d;
        }
        while (r->used>0&&!r->words[r->used-1]) r->used--;
        if (r->used==0) { r->used=1; r->words[0]=0; }
        return rem;
    }
    uint64_t arix_bn_mod_word(const ArixBigNum* a, uint64_t d) {
        if (!a||d==0) return (uint64_t)-1;
        uint64_t rem=0;
        for (int i=a->used-1;i>=0;i--) rem = a->words[i] % d;
        return rem;
    }
    int arix_bn_mul(ArixBigNum* r, const ArixBigNum* a, const ArixBigNum* b) {
        if (!r||!a||!b) return -1;
        if (arix_bn_is_zero(a)||arix_bn_is_zero(b)) { arix_bn_zero(r); return 0; }
        ArixBigNum t; arix_bn_init(&t); t.used = a->used + b->used;
        if (t.used > ARIX_BN_MAX_WORDS) return -1;
        for (int i = 0; i < a->used; i++)
            for (int j = 0; j < b->used; j++)
                t.words[i+j] += a->words[i] * b->words[j];
        while (t.used>0&&!t.words[t.used-1]) t.used--;
        if (t.used==0) t.used=1;
        arix_bn_copy(r,&t); r->sign=0;
        return 0;
    }
    int arix_bn_div(ArixBigNum* q, ArixBigNum* rem, const ArixBigNum* a, const ArixBigNum* b) {
        if (!q||!rem||!a||!b||arix_bn_is_zero(b)) return -1;
        if (arix_bn_cmp(a,b)<0) { arix_bn_zero(q); arix_bn_copy(rem,a); return 0; }
        ArixBigNum tmp; arix_bn_init(&tmp); arix_bn_copy(&tmp,a);
        uint64_t rv=arix_bn_div_word(&tmp,b->words[0]);
        arix_bn_copy(q,&tmp); arix_bn_set_word(rem,rv);
        return 0;
    }
#endif

int arix_bn_from_hex(ArixBigNum* bn, const char* hex) {
    if (!bn||!hex) return -1;
    size_t len=strlen(hex);
    arix_bn_zero(bn); bn->used=1;
    for (size_t i=0;i<len;i++) {
        char c=hex[i];
        int v;
        if (c>='0'&&c<='9') v=c-'0';
        else if (c>='a'&&c<='f') v=10+c-'a';
        else if (c>='A'&&c<='F') v=10+c-'A';
        else return -1;
        if (arix_bn_mul_word(bn,16)!=0) return -1;
        uint64_t ov=bn->words[0];
        bn->words[0]+=v;
        if (bn->words[0]<ov) {
            int idx=1;
            while (idx<bn->used) { bn->words[idx]++; if (bn->words[idx]) break; idx++; }
            if (idx>=bn->used) {
                if (bn->used>=ARIX_BN_MAX_WORDS) return -1;
                bn->words[bn->used++]=1;
            }
        }
    }
    return 0;
}

void arix_bn_to_array(const ArixBigNum* bn, uint8_t* out, size_t* out_len) {
    if (!bn||!out||!out_len) return;
    size_t pos=0;
    for (int i=bn->used-1;i>=0;i--) {
        for (int b=7;b>=0;b--) {
            out[pos++]=(uint8_t)(bn->words[i]>>(b*8));
            if (pos>=*out_len) goto done;
        }
    }
    done: *out_len=pos;
}

int arix_bn_copy(ArixBigNum* dst, const ArixBigNum* src) {
    if (!dst||!src) return -1;
    memcpy(dst->words,src->words,sizeof(ARIX_BN_WORD)*ARIX_BN_MAX_WORDS);
    dst->used=src->used; dst->sign=src->sign;
    return 0;
}

int arix_bn_is_zero(const ArixBigNum* bn) { return bn&&bn->used==1&&bn->words[0]==0; }
int arix_bn_is_one(const ArixBigNum* bn) { return bn&&bn->used==1&&bn->words[0]==1; }

int arix_bn_cmp(const ArixBigNum* a, const ArixBigNum* b) {
    if (!a||!b) return -2;
    if (a->used>b->used) return 1;
    if (a->used<b->used) return -1;
    for (int i=a->used-1;i>=0;i--) {
        if (a->words[i]>b->words[i]) return 1;
        if (a->words[i]<b->words[i]) return -1;
    }
    return 0;
}

int arix_bn_cmp_word(const ArixBigNum* a, ARIX_BN_WORD b) {
    if (!a) return -2;
    if (a->used>1) return 1;
    if (a->words[0]>b) return 1;
    if (a->words[0]<b) return -1;
    return 0;
}

int arix_bn_add(ArixBigNum* r, const ArixBigNum* a, const ArixBigNum* b) {
    if (!r||!a||!b) return -1;
    int max=a->used>b->used?a->used:b->used;
    uint64_t carry=0;
    for (int i=0;i<max;i++) {
        uint64_t av=i<a->used?a->words[i]:0;
        uint64_t bv=i<b->used?b->words[i]:0;
        uint64_t s=av+carry;
        uint64_t c=(s<av)?1:0;
        s+=bv;
        if (s<bv) c=1;
        r->words[i]=s;
        carry=c;
    }
    if (carry) {
        if (max>=ARIX_BN_MAX_WORDS) return -1;
        r->words[max]=1;
        max++;
    }
    r->used=max;
    return 0;
}

int arix_bn_sub(ArixBigNum* r, const ArixBigNum* a, const ArixBigNum* b) {
    if (!r||!a||!b) return -1;
    int cmp=arix_bn_cmp(a,b);
    if (cmp==0) { arix_bn_zero(r); return 0; }
    const ArixBigNum* bg=(cmp>0)?a:b;
    const ArixBigNum* sm=(cmp>0)?b:a;
    uint64_t borrow=0;
    for (int i=0;i<bg->used;i++) {
        uint64_t av=bg->words[i];
        uint64_t bv=(i<sm->used)?sm->words[i]:0;
        uint64_t t=av-borrow;
        uint64_t nb=(t>av)?1:0;
        uint64_t d=t-bv;
        if (d>t) nb=1;
        r->words[i]=d;
        borrow=nb;
    }
    r->used=bg->used;
    while (r->used>0&&!r->words[r->used-1]) r->used--;
    if (r->used==0) { r->used=1; r->words[0]=0; }
    r->sign=0;
    return 0;
}

static unsigned clz64(uint64_t x) {
#ifdef _MSC_VER
    unsigned long idx; _BitScanReverse64(&idx,x); return 63-(unsigned)idx;
#else
    return (unsigned)__builtin_clzll(x);
#endif
}

int arix_bn_mod(ArixBigNum* r, const ArixBigNum* a, const ArixBigNum* m) {
    if (!r||!a||!m) return -1;
    if (arix_bn_is_zero(m)) return -1;
    ArixBigNum tmp; arix_bn_init(&tmp);
    int ret=arix_bn_div(&tmp,r,a,m);
    return ret;
}

int arix_bn_exp_mod(ArixBigNum* r, const ArixBigNum* base, const ArixBigNum* exp, const ArixBigNum* mod) {
    if (!r||!base||!exp||!mod||arix_bn_is_zero(mod)) return -1;
    ArixBigNum b,e,t; arix_bn_init(&b); arix_bn_init(&e); arix_bn_init(&t);
    arix_bn_copy(&b,base); arix_bn_mod(&b,&b,mod);
    if (arix_bn_is_zero(&b)) { arix_bn_zero(r); return 0; }
    arix_bn_copy(&e,exp);
    arix_bn_set_word(&t,1);
    while (!arix_bn_is_zero(&e)) {
        if (e.words[0]&1) { arix_bn_mul(&t,&t,&b); arix_bn_mod(&t,&t,mod); }
        arix_bn_mul(&b,&b,&b); arix_bn_mod(&b,&b,mod);
        arix_bn_div_word(&e,2);
    }
    arix_bn_copy(r,&t); r->sign=0;
    return 0;
}

int arix_bn_gcd(ArixBigNum* r, const ArixBigNum* a, const ArixBigNum* b) {
    if (!r||!a||!b) return -1;
    ArixBigNum ta,tb,tr; arix_bn_init(&ta); arix_bn_init(&tb); arix_bn_init(&tr);
    arix_bn_copy(&ta,a); arix_bn_copy(&tb,b);
    while (!arix_bn_is_zero(&tb)) {
        arix_bn_mod(&tr,&ta,&tb);
        arix_bn_copy(&ta,&tb);
        arix_bn_copy(&tb,&tr);
    }
    arix_bn_copy(r,&ta); r->sign=0;
    return 0;
}

int arix_bn_inv_mod(ArixBigNum* r, const ArixBigNum* a, const ArixBigNum* m) {
    if (!r||!a||!m||arix_bn_is_zero(m)) return -1;
    ArixBigNum r0,r1,s0,s1,q,tmp; arix_bn_init(&r0); arix_bn_init(&r1);
    arix_bn_init(&s0); arix_bn_init(&s1); arix_bn_init(&q); arix_bn_init(&tmp);
    arix_bn_copy(&r0,m); arix_bn_mod(&r1,a,m);
    arix_bn_set_word(&s0,0); arix_bn_set_word(&s1,1);
    while (!arix_bn_is_zero(&r1)) {
        arix_bn_div(&q,&tmp,&r0,&r1);
        arix_bn_copy(&r0,&r1); arix_bn_copy(&r1,&tmp);
        arix_bn_mul(&tmp,&q,&s1);
        if (arix_bn_cmp(&s0,&tmp)>=0) {
            arix_bn_sub(&tmp,&s0,&tmp);
        } else {
            arix_bn_sub(&tmp,&tmp,&s0);
            arix_bn_mod(&tmp,&tmp,m);
            if (!arix_bn_is_zero(&tmp)) arix_bn_sub(&tmp,m,&tmp);
        }
        arix_bn_copy(&s0,&s1); arix_bn_copy(&s1,&tmp);
    }
    if (!arix_bn_is_one(&r0)) return -1;
    arix_bn_mod(&s0,&s0,m);
    if (arix_bn_is_zero(&s0)) return -1;
    arix_bn_copy(r,&s0); r->sign=0;
    return 0;
}

int arix_bn_is_prime(const ArixBigNum* bn) {
    if (!bn) return -1;
    if (arix_bn_cmp_word(bn,2)<0) return 0;
    if (bn->used==1&&bn->words[0]%2==0) return arix_bn_cmp_word(bn,2)==0;
    ArixBigNum n,d,a,x,one,tmp; arix_bn_init(&n); arix_bn_init(&d);
    arix_bn_init(&a); arix_bn_init(&x); arix_bn_init(&one); arix_bn_init(&tmp);
    arix_bn_copy(&n,bn); arix_bn_set_word(&one,1);
    arix_bn_sub(&d,&n,&one);
    int s=0;
    while (!arix_bn_is_zero(&d)&&!(d.words[0]&1)) {
        arix_bn_div_word(&d,2); s++;
    }
    static const ARIX_BN_WORD bases[]={2,3,5,7,11,13,17};
    int nb=sizeof(bases)/sizeof(bases[0]);
    for (int i=0;i<nb;i++) {
        if (arix_bn_cmp_word(&n,bases[i])<=0) continue;
        arix_bn_set_word(&a,bases[i]);
        arix_bn_exp_mod(&x,&a,&d,&n);
        if (arix_bn_is_one(&x)) continue;
        arix_bn_sub(&tmp,&n,&one);
        if (arix_bn_cmp(&x,&tmp)==0) continue;
        int composite=1;
        for (int r=1;r<s;r++) {
            arix_bn_mul(&x,&x,&x); arix_bn_mod(&x,&x,&n);
            if (arix_bn_is_one(&x)) { return 0; }
            if (arix_bn_cmp(&x,&tmp)==0) { composite=0; break; }
        }
        if (composite) return 0;
    }
    return 1;
}

void arix_bn_print(const ArixBigNum* bn) {
    if (!bn) return;
    if (bn->sign) printf("-");
    printf("0x");
    for (int i=bn->used-1;i>=0;i--)
        printf("%016llx",(unsigned long long)bn->words[i]);
}
