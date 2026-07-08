#include "entropy_pool.h"
#include "siphash.h"
#include <string.h>
#include <time.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#endif

static uint64_t get_tsc(void) {
#ifdef _WIN32
    LARGE_INTEGER c; QueryPerformanceCounter(&c); return (uint64_t)c.QuadPart;
#else
    struct timeval tv; gettimeofday(&tv,NULL);
    return (uint64_t)tv.tv_sec*1000000ULL+tv.tv_usec;
#endif
}

int arix_entropy_pool_init(ArixEntropyPool* ep) {
    if (!ep) return -1;
    memset(ep,0,sizeof(*ep));
    for (int i=0;i<ARIX_ENTROPY_SOURCES;i++) ep->source_available[i]=1;
    return 0;
}

int arix_entropy_pool_add(ArixEntropyPool* ep, ArixEntropySource src, const uint8_t* data, size_t len) {
    if (!ep||!data||len==0) return -1;
    uint8_t sip_key[16]={0};
    ArixSipHash sh;
    arix_siphash_init(&sh,sip_key);
    arix_siphash_update(&sh,data,len);
    uint64_t hash=arix_siphash_finalize(&sh);

    ep->pool[ep->pool_index]^=(uint8_t)(hash>>56);
    if (++ep->pool_index>=ARIX_ENTROPY_POOL_SIZE) ep->pool_index=0;
    ep->pool[ep->pool_index]^=(uint8_t)(hash>>48);
    if (++ep->pool_index>=ARIX_ENTROPY_POOL_SIZE) ep->pool_index=0;
    ep->pool[ep->pool_index]^=(uint8_t)(hash>>40);
    if (++ep->pool_index>=ARIX_ENTROPY_POOL_SIZE) ep->pool_index=0;
    ep->pool[ep->pool_index]^=(uint8_t)(hash>>32);
    if (++ep->pool_index>=ARIX_ENTROPY_POOL_SIZE) ep->pool_index=0;
    ep->pool[ep->pool_index]^=(uint8_t)(hash>>24);
    if (++ep->pool_index>=ARIX_ENTROPY_POOL_SIZE) ep->pool_index=0;
    ep->pool[ep->pool_index]^=(uint8_t)(hash>>16);
    if (++ep->pool_index>=ARIX_ENTROPY_POOL_SIZE) ep->pool_index=0;
    ep->pool[ep->pool_index]^=(uint8_t)(hash>>8);
    if (++ep->pool_index>=ARIX_ENTROPY_POOL_SIZE) ep->pool_index=0;
    ep->pool[ep->pool_index]^=(uint8_t)hash;
    if (++ep->pool_index>=ARIX_ENTROPY_POOL_SIZE) ep->pool_index=0;

    ep->source_available[src]=1;
    ep->last_collection[src]=(uint64_t)time(NULL);
    ep->entropy_estimate+=2;
    if (ep->entropy_estimate>ARIX_ENTROPY_POOL_SIZE) ep->entropy_estimate=ARIX_ENTROPY_POOL_SIZE;
    return 0;
}

int arix_entropy_pool_add_rdtsc(ArixEntropyPool* ep) {
    if (!ep) return -1;
    uint64_t tsc=get_tsc();
    uint64_t tsc2=get_tsc();
    uint64_t jitter=tsc2>tsc?tsc2-tsc:tsc-tsc2;
    return arix_entropy_pool_add(ep,ARIX_ENTROPY_SOURCE_RDTSC,(const uint8_t*)&jitter,sizeof(jitter));
}

int arix_entropy_pool_add_os(ArixEntropyPool* ep) {
    if (!ep) return -1;
    uint8_t buf[32];
    int got=0;
#ifdef _WIN32
    HCRYPTPROV hProv=0;
    if (CryptAcquireContext(&hProv,NULL,NULL,PROV_RSA_FULL,CRYPT_VERIFYCONTEXT)) {
        if (CryptGenRandom(hProv,sizeof(buf),buf)) got=1;
        CryptReleaseContext(hProv,0);
    }
#else
    FILE* f=fopen("/dev/urandom","rb");
    if (f) { got=(fread(buf,1,sizeof(buf),f)==sizeof(buf)); fclose(f); }
#endif
    if (!got) { srand((unsigned)time(NULL)); for (int i=0;i<32;i++) buf[i]=(uint8_t)(rand()%256); }
    return arix_entropy_pool_add(ep,ARIX_ENTROPY_SOURCE_OS,buf,sizeof(buf));
}

int arix_entropy_pool_collect(ArixEntropyPool* ep) {
    if (!ep) return -1;
    arix_entropy_pool_add_rdtsc(ep);
    arix_entropy_pool_add_os(ep);
    return 0;
}

int arix_entropy_pool_get(ArixEntropyPool* ep, uint8_t* out, size_t out_len) {
    if (!ep||!out) return -1;
    if (ep->entropy_estimate<ARIX_ENTROPY_THRESHOLD) arix_entropy_pool_collect(ep);
    uint8_t sip_key[16];
    memcpy(sip_key,ep->pool,16);
    ArixSipHash sh;
    arix_siphash_init(&sh,sip_key);
    arix_siphash_update(&sh,ep->pool,ARIX_ENTROPY_POOL_SIZE);
    arix_siphash_update(&sh,(uint8_t*)&out_len,sizeof(out_len));
    uint64_t h=arix_siphash_finalize(&sh);
    for (size_t i=0;i<out_len;i++) { out[i]=(uint8_t)(h>>(i%8)*8); if (i%8==7) { h=get_tsc(); } }
    ep->entropy_estimate-=out_len;
    if (ep->entropy_estimate<0) ep->entropy_estimate=0;
    return 0;
}

int arix_entropy_pool_estimate(const ArixEntropyPool* ep) { return ep?ep->entropy_estimate:0; }

void arix_entropy_pool_stir(ArixEntropyPool* ep) {
    if (!ep) return;
    uint8_t sip_key[16];
    memcpy(sip_key,ep->pool,16);
    for (int i=0;i<4;i++) {
        uint8_t hash[32];
        arix_entropy_pool_get(ep,hash,32);
        for (int j=0;j<32&&(i*32+j)<ARIX_ENTROPY_POOL_SIZE;j++)
            ep->pool[i*32+j]=hash[j];
    }
}
