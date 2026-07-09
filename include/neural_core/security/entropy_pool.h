#ifndef SNEPPX_ENTROPY_POOL_H
#define SNEPPX_ENTROPY_POOL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_ENTROPY_POOL_SIZE 256
#define SNEPPX_ENTROPY_THRESHOLD 128
#define SNEPPX_ENTROPY_SOURCES 8

typedef enum {
    SNEPPX_ENTROPY_SOURCE_RDTSC = 0,
    SNEPPX_ENTROPY_SOURCE_OS = 1,
    SNEPPX_ENTROPY_SOURCE_INTERRUPT_JITTER = 2,
    SNEPPX_ENTROPY_SOURCE_NETWORK = 3,
    SNEPPX_ENTROPY_SOURCE_DISK_TIMING = 4,
    SNEPPX_ENTROPY_SOURCE_MOUSE = 5,
    SNEPPX_ENTROPY_SOURCE_KEYBOARD = 6,
    SNEPPX_ENTROPY_SOURCE_MICROPHONE = 7,
} SNEPPXEntropySource;

typedef struct {
    uint8_t pool[SNEPPX_ENTROPY_POOL_SIZE];
    int pool_index;
    int entropy_estimate;
    int source_available[SNEPPX_ENTROPY_SOURCES];
    uint64_t last_collection[SNEPPX_ENTROPY_SOURCES];
} SNEPPXEntropyPool;

int  SNEPPX_entropy_pool_init(SNEPPXEntropyPool* ep);
int  SNEPPX_entropy_pool_add(SNEPPXEntropyPool* ep, SNEPPXEntropySource src, const uint8_t* data, size_t len);
int  SNEPPX_entropy_pool_add_rdtsc(SNEPPXEntropyPool* ep);
int  SNEPPX_entropy_pool_add_os(SNEPPXEntropyPool* ep);
int  SNEPPX_entropy_pool_collect(SNEPPXEntropyPool* ep);
int  SNEPPX_entropy_pool_get(SNEPPXEntropyPool* ep, uint8_t* out, size_t out_len);
int  SNEPPX_entropy_pool_estimate(const SNEPPXEntropyPool* ep);
void SNEPPX_entropy_pool_stir(SNEPPXEntropyPool* ep);

#ifdef __cplusplus
}
#endif
#endif
