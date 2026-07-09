#ifndef SNEPPX_MEMORY_HARDENING_H
#define SNEPPX_MEMORY_HARDENING_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_MEM_GUARD_PAGE_SIZE 4096
#define SNEPPX_MEM_QUARANTINE_SIZE 128
#define SNEPPX_MAX_PAC_KEYS 16

/* Double-free / use-after-free detection */
typedef struct {
    void* entries[SNEPPX_MEM_QUARANTINE_SIZE];
    size_t sizes[SNEPPX_MEM_QUARANTINE_SIZE];
    uint64_t canaries[SNEPPX_MEM_QUARANTINE_SIZE];
    int count;
    int index;
} SNEPPXMemQuarantine;

int  SNEPPX_mem_quarantine_init(SNEPPXMemQuarantine* q);
int  SNEPPX_mem_quarantine_add(SNEPPXMemQuarantine* q, void* ptr, size_t size);
int  SNEPPX_mem_quarantine_check(SNEPPXMemQuarantine* q, const void* ptr);

/* Heap metadata encryption */
typedef struct {
    uint64_t xor_key;
    int enabled;
} SNEPPXHeapMetadataEncrypt;

int  SNEPPX_heap_metadata_init(SNEPPXHeapMetadataEncrypt* hme);
void SNEPPX_heap_metadata_encrypt(SNEPPXHeapMetadataEncrypt* hme, void* metadata, size_t len);
void SNEPPX_heap_metadata_decrypt(SNEPPXHeapMetadataEncrypt* hme, void* metadata, size_t len);

/* W^X enforcement */
int  SNEPPX_mem_enforce_wx(void* addr, size_t size);
int  SNEPPX_mem_set_rx(void* addr, size_t size);
int  SNEPPX_mem_set_rw(void* addr, size_t size);

/* Seccomp-BPF sandbox */
typedef struct {
    int enabled;
    int allow_read;
    int allow_write;
    int allow_open;
    int allow_socket;
    int allow_exec;
} SNEPPXSeccompConfig;

int  SNEPPX_seccomp_init(SNEPPXSeccompConfig* cfg);
int  SNEPPX_seccomp_apply(void);

/* Pointer authentication */
typedef struct {
    uint64_t pac_keys[SNEPPX_MAX_PAC_KEYS];
    int key_count;
} SNEPPXPAC;

int  SNEPPX_pac_init(SNEPPXPAC* pac);
uint64_t SNEPPX_pac_sign(SNEPPXPAC* pac, const void* pointer, int key_idx);
int  SNEPPX_pac_verify(SNEPPXPAC* pac, const void* pointer, uint64_t signature, int key_idx);

/* Control Flow Guard */
typedef struct {
    uintptr_t valid_targets[1024];
    int target_count;
} SNEPPXCFG;

int  SNEPPX_cfg_init(SNEPPXCFG* cfg);
int  SNEPPX_cfg_add_target(SNEPPXCFG* cfg, void* target);
int  SNEPPX_cfg_validate(SNEPPXCFG* cfg, void* target);

/* Safe stack (shadow call stack) */
#define SNEPPX_SHADOW_STACK_DEPTH 256

typedef struct {
    uintptr_t stack[SNEPPX_SHADOW_STACK_DEPTH];
    int sp;
    int overflow_detected;
} SNEPPXShadowStack;

int  SNEPPX_shadow_stack_init(SNEPPXShadowStack* ss);
int  SNEPPX_shadow_stack_push(SNEPPXShadowStack* ss, uintptr_t return_addr);
int  SNEPPX_shadow_stack_pop(SNEPPXShadowStack* ss, uintptr_t* return_addr);

/* Thread-local canary pool */
typedef struct {
    uint64_t canaries[64];
    int count;
} SNEPPXThreadCanaryPool;

int  SNEPPX_tls_canary_pool_init(SNEPPXThreadCanaryPool* pool);
uint64_t SNEPPX_tls_canary_alloc(SNEPPXThreadCanaryPool* pool);
int  SNEPPX_tls_canary_check(SNEPPXThreadCanaryPool* pool, uint64_t canary);

/* Guard page pool */
typedef struct {
    void* pages[64];
    size_t sizes[64];
    int count;
} SNEPPXGuardPagePool;

int  SNEPPX_guard_pool_init(SNEPPXGuardPagePool* pool);
void* SNEPPX_guard_pool_alloc(SNEPPXGuardPagePool* pool, size_t size);
int  SNEPPX_guard_pool_free(SNEPPXGuardPagePool* pool, void* ptr);

/* Memory pressure detection */
typedef struct {
    size_t total_allocated;
    size_t peak_allocated;
    size_t allocation_count;
    size_t allocation_limit;
    uint64_t last_warning_time;
    int pressure_level;
} SNEPPXMemPressure;

int  SNEPPX_mem_pressure_init(SNEPPXMemPressure* mp, size_t limit);
int  SNEPPX_mem_pressure_track(SNEPPXMemPressure* mp, size_t size);
int  SNEPPX_mem_pressure_release(SNEPPXMemPressure* mp, size_t size);
int  SNEPPX_mem_pressure_check(SNEPPXMemPressure* mp);

#ifdef __cplusplus
}
#endif
#endif
