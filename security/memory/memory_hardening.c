#include "memory_hardening.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#include <sys/prctl.h>
#endif
#ifdef __linux__
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/syscall.h>
#endif

/* --- Quarantine --- */
int arix_mem_quarantine_init(ArixMemQuarantine* q) {
    if (!q) return -1;
    memset(q,0,sizeof(*q));
    return 0;
}

int arix_mem_quarantine_add(ArixMemQuarantine* q, void* ptr, size_t size) {
    if (!q||!ptr) return -1;
    for (int i=0;i<q->count;i++) if (q->entries[i]==ptr) return -1;
    q->entries[q->index]=ptr;
    q->sizes[q->index]=size;
    q->canaries[q->index]=(uint64_t)(uintptr_t)ptr ^ (uint64_t)time(NULL);
    memset(ptr,0xFD,size);
    q->index=(q->index+1)%ARIX_MEM_QUARANTINE_SIZE;
    if (q->count<ARIX_MEM_QUARANTINE_SIZE) q->count++;
    return 0;
}

int arix_mem_quarantine_check(ArixMemQuarantine* q, const void* ptr) {
    if (!q||!ptr) return 0;
    for (int i=0;i<q->count;i++) if (q->entries[i]==ptr) return 1;
    return 0;
}

/* --- Heap Metadata Encryption --- */
int arix_heap_metadata_init(ArixHeapMetadataEncrypt* hme) {
    if (!hme) return -1;
    hme->xor_key=(uint64_t)time(NULL)^0x9E3779B97F4A7C15ULL;
    hme->xor_key=hme->xor_key*6364136223846793005ULL+1442695040888963407ULL;
    hme->enabled=1;
    return 0;
}

void arix_heap_metadata_encrypt(ArixHeapMetadataEncrypt* hme, void* metadata, size_t len) {
    if (!hme||!hme->enabled||!metadata) return;
    uint8_t* p=(uint8_t*)metadata;
    for (size_t i=0;i<len;i++) {
        p[i]^=(uint8_t)(hme->xor_key>>((i%8)*8));
        if (i%8==7) hme->xor_key=hme->xor_key*6364136223846793005ULL+1;
    }
}

void arix_heap_metadata_decrypt(ArixHeapMetadataEncrypt* hme, void* metadata, size_t len) {
    arix_heap_metadata_encrypt(hme,metadata,len);
}

/* --- W^X Enforcement --- */
int arix_mem_enforce_wx(void* addr, size_t size) {
    (void)addr;(void)size;
#ifdef _WIN32
    DWORD old; return VirtualProtect(addr,size,PAGE_EXECUTE_READWRITE,&old)?0:-1;
#else
    if (!addr||size==0) return -1;
    size_t ps=getpagesize();
    void* aligned=(void*)((uintptr_t)addr&~(ps-1));
    size_t aligned_size=((uintptr_t)addr+size+ps-1)&~(ps-1);
    aligned_size-=(uintptr_t)aligned;
    return mprotect(aligned,aligned_size,PROT_READ|PROT_WRITE|PROT_EXEC);
#endif
}

int arix_mem_set_rx(void* addr, size_t size) {
#ifdef _WIN32
    DWORD old; return VirtualProtect(addr,size,PAGE_EXECUTE_READ,&old)?0:-1;
#else
    size_t ps=getpagesize();
    void* aligned=(void*)((uintptr_t)addr&~(ps-1));
    return mprotect(aligned,((uintptr_t)addr+size+ps-1)&~(ps-1)-(uintptr_t)aligned,PROT_READ|PROT_EXEC);
#endif
}

int arix_mem_set_rw(void* addr, size_t size) {
#ifdef _WIN32
    DWORD old; return VirtualProtect(addr,size,PAGE_READWRITE,&old)?0:-1;
#else
    size_t ps=getpagesize();
    void* aligned=(void*)((uintptr_t)addr&~(ps-1));
    return mprotect(aligned,((uintptr_t)addr+size+ps-1)&~(ps-1)-(uintptr_t)aligned,PROT_READ|PROT_WRITE);
#endif
}

/* --- Seccomp --- */
int arix_seccomp_init(ArixSeccompConfig* cfg) {
    if (!cfg) return -1;
    memset(cfg,0,sizeof(*cfg));
    cfg->allow_read=1; cfg->allow_write=1;
    cfg->allow_open=0; cfg->allow_socket=0;
    cfg->allow_exec=0; cfg->enabled=1;
    return 0;
}

int arix_seccomp_apply(void) {
#ifdef __linux__
    if (prctl(PR_SET_NO_NEW_PRIVS,1,0,0,0)<0) return -1;
    struct sock_filter filter[]={
        BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(struct seccomp_data,nr)),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_read,0,1), BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ALLOW),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_write,0,1), BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ALLOW),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_mmap,0,1), BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ALLOW),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_munmap,0,1), BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ALLOW),
        BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_exit_group,0,1), BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ALLOW),
        BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_KILL),
    };
    struct sock_fprog prog={.len=sizeof(filter)/sizeof(filter[0]),.filter=filter};
    return prctl(PR_SET_SECCOMP,SECCOMP_MODE_FILTER,&prog);
#else
    return 0;
#endif
}

/* --- Pointer Authentication --- */
int arix_pac_init(ArixPAC* pac) {
    if (!pac) return -1;
    for (int i=0;i<ARIX_MAX_PAC_KEYS;i++) {
        pac->pac_keys[i]=(uint64_t)rand()^((uint64_t)rand()<<32)^((uint64_t)time(NULL)<<(i%8));
        pac->pac_keys[i]=pac->pac_keys[i]*6364136223846793005ULL+1442695040888963407ULL;
    }
    pac->key_count=ARIX_MAX_PAC_KEYS;
    return 0;
}

uint64_t arix_pac_sign(ArixPAC* pac, const void* pointer, int key_idx) {
    if (!pac||!pointer||key_idx>=pac->key_count) return 0;
    uint64_t addr=(uint64_t)(uintptr_t)pointer;
    uint64_t key=pac->pac_keys[key_idx];
    uint64_t hash=addr;
    for (int i=0;i<8;i++) hash=(hash>>1)^(key&(1ULL<<63)?0xC96C5795D7870F42ULL:0);
    return (addr<<16)|(hash&0xFFFF);
}

int arix_pac_verify(ArixPAC* pac, const void* pointer, uint64_t signature, int key_idx) {
    if (!pac||!pointer||key_idx>=pac->key_count) return 0;
    uint64_t expected=arix_pac_sign(pac,pointer,key_idx);
    return (expected==signature)?1:0;
}

/* --- CFG --- */
int arix_cfg_init(ArixCFG* cfg) { if (!cfg) return -1; memset(cfg,0,sizeof(*cfg)); return 0; }

int arix_cfg_add_target(ArixCFG* cfg, void* target) {
    if (!cfg||!target||cfg->target_count>=1024) return -1;
    cfg->valid_targets[cfg->target_count++]=(uintptr_t)target;
    return 0;
}

int arix_cfg_validate(ArixCFG* cfg, void* target) {
    if (!cfg||!target) return 0;
    uintptr_t t=(uintptr_t)target;
    for (int i=0;i<cfg->target_count;i++) if (cfg->valid_targets[i]==t) return 1;
    return 0;
}

/* --- Shadow Stack --- */
int arix_shadow_stack_init(ArixShadowStack* ss) {
    if (!ss) return -1;
    memset(ss,0,sizeof(*ss));
    ss->sp=-1;
    return 0;
}

int arix_shadow_stack_push(ArixShadowStack* ss, uintptr_t return_addr) {
    if (!ss||ss->sp>=ARIX_SHADOW_STACK_DEPTH-1) { if (ss) ss->overflow_detected=1; return -1; }
    ss->stack[++ss->sp]=return_addr;
    return 0;
}

int arix_shadow_stack_pop(ArixShadowStack* ss, uintptr_t* return_addr) {
    if (!ss||!return_addr||ss->sp<0) return -1;
    *return_addr=ss->stack[ss->sp--];
    return 0;
}

/* --- TLS Canary Pool --- */
int arix_tls_canary_pool_init(ArixThreadCanaryPool* pool) {
    if (!pool) return -1;
    for (int i=0;i<64;i++) pool->canaries[i]=(uint64_t)rand()^((uint64_t)rand()<<32);
    pool->count=64;
    return 0;
}

uint64_t arix_tls_canary_alloc(ArixThreadCanaryPool* pool) {
    if (!pool||pool->count<=0) return 0;
    return pool->canaries[--pool->count];
}

int arix_tls_canary_check(ArixThreadCanaryPool* pool, uint64_t canary) {
    if (!pool) return 0;
    for (int i=0;i<64-pool->count;i++) if (pool->canaries[i]==canary) return 1;
    return 0;
}

/* --- Guard Page Pool --- */
int arix_guard_pool_init(ArixGuardPagePool* pool) {
    if (!pool) return -1;
    memset(pool,0,sizeof(*pool));
    return 0;
}

void* arix_guard_pool_alloc(ArixGuardPagePool* pool, size_t size) {
    (void)pool;
    if (size==0) return NULL;
    size_t total=size+ARIX_MEM_GUARD_PAGE_SIZE*2;
#ifdef _WIN32
    void* base=VirtualAlloc(NULL,total,MEM_RESERVE,PAGE_NOACCESS);
    if (!base) return NULL;
    void* user=(char*)base+ARIX_MEM_GUARD_PAGE_SIZE;
    if (!VirtualAlloc(user,size,MEM_COMMIT,PAGE_READWRITE)) { VirtualFree(base,0,MEM_RELEASE); return NULL; }
    return user;
#else
    size_t ps=sysconf(_SC_PAGESIZE);
    size_t aligned_total=((total+ps-1)/ps)*ps;
    void* base=mmap(NULL,aligned_total,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if (base==MAP_FAILED) return NULL;
    void* user=(char*)base+ps;
    if (mprotect(user,size,PROT_READ|PROT_WRITE)!=0) { munmap(base,aligned_total); return NULL; }
    return user;
#endif
}

int arix_guard_pool_free(ArixGuardPagePool* pool, void* ptr) {
    (void)pool;
    if (!ptr) return -1;
    size_t total=ARIX_MEM_GUARD_PAGE_SIZE*2;
#ifdef _WIN32
    VirtualFree((char*)ptr-ARIX_MEM_GUARD_PAGE_SIZE,0,MEM_RELEASE);
#else
    size_t ps=sysconf(_SC_PAGESIZE);
    void* base=(char*)ptr-ps;
    munmap(base,ps*3);
#endif
    return 0;
}

/* --- Memory Pressure --- */
int arix_mem_pressure_init(ArixMemPressure* mp, size_t limit) {
    if (!mp) return -1;
    memset(mp,0,sizeof(*mp));
    mp->allocation_limit=limit;
    return 0;
}

int arix_mem_pressure_track(ArixMemPressure* mp, size_t size) {
    if (!mp) return -1;
    mp->total_allocated+=size;
    mp->allocation_count++;
    if (mp->total_allocated>mp->peak_allocated) mp->peak_allocated=mp->total_allocated;
    if (mp->total_allocated>mp->allocation_limit) mp->pressure_level=1;
    if (mp->total_allocated>mp->allocation_limit*2) mp->pressure_level=2;
    return mp->pressure_level;
}

int arix_mem_pressure_release(ArixMemPressure* mp, size_t size) {
    if (!mp) return -1;
    mp->total_allocated-=(mp->total_allocated>=size)?size:mp->total_allocated;
    if (mp->total_allocated<=mp->allocation_limit) mp->pressure_level=0;
    return 0;
}

int arix_mem_pressure_check(ArixMemPressure* mp) {
    if (!mp) return -1;
    return mp->pressure_level;
}

/* === Internal Data Structures === */

#define ARIX_FREELIST_CANARY_VALUE 0xDEADBEEFCAFEBABEULL
#define ARIX_FREELIST_DEFAULT_MAX 2048
#define ARIX_MEM_POOL_MAX_BLOCKS 65536
#define ARIX_SCRUB_PASSES 4
#define ARIX_FREELIST_CANARY_SIZE sizeof(uint64_t)
#define ARIX_BITMAP_BITS_PER_BYTE 8
#define ARIX_POOL_SCRUB_BYTE 0xAB

typedef struct {
    void** entries;
    size_t* sizes;
    uint64_t* stored_canaries;
    int count;
    int max_count;
} ArixFreeList;

typedef struct {
    size_t current;
    int pressure_level;
    size_t peak;
} ArixMemPressureStats;

typedef struct {
    size_t block_size;
    int total_blocks;
    int free_blocks;
    uint8_t* bitmap;
    void* memory;
} ArixMemPool;

static ArixFreeList g_freelist;
static int g_freelist_initialized = 0;

static const uint8_t g_scrub_patterns[4] = {0xFD, 0xDF, 0xAB, 0x00};
static int g_scrub_phase = 0;

/* === Freelist Integrity === */

static uint64_t arix_freelist_canary_value(void* ptr, size_t size) {
    return (uint64_t)(uintptr_t)ptr ^ (uint64_t)size ^ ARIX_FREELIST_CANARY_VALUE;
}

static int arix_freelist_write_canary(void* ptr, size_t size) {
    if (!ptr || size < sizeof(uint64_t)) return -1;
    uint64_t* canary = (uint64_t*)((uint8_t*)ptr + size - sizeof(uint64_t));
    *canary = arix_freelist_canary_value(ptr, size);
    return 0;
}

static int arix_freelist_verify_canary(const void* ptr, size_t size) {
    if (!ptr || size < sizeof(uint64_t)) return 0;
    const uint64_t* canary = (const uint64_t*)((const uint8_t*)ptr + size - sizeof(uint64_t));
    return (*canary == arix_freelist_canary_value((void*)ptr, size)) ? 1 : 0;
}

static int arix_freelist_init_internal(int max_entries) {
    if (g_freelist_initialized) return 0;
    g_freelist.entries = (void**)calloc((size_t)max_entries, sizeof(void*));
    if (!g_freelist.entries) return -1;
    g_freelist.sizes = (size_t*)calloc((size_t)max_entries, sizeof(size_t));
    if (!g_freelist.sizes) { free(g_freelist.entries); g_freelist.entries = NULL; return -1; }
    g_freelist.stored_canaries = (uint64_t*)calloc((size_t)max_entries, sizeof(uint64_t));
    if (!g_freelist.stored_canaries) { free(g_freelist.entries); free(g_freelist.sizes); g_freelist.entries = NULL; g_freelist.sizes = NULL; return -1; }
    g_freelist.count = 0;
    g_freelist.max_count = max_entries;
    g_freelist_initialized = 1;
    return 0;
}

int arix_freelist_check(ArixFreeList* fl) {
    if (!fl) {
        if (!g_freelist_initialized) return -1;
        fl = &g_freelist;
    }
    for (int i = 0; i < fl->count; i++) {
        if (!fl->entries[i]) continue;
        if (!arix_freelist_verify_canary(fl->entries[i], fl->sizes[i])) return -1;
        if (fl->stored_canaries[i] != arix_freelist_canary_value(fl->entries[i], fl->sizes[i])) return -1;
    }
    return 0;
}

static int arix_freelist_add(void* ptr, size_t size) {
    if (!g_freelist_initialized) if (arix_freelist_init_internal(1024) != 0) return -1;
    if (!ptr) return -1;
    if (g_freelist.count >= g_freelist.max_count) return -1;
    for (int i = 0; i < g_freelist.count; i++) if (g_freelist.entries[i] == ptr) return -1;
    arix_freelist_write_canary(ptr, size);
    g_freelist.entries[g_freelist.count] = ptr;
    g_freelist.sizes[g_freelist.count] = size;
    g_freelist.stored_canaries[g_freelist.count] = arix_freelist_canary_value(ptr, size);
    g_freelist.count++;
    return 0;
}

static int arix_freelist_remove(void* ptr) {
    if (!g_freelist_initialized || !ptr) return -1;
    for (int i = 0; i < g_freelist.count; i++) {
        if (g_freelist.entries[i] == ptr) {
            g_freelist.entries[i] = g_freelist.entries[g_freelist.count - 1];
            g_freelist.sizes[i] = g_freelist.sizes[g_freelist.count - 1];
            g_freelist.stored_canaries[i] = g_freelist.stored_canaries[g_freelist.count - 1];
            g_freelist.count--;
            return 0;
        }
    }
    return -1;
}

/* === Memory Scrub === */

int arix_mem_scrub(void* ptr, size_t size) {
    if (!ptr || size == 0) return -1;
    uint8_t* p = (uint8_t*)ptr;
    size_t half = size / 2;
    size_t quarter = size / 4;
    size_t three_quarter = half + quarter;
    size_t pattern_idx = (size_t)g_scrub_phase % 4;
    volatile uint8_t sink = 0;
    for (size_t pass = 0; pass < ARIX_SCRUB_PASSES; pass++) {
        uint8_t current_pat = g_scrub_patterns[(pattern_idx + pass) % 4];
        for (size_t i = 0; i < size; i++) {
            p[i] ^= current_pat;
            p[i] = (uint8_t)((p[i] << 3) | (p[i] >> 5));
            p[i] ^= (uint8_t)(pass * 0x33);
            sink ^= p[i];
        }
        if (pass == 0) {
            for (size_t i = 0; i < size; i++) p[i] = 0xFD;
        }
        if (pass == 1) {
            for (size_t i = 0; i < half; i++) p[i] = (uint8_t)(p[i] ^ 0xDF);
            for (size_t i = half; i < size; i++) p[i] = (uint8_t)(p[i] ^ 0xBA);
        }
        if (pass == 2) {
            for (size_t i = 0; i < quarter; i++) p[i] = (uint8_t)(p[i] * 0xAB);
            for (size_t i = quarter; i < three_quarter; i++) p[i] = (uint8_t)(p[i] ^ 0xCD);
            for (size_t i = three_quarter; i < size; i++) p[i] = (uint8_t)(p[i] * 0xEF);
        }
        if (pass == 3) {
            for (size_t i = 0; i < size; i++) p[i] = 0x00;
        }
    }
    (void)sink;
    g_scrub_phase = (g_scrub_phase + 1) % 1024;
    return 0;
}

/* === Expanded W^X === */

int arix_mem_enforce_wx_strict(void) {
    int total_fixed = 0;
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    MEMORY_BASIC_INFORMATION mbi;
    uint8_t* addr = (uint8_t*)si.lpMinimumApplicationAddress;
    while (addr < (uint8_t*)si.lpMaximumApplicationAddress) {
        size_t query = VirtualQuery((void*)addr, &mbi, sizeof(mbi));
        if (query == 0) break;
        if (mbi.State == MEM_COMMIT && (mbi.Protect & 0xF0) != 0) {
            DWORD target = mbi.Protect;
            if ((mbi.Protect & PAGE_EXECUTE_READWRITE) == PAGE_EXECUTE_READWRITE) {
                target = PAGE_READWRITE;
            } else if ((mbi.Protect & PAGE_EXECUTE_WRITECOPY) == PAGE_EXECUTE_WRITECOPY) {
                target = PAGE_WRITECOPY;
            } else if (mbi.Protect & PAGE_EXECUTE) {
                if (mbi.Protect & PAGE_READONLY) target = PAGE_EXECUTE_READ;
                else if (mbi.Protect & PAGE_READWRITE) target = PAGE_EXECUTE_READWRITE;
                else target = PAGE_EXECUTE_READ;
            }
            if (target != mbi.Protect) {
                DWORD old;
                if (VirtualProtect(mbi.BaseAddress, mbi.RegionSize, target, &old)) total_fixed++;
            }
        }
        addr += mbi.RegionSize;
    }
    (void)si;
#else
    FILE* maps = fopen("/proc/self/maps", "r");
    if (!maps) return -1;
    char line[512];
    while (fgets(line, sizeof(line), maps)) {
        unsigned long start, end;
        char perms[8] = {0};
        if (sscanf(line, "%lx-%lx %4s", &start, &end, perms) < 3) continue;
        size_t len = end - start;
        if (len == 0) continue;
        if (strchr(perms, 'x') && strchr(perms, 'w')) {
            int prot = PROT_READ | PROT_WRITE;
            if (mprotect((void*)(uintptr_t)start, len, prot) == 0) total_fixed++;
        } else if (strchr(perms, 'x') && !strchr(perms, 'r') && !strchr(perms, 'w')) {
            int prot = PROT_READ | PROT_EXEC;
            if (mprotect((void*)(uintptr_t)start, len, prot) == 0) total_fixed++;
        } else if (perms[0] == '-' && perms[1] == '-' && perms[2] == 'x') {
            int prot = PROT_READ | PROT_EXEC;
            if (mprotect((void*)(uintptr_t)start, len, prot) == 0) total_fixed++;
        }
    }
    fclose(maps);
#endif
    return total_fixed;
}

int arix_mem_enforce_wx_check(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    MEMORY_BASIC_INFORMATION mbi;
    uint8_t* addr = (uint8_t*)si.lpMinimumApplicationAddress;
    while (addr < (uint8_t*)si.lpMaximumApplicationAddress) {
        if (VirtualQuery((void*)addr, &mbi, sizeof(mbi)) == 0) break;
        if (mbi.State == MEM_COMMIT && (mbi.Protect & 0xF0) != 0) {
            if ((mbi.Protect & PAGE_EXECUTE_READWRITE) == PAGE_EXECUTE_READWRITE) {
                (void)si;
                return 1;
            }
        }
        addr += mbi.RegionSize;
    }
    (void)si;
    return 0;
#else
    FILE* maps = fopen("/proc/self/maps", "r");
    if (!maps) return -1;
    int found = 0;
    char line[512];
    while (fgets(line, sizeof(line), maps)) {
        unsigned long start, end;
        char perms[8] = {0};
        if (sscanf(line, "%lx-%lx %4s", &start, &end, perms) < 3) continue;
        if (perms[0] == 'r' && perms[1] == 'w' && perms[2] == 'x') { found = 1; break; }
    }
    fclose(maps);
    return found;
#endif
}

/* === Expanded CFG === */

int arix_cfg_remove_target(ArixCFG* cfg, void* target) {
    if (!cfg || !target) return -1;
    uintptr_t t = (uintptr_t)target;
    for (int i = 0; i < cfg->target_count; i++) {
        if (cfg->valid_targets[i] == t) {
            cfg->valid_targets[i] = cfg->valid_targets[cfg->target_count - 1];
            cfg->target_count--;
            return 0;
        }
    }
    return -1;
}

int arix_cfg_clear(ArixCFG* cfg) {
    if (!cfg) return -1;
    memset(cfg->valid_targets, 0, sizeof(cfg->valid_targets));
    cfg->target_count = 0;
    return 0;
}

/* === Expanded Memory Pressure === */

ArixMemPressureStats arix_mem_pressure_get_stats(ArixMemPressure* mp) {
    ArixMemPressureStats stats;
    stats.current = 0;
    stats.pressure_level = -1;
    stats.peak = 0;
    if (!mp) return stats;
    stats.current = mp->total_allocated;
    stats.pressure_level = mp->pressure_level;
    stats.peak = mp->peak_allocated;
    return stats;
}

int arix_mem_pressure_set_limit(ArixMemPressure* mp, size_t limit) {
    if (!mp) return -1;
    mp->allocation_limit = limit;
    if (mp->total_allocated > limit) mp->pressure_level = 1;
    if (mp->total_allocated > limit * 2) mp->pressure_level = 2;
    if (mp->total_allocated <= limit) mp->pressure_level = 0;
    return 0;
}

/* === Memory Pool === */

int arix_mem_pool_init(ArixMemPool* pool, size_t block_size, int blocks) {
    if (!pool || block_size == 0 || blocks <= 0 || blocks > ARIX_MEM_POOL_MAX_BLOCKS) return -1;
    if (block_size < sizeof(uint64_t)) block_size = sizeof(uint64_t);
    pool->block_size = block_size;
    pool->total_blocks = blocks;
    pool->free_blocks = blocks;
    size_t bitmap_size = (size_t)((blocks + ARIX_BITMAP_BITS_PER_BYTE - 1) / ARIX_BITMAP_BITS_PER_BYTE);
    pool->bitmap = (uint8_t*)calloc(bitmap_size, 1);
    if (!pool->bitmap) return -1;
    pool->memory = (uint8_t*)calloc((size_t)blocks, block_size);
    if (!pool->memory) {
        free(pool->bitmap);
        pool->bitmap = NULL;
        return -1;
    }
    memset(pool->bitmap, 0, bitmap_size);
    memset(pool->memory, 0xAB, (size_t)blocks * block_size);
    return 0;
}

void* arix_mem_pool_alloc(ArixMemPool* pool) {
    if (!pool || !pool->bitmap || !pool->memory) return NULL;
    if (pool->free_blocks <= 0) return NULL;
    for (int i = 0; i < pool->total_blocks; i++) {
        int byte_idx = i / ARIX_BITMAP_BITS_PER_BYTE;
        int bit_idx = i % ARIX_BITMAP_BITS_PER_BYTE;
        if (!(pool->bitmap[byte_idx] & (uint8_t)(1 << bit_idx))) {
            pool->bitmap[byte_idx] |= (uint8_t)(1 << bit_idx);
            pool->free_blocks--;
            void* block = (uint8_t*)pool->memory + (size_t)i * pool->block_size;
            memset(block, 0, pool->block_size);
            return block;
        }
    }
    return NULL;
}

int arix_mem_pool_free(ArixMemPool* pool, void* ptr) {
    if (!pool || !pool->bitmap || !pool->memory || !ptr) return -1;
    uintptr_t mem_start = (uintptr_t)pool->memory;
    uintptr_t mem_end = mem_start + (uintptr_t)pool->total_blocks * pool->block_size;
    uintptr_t p = (uintptr_t)ptr;
    if (p < mem_start || p >= mem_end) return -1;
    uintptr_t offset = p - mem_start;
    if (offset % pool->block_size != 0) return -1;
    int idx = (int)(offset / pool->block_size);
    if (idx < 0 || idx >= pool->total_blocks) return -1;
    int byte_idx = idx / ARIX_BITMAP_BITS_PER_BYTE;
    int bit_idx = idx % ARIX_BITMAP_BITS_PER_BYTE;
    if (!(pool->bitmap[byte_idx] & (uint8_t)(1 << bit_idx))) return -1;
    memset(ptr, ARIX_POOL_SCRUB_BYTE, pool->block_size);
    pool->bitmap[byte_idx] &= (uint8_t)~(1 << bit_idx);
    pool->free_blocks++;
    return 0;
}

/* === Page Alignment === */

static size_t arix_mem_page_size(void);

int arix_mem_is_page_aligned(const void* addr) {
    if (!addr) return 0;
    size_t ps = arix_mem_page_size();
    return (((uintptr_t)addr & (ps - 1)) == 0) ? 1 : 0;
}

static size_t arix_mem_page_size(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (size_t)si.dwPageSize;
#elif defined(_SC_PAGESIZE)
    long sz = sysconf(_SC_PAGESIZE);
    return (sz > 0) ? (size_t)sz : 4096;
#elif defined(_SC_PAGE_SIZE)
    long sz = sysconf(_SC_PAGE_SIZE);
    return (sz > 0) ? (size_t)sz : 4096;
#else
    return 4096;
#endif
}

static int arix_mem_round_up_to_page(size_t* size) {
    if (!size || *size == 0) return -1;
    size_t ps = arix_mem_page_size();
    *size = (*size + ps - 1) & ~(ps - 1);
    return 0;
}

static int arix_mem_num_pages(size_t size) {
    size_t ps = arix_mem_page_size();
    if (size == 0) return 0;
    return (int)((size + ps - 1) / ps);
}

/* === Expanded Shadow Stack === */

int arix_shadow_stack_peek(ArixShadowStack* ss, uintptr_t* return_addr) {
    if (!ss || !return_addr) return -1;
    if (ss->sp < 0) return -1;
    *return_addr = ss->stack[ss->sp];
    return 0;
}

int arix_shadow_stack_depth(ArixShadowStack* ss) {
    if (!ss) return -1;
    return ss->sp + 1;
}

int arix_shadow_stack_validate(ArixShadowStack* ss) {
    if (!ss) return -1;
    if (ss->overflow_detected) return -1;
    if (ss->sp < -1 || ss->sp >= ARIX_SHADOW_STACK_DEPTH) return -1;
    if (ss->sp < 0) return 0;
    uint64_t check = 0;
    for (int i = 0; i <= ss->sp; i++) {
        if (ss->stack[i] == 0 || ss->stack[i] == (uintptr_t)-1) return -1;
        if (i > 0) {
            if (ss->stack[i] == ss->stack[i - 1] && ss->stack[i] == 0) return -1;
        }
        check ^= (uint64_t)ss->stack[i];
        check = (check << 7) | (check >> 57);
    }
    uint64_t expected = 0;
    for (int i = 0; i <= ss->sp; i++) {
        expected ^= (uint64_t)ss->stack[i];
        expected = (expected << 7) | (expected >> 57);
    }
    if (check != expected) return -1;
    for (int i = ss->sp + 1; i < ARIX_SHADOW_STACK_DEPTH; i++) {
        if (ss->stack[i] != 0) return -1;
    }
    return 0;
}

/* === Freelist Integration with Quarantine === */

int arix_quarantine_add_with_freelist(ArixMemQuarantine* q, void* ptr, size_t size) {
    int ret = arix_mem_quarantine_add(q, ptr, size);
    if (ret == 0) arix_freelist_add(ptr, size);
    return ret;
}

int arix_quarantine_check_with_freelist(ArixMemQuarantine* q, const void* ptr) {
    if (arix_mem_quarantine_check(q, ptr)) return 1;
    if (g_freelist_initialized) {
        for (int i = 0; i < g_freelist.count; i++) {
            if (g_freelist.entries[i] == ptr) return 1;
        }
    }
    return 0;
}
