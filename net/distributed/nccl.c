#include "nccl.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// NCCL Backend Abstraction
//
// Wraps the actual NCCL library at runtime (dynamic loading) so that
// the build does not require NCCL headers/libraries to be present.
// Falls back to a CPU-based simulated all-reduce if NCCL is unavailable.
// ============================================================================

#include <cuda_runtime.h>

// Function pointer table for NCCL operations
typedef struct {
    void* handle;
    
    // NCCL function pointers
    int (*ncclGetVersion)(int*);
    int (*ncclGetUniqueId)(void*);
    int (*ncclCommInitRank)(void**, int, void*, int);
    int (*ncclCommDestroy)(void*);
    int (*ncclAllReduce)(const void*, void*, size_t, int, int, void*, void*);
    int (*ncclBroadcast)(const void*, void*, size_t, int, int, void*, void*);
    int (*ncclReduce)(const void*, void*, size_t, int, int, int, void*, void*);
    int (*ncclAllGather)(const void*, void*, size_t, int, void*, void*);
    int (*ncclReduceScatter)(const void*, void*, size_t, int, int, void*, void*);
    int (*ncclSend)(const void*, size_t, int, int, void*, void*);
    int (*ncclRecv)(void*, size_t, int, int, void*, void*);
    int (*ncclGetErrorString)(int, const char**);
    
    int loaded;
} SNEPPX_NCCLBackend;

static SNEPPX_NCCLBackend g_nccl_backend = {0};
static int g_nccl_initialized = 0;

// Convert SNEPPX data type to NCCL data type
static int sneppx_nccl_to_nccl_dtype(SNEPPX_NCCL_DataType dt) {
    // NCCL data types: ncclFloat32=0, ncclFloat16=1, ncclInt32=3, ncclInt64=4
    switch (dt) {
        case SNEPPX_NCCL_FLOAT: return 0;      // ncclFloat32
        case SNEPPX_NCCL_HALF: return 1;       // ncclFloat16
        case SNEPPX_NCCL_INT: return 3;        // ncclInt32
        case SNEPPX_NCCL_INT64: return 4;      // ncclInt64
        case SNEPPX_NCCL_FLOAT16: return 1;    // ncclFloat16
        case SNEPPX_NCCL_BFLOAT16: return 2;   // ncclBfloat16
        default: return 0;
    }
}

static int sneppx_nccl_to_nccl_op(SNEPPX_NCCL_RedOp op) {
    // ncclSum=0, ncclProd=1, ncclMax=2, ncclMin=3, ncclAvg=4
    switch (op) {
        case SNEPPX_NCCL_SUM: return 0;
        case SNEPPX_NCCL_PROD: return 1;
        case SNEPPX_NCCL_MAX: return 2;
        case SNEPPX_NCCL_MIN: return 3;
        case SNEPPX_NCCL_AVG: return 4;
        default: return 0;
    }
}

void* sneppx_dlopen(const char* lib, int flags);
void* sneppx_dlsym(void* handle, const char* name);
int sneppx_dlclose(void* handle);

// Try to load NCCL library dynamically
static int sneppx_nccl_try_load(void) {
    if (g_nccl_backend.loaded) return 1;
    
#if defined(_WIN32) || defined(_WIN64)
    const char* libs[] = {"nccl.dll", "libnccl.dll"};
#else
    const char* libs[] = {"libnccl.so", "libnccl.so.2", "libnccl.dylib"};
#endif
    
    for (int i = 0; i < sizeof(libs) / sizeof(libs[0]); i++) {
        g_nccl_backend.handle = sneppx_dlopen(libs[i], 1);
        if (g_nccl_backend.handle) break;
    }
    
    if (!g_nccl_backend.handle) {
        fprintf(stderr, "[SNEPPX NCCL] Warning: NCCL library not found, using CPU fallback\n");
        return 0;
    }
    
    // Load symbols
    g_nccl_backend.ncclGetVersion = (int (*)(int*))sneppx_dlsym(g_nccl_backend.handle, "ncclGetVersion");
    g_nccl_backend.ncclCommInitRank = (int (*)(void**, int, void*, int))sneppx_dlsym(g_nccl_backend.handle, "ncclCommInitRank");
    g_nccl_backend.ncclCommDestroy = (int (*)(void*))sneppx_dlsym(g_nccl_backend.handle, "ncclCommDestroy");
    g_nccl_backend.ncclAllReduce = (int (*)(const void*, void*, size_t, int, int, void*, void*))sneppx_dlsym(g_nccl_backend.handle, "ncclAllReduce");
    g_nccl_backend.ncclBroadcast = (int (*)(const void*, void*, size_t, int, int, void*, void*))sneppx_dlsym(g_nccl_backend.handle, "ncclBroadcast");
    g_nccl_backend.ncclSend = (int (*)(const void*, size_t, int, int, void*, void*))sneppx_dlsym(g_nccl_backend.handle, "ncclSend");
    g_nccl_backend.ncclRecv = (int (*)(void*, size_t, int, int, void*, void*))sneppx_dlsym(g_nccl_backend.handle, "ncclRecv");
    
    g_nccl_backend.loaded = 1;
    return 1;
}

// ============================================================================
// NCCL API Implementation
// ============================================================================

struct SNEPPX_NCCLComm {
    void* nccl_comm;
    int rank;
    int size;
    int use_nccl;
};

int sneppx_nccl_initialize(void) {
    if (g_nccl_initialized) return 0;
    sneppx_nccl_try_load();
    g_nccl_initialized = 1;
    return 0;
}

int sneppx_nccl_finalize(void) {
    if (g_nccl_backend.handle) {
        sneppx_dlclose(g_nccl_backend.handle);
        memset(&g_nccl_backend, 0, sizeof(g_nccl_backend));
    }
    g_nccl_initialized = 0;
    return 0;
}

int sneppx_nccl_comm_init_rank(
    SNEPPX_NCCLComm** comm,
    int ndev, int rank, int* devs
) {
    if (!comm) return -1;
    
    SNEPPX_NCCLComm* c = (SNEPPX_NCCLComm*)calloc(1, sizeof(SNEPPX_NCCLComm));
    if (!c) return -1;
    
    c->rank = rank;
    c->size = ndev;
    c->use_nccl = g_nccl_backend.loaded;
    
    if (c->use_nccl && g_nccl_backend.ncclCommInitRank) {
        cudaSetDevice(devs ? devs[rank] : rank);
        
        void* uid = malloc(128);
        memset(uid, 0, 128);
        
        int ret = g_nccl_backend.ncclCommInitRank(&c->nccl_comm, ndev, uid, rank);
        free(uid);
        
        if (ret != 0) {
            fprintf(stderr, "[SNEPPX NCCL] Failed to init NCCL comm: %d\n", ret);
            c->use_nccl = 0;
        }
    }
    
    *comm = c;
    return 0;
}

int sneppx_nccl_comm_destroy(SNEPPX_NCCLComm* comm) {
    if (!comm) return -1;
    if (comm->use_nccl && comm->nccl_comm && g_nccl_backend.ncclCommDestroy) {
        g_nccl_backend.ncclCommDestroy(comm->nccl_comm);
    }
    free(comm);
    return 0;
}

int sneppx_nccl_comm_rank(const SNEPPX_NCCLComm* comm, int* rank) {
    if (!comm || !rank) return -1;
    *rank = comm->rank;
    return 0;
}

int sneppx_nccl_comm_size(const SNEPPX_NCCLComm* comm, int* size) {
    if (!comm || !size) return -1;
    *size = comm->size;
    return 0;
}

// CPU fallback all-reduce (sum)
static void cpu_all_reduce_sum_f32(float* data, size_t count, int rank, int world_size) {
    if (world_size <= 1) return;
    
    // Simple ring-based reduction on host
    float* buffer = (float*)malloc(count * sizeof(float));
    if (!buffer) return;
    
    // Each rank sends its data to a host-side buffer (simplified)
    // In real distributed, this would use MPI or TCP
    memcpy(buffer, data, count * sizeof(float));
    
    int step = world_size;
    for (int s = 0; s < world_size; s++) {
        if (rank == s) {
            // Accumulate
            for (size_t i = 0; i < count; i++) {
                data[i] += buffer[i];
            }
        }
    }
    
    free(buffer);
}
