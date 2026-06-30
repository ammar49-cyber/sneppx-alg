#ifndef ARIX_CUDA_DRIVER_H
#define ARIX_CUDA_DRIVER_H
/*
 * CUDA Driver Interface — v1.0 (GPU acceleration)
 *
 * PURPOSE: Abstract NVIDIA GPU device behind the kernel's device operations
 * structure.  This header declares the CUDA driver registration, device context,
 * stream/event management, and kernel dispatch interface.
 *
 * DEPENDENCIES: polymorphic_memory_allocator.h (device allocator), multidimensional_tensor_engine.h (tensor struct)
 * VERSION: v1.0 — GPU training
 *
 * IMPLEMENTATION STATUS: SKELETON — no real CUDA calls
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- forward declarations from core ---------- */
typedef struct ArixTensor ArixTensor;

/* ---------- Device properties queried at runtime ---------- */
typedef struct {
    char     name[256];
    size_t   global_mem_bytes;
    size_t   shared_mem_per_block;
    int      warp_size;
    int      max_threads_per_block;
    int      max_blocks_per_grid[3];
    int      compute_capability_major;
    int      compute_capability_minor;
    int      num_sms;               /* streaming multiprocessors */
    int      supports_tensor_cores;
    int      supports_cooperative_groups;
} ArixCUDADeviceProps;

/* ---------- Stream / event handles (opaque) ---------- */
typedef struct ArixCUDAStream {
    void*    handle;                /* cudaStream_t stored as void* */
    int      device_id;
    int      priority;
} ArixCUDAStream;

typedef struct ArixCUDAEvent {
    void*    handle;                /* cudaEvent_t stored as void* */
    int      device_id;
} ArixCUDAEvent;

/* ---------- Device context — one per GPU ---------- */
typedef struct {
    int                     device_id;
    ArixCUDADeviceProps     props;
    size_t                  alloc_bytes;       /* current device allocations */
    size_t                  peak_alloc_bytes;
    ArixCUDAStream**        streams;           /* owned stream array */
    int                     num_streams;
    void*                   blas_handle;       /* cuBLAS handle */
    void*                   dnn_handle;        /* cuDNN handle (v1.0) */
    void*                   solver_handle;     /* cuSOLVER handle (v2.0) */
    int                     error_state;       /* non-zero after fatal error */
} ArixCUDAContext;

/* ---------- Kernel dispatch descriptor ---------- */
typedef struct {
    const char*      kernel_name;     /* PTX / CUBIN symbol name */
    void*            kernel_func;     /* loaded module function pointer */
    int              grid_x, grid_y, grid_z;
    int              block_x, block_y, block_z;
    size_t           shared_mem_bytes;
    ArixCUDAStream*  stream;
} ArixCUDAKernelLaunch;

/* ---------- Driver registration ---------- */
/* Called once at arix_init() to make the CUDA driver available.
 * Registers device query, memory ops, and kernel dispatch in the
 * global driver registry (declared in kernel/arch.c). */
int arix_cuda_register_driver(void);

/* ---------- Device lifecycle ---------- */
int  arix_cuda_get_device_count(int* count);                /* v1.0 */
int  arix_cuda_get_device_props(int dev_id, ArixCUDADeviceProps* props);
int  arix_cuda_set_device(int dev_id);                      /* v1.0 */
int  arix_cuda_get_device(int* dev_id);

/* ---------- Context management ---------- */
ArixCUDAContext* arix_cuda_create_context(int device_id);   /* v1.0 */
void             arix_cuda_destroy_context(ArixCUDAContext* ctx);
int              arix_cuda_context_error(const ArixCUDAContext* ctx);

/* ---------- Stream / event ---------- */
int  arix_cuda_stream_create(ArixCUDAStream** stream, int priority);      /* v1.0 */
void arix_cuda_stream_destroy(ArixCUDAStream* stream);
int  arix_cuda_stream_synchronize(ArixCUDAStream* stream);                /* v1.0 */
int  arix_cuda_event_create(ArixCUDAEvent** event);
void arix_cuda_event_destroy(ArixCUDAEvent* event);
int  arix_cuda_event_record(ArixCUDAEvent* event, ArixCUDAStream* stream);
int  arix_cuda_event_synchronize(ArixCUDAEvent* event);
int  arix_cuda_event_elapsed_ms(float* ms, ArixCUDAEvent* start, ArixCUDAEvent* end);

/* ---------- Memory management (device) ---------- */
int  arix_cuda_mem_alloc(void** dev_ptr, size_t bytes);                    /* v1.0 */
int  arix_cuda_mem_free(void* dev_ptr);                                    /* v1.0 */
int  arix_cuda_mem_htod(void* dev_dst, const void* host_src, size_t bytes); /* v1.0 */
int  arix_cuda_mem_dtoh(void* host_dst, const void* dev_src, size_t bytes);
int  arix_cuda_mem_dtod(void* dev_dst, const void* dev_src, size_t bytes); /* v1.0 */
int  arix_cuda_mem_set(void* dev_ptr, int value, size_t bytes);            /* v1.0 */

/* ---------- Kernel dispatch ---------- */
int  arix_cuda_launch_kernel(const ArixCUDAKernelLaunch* launch);          /* v1.0 */
int  arix_cuda_launch_kernel_async(const ArixCUDAKernelLaunch* launch);    /* v1.0 */

/* ---------- Tensor-core wrappers (v1.0) ---------- */
typedef struct {
    int    m, n, k;
    void*  a;                /* device ptr */
    void*  b;
    void*  c;
    int    lda, ldb, ldc;
    int    dtype;            /* 0 = f16, 1 = bf16, 2 = tf32 */
} ArixCUDATensorCoreGemm;

int arix_cuda_tc_gemm(const ArixCUDATensorCoreGemm* desc, ArixCUDAStream* stream);

/* ---------- Warp-level primitives (v1.0) ---------- */
uint32_t arix_cuda_warp_ballot(int predicate);
uint32_t arix_cuda_warp_reduce_sum(uint32_t value);
float    arix_cuda_warp_reduce_sum_f32(float value);

/* ---------- Error handling ---------- */
const char* arix_cuda_error_string(int error_code);

#ifdef __cplusplus
}
#endif

#endif /* ARIX_CUDA_DRIVER_H */
