#ifndef SNEPPX_ROCM_DRIVER_H
#define SNEPPX_ROCM_DRIVER_H
/*
 * ROCm Driver Interface — v1.0 (AMD GPU acceleration)
 *
 * PURPOSE: Abstract AMD GPU devices behind the kernel's device operations
 * structure.  Mirrors the CUDA driver interface so that tensor operations
 * can be dispatched to either NVIDIA or AMD hardware transparently.
 *
 * DEPENDENCIES: polymorphic_memory_allocator.h, multidimensional_tensor_engine.h
 * VERSION: v1.0
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SNEPPXTensor SNEPPXTensor;

typedef struct {
    char     name[256];
    size_t   global_mem_bytes;
    size_t   shared_mem_per_group;
    int      wavefront_size;
    int      max_threads_per_workgroup;
    int      max_workgroup_size[3];
    int      gcn_arch_major;
    int      gcn_arch_minor;
    int      num_cus;
    int      supports_matrix_core;
} SNEPPXROCmDeviceProps;

typedef struct SNEPPXROCmStream {
    void*    handle;
    int      device_id;
} SNEPPXROCmStream;

typedef struct SNEPPXROCmEvent {
    void*    handle;
    int      device_id;
} SNEPPXROCmEvent;

typedef struct {
    int                     device_id;
    SNEPPXROCmDeviceProps     props;
    size_t                  alloc_bytes;
    size_t                  peak_alloc_bytes;
    SNEPPXROCmStream**        streams;
    int                     num_streams;
    void*                   blas_handle;
    void*                   dnn_handle;
    int                     error_state;
} SNEPPXROCmContext;

typedef struct {
    const char*      kernel_name;
    void*            kernel_func;
    int              grid_x, grid_y, grid_z;
    int              group_x, group_y, group_z;
    size_t           local_mem_bytes;
    SNEPPXROCmStream*  stream;
} SNEPPXROCmKernelLaunch;

int SNEPPX_rocm_register_driver(void);

int  SNEPPX_rocm_get_device_count(int* count);
int  SNEPPX_rocm_get_device_props(int dev_id, SNEPPXROCmDeviceProps* props);
int  SNEPPX_rocm_set_device(int dev_id);

SNEPPXROCmContext* SNEPPX_rocm_create_context(int device_id);
void             SNEPPX_rocm_destroy_context(SNEPPXROCmContext* ctx);

int SNEPPX_rocm_stream_create(SNEPPXROCmStream** stream);
void SNEPPX_rocm_stream_destroy(SNEPPXROCmStream* stream);
int SNEPPX_rocm_stream_synchronize(SNEPPXROCmStream* stream);
int SNEPPX_rocm_event_create(SNEPPXROCmEvent** event);
void SNEPPX_rocm_event_destroy(SNEPPXROCmEvent* event);
int SNEPPX_rocm_event_record(SNEPPXROCmEvent* event, SNEPPXROCmStream* stream);
int SNEPPX_rocm_event_synchronize(SNEPPXROCmEvent* event);

int SNEPPX_rocm_mem_alloc(void** dev_ptr, size_t bytes);
int SNEPPX_rocm_mem_free(void* dev_ptr);
int SNEPPX_rocm_mem_htod(void* dev_dst, const void* host_src, size_t bytes);
int SNEPPX_rocm_mem_dtoh(void* host_dst, const void* dev_src, size_t bytes);
int SNEPPX_rocm_mem_dtod(void* dev_dst, const void* dev_src, size_t bytes);

int SNEPPX_rocm_launch_kernel(const SNEPPXROCmKernelLaunch* launch);
int SNEPPX_rocm_launch_kernel_async(const SNEPPXROCmKernelLaunch* launch);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_ROCM_DRIVER_H */
