#ifndef ARIX_ROCM_DRIVER_H
#define ARIX_ROCM_DRIVER_H
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

typedef struct ArixTensor ArixTensor;

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
} ArixROCmDeviceProps;

typedef struct ArixROCmStream {
    void*    handle;
    int      device_id;
} ArixROCmStream;

typedef struct ArixROCmEvent {
    void*    handle;
    int      device_id;
} ArixROCmEvent;

typedef struct {
    int                     device_id;
    ArixROCmDeviceProps     props;
    size_t                  alloc_bytes;
    size_t                  peak_alloc_bytes;
    ArixROCmStream**        streams;
    int                     num_streams;
    void*                   blas_handle;
    void*                   dnn_handle;
    int                     error_state;
} ArixROCmContext;

typedef struct {
    const char*      kernel_name;
    void*            kernel_func;
    int              grid_x, grid_y, grid_z;
    int              group_x, group_y, group_z;
    size_t           local_mem_bytes;
    ArixROCmStream*  stream;
} ArixROCmKernelLaunch;

int arix_rocm_register_driver(void);

int  arix_rocm_get_device_count(int* count);
int  arix_rocm_get_device_props(int dev_id, ArixROCmDeviceProps* props);
int  arix_rocm_set_device(int dev_id);

ArixROCmContext* arix_rocm_create_context(int device_id);
void             arix_rocm_destroy_context(ArixROCmContext* ctx);

int arix_rocm_stream_create(ArixROCmStream** stream);
void arix_rocm_stream_destroy(ArixROCmStream* stream);
int arix_rocm_stream_synchronize(ArixROCmStream* stream);
int arix_rocm_event_create(ArixROCmEvent** event);
void arix_rocm_event_destroy(ArixROCmEvent* event);
int arix_rocm_event_record(ArixROCmEvent* event, ArixROCmStream* stream);
int arix_rocm_event_synchronize(ArixROCmEvent* event);

int arix_rocm_mem_alloc(void** dev_ptr, size_t bytes);
int arix_rocm_mem_free(void* dev_ptr);
int arix_rocm_mem_htod(void* dev_dst, const void* host_src, size_t bytes);
int arix_rocm_mem_dtoh(void* host_dst, const void* dev_src, size_t bytes);
int arix_rocm_mem_dtod(void* dev_dst, const void* dev_src, size_t bytes);

int arix_rocm_launch_kernel(const ArixROCmKernelLaunch* launch);
int arix_rocm_launch_kernel_async(const ArixROCmKernelLaunch* launch);

#ifdef __cplusplus
}
#endif

#endif /* ARIX_ROCM_DRIVER_H */
