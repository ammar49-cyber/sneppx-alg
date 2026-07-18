#ifndef SNEPPX_VULKAN_COMPUTE_H
#define SNEPPX_VULKAN_COMPUTE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Vulkan compute backend. When SNEPPX_BUILD_VULKAN is defined the backend
 * performs genuine computation through the portable reference-compute kernels
 * (see neural_core/drivers/reference_compute.h) so it is not a silent no-op.
 * Without the flag the entry points report SNEPPX_DRIVER_UNSUPPORTED. */

int  SNEPPX_vulkan_init(void);
void SNEPPX_vulkan_cleanup(void);
int  SNEPPX_vulkan_get_device_count(int* count);
int  SNEPPX_vulkan_get_device_props(int dev_id, char* name, size_t name_max, size_t* global_mem);

void* SNEPPX_vulkan_create_compute_pipeline(const char* shader_path, const char* entry_point);
void  SNEPPX_vulkan_destroy_compute_pipeline(void* pipeline);

int SNEPPX_vulkan_create_buffer(void** buf, size_t size, int usage, void* pipeline);
int SNEPPX_vulkan_destroy_buffer(void* buf, void* pipeline);
int SNEPPX_vulkan_write_buffer(void* buf, const void* data, size_t offset, size_t size, void* pipeline);
int SNEPPX_vulkan_read_buffer(void* dst, const void* buf, size_t offset, size_t size, void* pipeline);

/* Dispatch a compute kernel over `buffers` (float* array of `num_buffers`).
 * `entry_point` selects the operation ("gemm", "relu", "sigmoid", "tanh", ...).
 * `group_x` is the problem dimension (e.g. matrix size for gemm). */
int SNEPPX_vulkan_dispatch(void* pipeline, void** buffers, size_t num_buffers,
                           unsigned int group_x, unsigned int group_y, unsigned int group_z,
                           const char* entry_point);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_VULKAN_COMPUTE_H */
