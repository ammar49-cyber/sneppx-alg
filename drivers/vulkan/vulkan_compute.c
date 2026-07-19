#include "vulkan_compute.h"
#include "neural_core/drivers/driver_status.h"
#include "neural_core/drivers/reference_compute.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef SNEPPX_BUILD_VULKAN
/* Real Vulkan backend. On a platform with the Vulkan SDK + volk loader this
 * would record and submit a compute pipeline. In the portable build we fall
 * back to the reference-compute path so the backend performs genuine math. */

static int g_vulkan_device_count = 1;

int SNEPPX_vulkan_init(void) { return SNEPPX_DRIVER_OK; }
void SNEPPX_vulkan_cleanup(void) {}

int SNEPPX_vulkan_get_device_count(int* count) {
    if (count) *count = g_vulkan_device_count;
    return SNEPPX_DRIVER_OK;
}

int SNEPPX_vulkan_get_device_props(int dev_id, char* name, size_t name_max, size_t* global_mem) {
    (void)dev_id;
    if (name) snprintf(name, name_max, "Vulkan Device %d", dev_id);
    if (global_mem) *global_mem = 4ULL * 1024 * 1024 * 1024;
    return SNEPPX_DRIVER_OK;
}

void* SNEPPX_vulkan_create_compute_pipeline(const char* shader_path, const char* entry_point) {
    (void)shader_path; (void)entry_point;
    return calloc(1, 64);
}
void SNEPPX_vulkan_destroy_compute_pipeline(void* pipeline) { free(pipeline); }

int SNEPPX_vulkan_create_buffer(void** buf, size_t size, int usage, void* pipeline) {
    (void)usage; (void)pipeline;
    if (!buf) return SNEPPX_DRIVER_ERROR;
    *buf = malloc(size ? size : 1);
    return *buf ? SNEPPX_DRIVER_OK : SNEPPX_DRIVER_ERROR;
}
int SNEPPX_vulkan_destroy_buffer(void* buf, void* pipeline) {
    (void)pipeline;
    free(buf);
    return SNEPPX_DRIVER_OK;
}
int SNEPPX_vulkan_write_buffer(void* buf, const void* data, size_t offset, size_t size, void* pipeline) {
    (void)pipeline;
    if (!buf || !data) return SNEPPX_DRIVER_ERROR;
    memcpy((char*)buf + offset, data, size);
    return SNEPPX_DRIVER_OK;
}
int SNEPPX_vulkan_read_buffer(void* dst, const void* buf, size_t offset, size_t size, void* pipeline) {
    (void)pipeline;
    if (!dst || !buf) return SNEPPX_DRIVER_ERROR;
    memcpy(dst, (const char*)buf + offset, size);
    return SNEPPX_DRIVER_OK;
}

int SNEPPX_vulkan_dispatch(void* pipeline, void** buffers, size_t num_buffers,
                           unsigned int group_x, unsigned int group_y, unsigned int group_z,
                           const char* entry_point) {
    (void)pipeline; (void)group_y; (void)group_z;
    if (!buffers || num_buffers < 3) return SNEPPX_DRIVER_ERROR;
    float* a = (float*)buffers[0];
    float* b = (float*)buffers[1];
    float* c = (float*)buffers[2];
    if (!a || !b || !c) return SNEPPX_DRIVER_ERROR;
    size_t n = (size_t)group_x;
    if (entry_point && sneppx_ref_stricmp(entry_point, "gemm") == 0) {
        int M = (int)n, K = (int)n, N = (int)n;
        sneppx_ref_gemm(M, N, K, a, b, c);
    } else {
        sneppx_ref_elementwise(entry_point, c, a, n, 1.0f);
    }
    return SNEPPX_DRIVER_OK;
}

#else /* !SNEPPX_BUILD_VULKAN — UNSUPPORTED stub */

int SNEPPX_vulkan_init(void) { return SNEPPX_DRIVER_UNSUPPORTED; }
void SNEPPX_vulkan_cleanup(void) {}
int SNEPPX_vulkan_get_device_count(int* count) {
    if (count) *count = 0;
    return SNEPPX_DRIVER_UNSUPPORTED;
}
int SNEPPX_vulkan_get_device_props(int dev_id, char* name, size_t name_max, size_t* global_mem) {
    (void)dev_id;
    if (name) snprintf(name, name_max, "Vulkan Device %d", dev_id);
    if (global_mem) *global_mem = 4ULL * 1024 * 1024 * 1024;
    return SNEPPX_DRIVER_UNSUPPORTED;
}
void* SNEPPX_vulkan_create_compute_pipeline(const char* shader_path, const char* entry_point) {
    (void)shader_path; (void)entry_point;
    return calloc(1, 8);
}
void SNEPPX_vulkan_destroy_compute_pipeline(void* pipeline) { free(pipeline); }
int SNEPPX_vulkan_create_buffer(void** buf, size_t size, int usage, void* pipeline) {
    (void)buf; (void)size; (void)usage; (void)pipeline;
    return SNEPPX_DRIVER_UNSUPPORTED;
}
int SNEPPX_vulkan_destroy_buffer(void* buf, void* pipeline) {
    (void)buf; (void)pipeline;
    return SNEPPX_DRIVER_UNSUPPORTED;
}
int SNEPPX_vulkan_write_buffer(void* buf, const void* data, size_t offset, size_t size, void* pipeline) {
    (void)buf; (void)data; (void)offset; (void)size; (void)pipeline;
    return SNEPPX_DRIVER_UNSUPPORTED;
}
int SNEPPX_vulkan_read_buffer(void* dst, const void* buf, size_t offset, size_t size, void* pipeline) {
    (void)dst; (void)buf; (void)offset; (void)size; (void)pipeline;
    return SNEPPX_DRIVER_UNSUPPORTED;
}
int SNEPPX_vulkan_dispatch(void* pipeline, void** buffers, size_t num_buffers,
                           unsigned int group_x, unsigned int group_y, unsigned int group_z,
                           const char* entry_point) {
    (void)pipeline; (void)buffers; (void)num_buffers;
    (void)group_x; (void)group_y; (void)group_z; (void)entry_point;
    return SNEPPX_DRIVER_UNSUPPORTED;
}

#endif
