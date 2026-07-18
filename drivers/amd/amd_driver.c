#include "amd_driver.h"
#include "neural_core/drivers/driver_status.h"
#include <stdlib.h>

int SNEPPX_amd_register(void) { return SNEPPX_DRIVER_UNSUPPORTED; }
int SNEPPX_amd_get_device_count(int* count) { if (count) *count = 0; return SNEPPX_DRIVER_UNSUPPORTED; }
int SNEPPX_amd_get_device_props(int dev_id, char* name, size_t name_max, size_t* global_mem, int* compute_units) { (void)dev_id; if (name) snprintf(name, name_max, "AMD Device %d", dev_id); if (global_mem) *global_mem = 16ULL*1024*1024*1024; if (compute_units) *compute_units = 40; return 0; }
void* SNEPPX_amd_create_context(int device_id) { (void)device_id; return calloc(1, 8); }
void SNEPPX_amd_destroy_context(void* ctx) { free(ctx); }
int SNEPPX_amd_mem_alloc(void** dev_ptr, size_t bytes) { (void)dev_ptr; (void)bytes; return 0; }
int SNEPPX_amd_mem_free(void* dev_ptr) { (void)dev_ptr; return 0; }
int SNEPPX_amd_memcpy_htod(void* dev_dst, const void* host_src, size_t bytes) { (void)dev_dst; (void)host_src; (void)bytes; return 0; }
int SNEPPX_amd_memcpy_dtoh(void* host_dst, const void* dev_src, size_t bytes) { (void)host_dst; (void)dev_src; (void)bytes; return 0; }
int SNEPPX_amd_launch_kernel(const char* kernel_name, void* queue, void** args, size_t num_args, size_t global_size, size_t local_size) { (void)kernel_name; (void)queue; (void)args; (void)num_args; (void)global_size; (void)local_size; return 0; }
