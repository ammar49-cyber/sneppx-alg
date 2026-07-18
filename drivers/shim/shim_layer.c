#include "shim_layer.h"
#include "neural_core/drivers/driver_status.h"
#include <stdlib.h>
#include <string.h>

int SNEPPX_shim_init(const char* backend_type) { (void)backend_type; return SNEPPX_DRIVER_UNSUPPORTED; }
void SNEPPX_shim_cleanup(void) {}
int SNEPPX_shim_get_device_count(int* count) { if (count) *count = 0; return SNEPPX_DRIVER_UNSUPPORTED; }
int SNEPPX_shim_get_device_props(int dev_id, char* name, size_t name_max, int* dev_type, size_t* global_mem) { (void)dev_id; if (name) snprintf(name, name_max, "Shim Device %d", dev_id); if (dev_type) *dev_type = 0; if (global_mem) *global_mem = 4ULL*1024*1024*1024; return 0; }
void* SNEPPX_shim_create_context(int device_id) { (void)device_id; return calloc(1, 8); }
void SNEPPX_shim_destroy_context(void* ctx) { free(ctx); }
int SNEPPX_shim_mem_alloc(void** dev_ptr, size_t bytes, void* ctx) { (void)dev_ptr; (void)bytes; (void)ctx; return 0; }
int SNEPPX_shim_mem_free(void* dev_ptr, void* ctx) { (void)dev_ptr; (void)ctx; return 0; }
int SNEPPX_shim_memcpy_htod(void* dev_dst, const void* host_src, size_t bytes, void* ctx) { (void)dev_dst; (void)host_src; (void)bytes; (void)ctx; return 0; }
int SNEPPX_shim_memcpy_dtoh(void* host_dst, const void* dev_src, size_t bytes, void* ctx) { (void)host_dst; (void)dev_src; (void)bytes; (void)ctx; return 0; }
int SNEPPX_shim_launch_kernel(const char* kernel_name, void* ctx, void** args, size_t num_args, size_t global_size, size_t local_size) { (void)kernel_name; (void)ctx; (void)args; (void)num_args; (void)global_size; (void)local_size; return 0; }
int SNEPPX_shim_synchronize(void* ctx) { (void)ctx; return 0; }
