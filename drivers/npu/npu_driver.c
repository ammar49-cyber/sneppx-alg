#include "npu_driver.h"
#include "neural_core/drivers/driver_status.h"
#include <stdlib.h>

int SNEPPX_npu_register_driver(void) { return SNEPPX_DRIVER_UNSUPPORTED; }
int SNEPPX_npu_get_device_count(int* count) { if (count) *count = 0; return SNEPPX_DRIVER_UNSUPPORTED; }
int SNEPPX_npu_get_device_props(int dev_id, void* props) { (void)dev_id; if (props) __builtin_memset(props, 0, 128); return 0; }
void* SNEPPX_npu_create_context(int device_id) { (void)device_id; return calloc(1, 64); }
void SNEPPX_npu_destroy_context(void* ctx) { free(ctx); }
int SNEPPX_npu_mem_alloc(void** dev_ptr, size_t bytes, void* ctx) { (void)dev_ptr; (void)bytes; (void)ctx; return 0; }
int SNEPPX_npu_mem_free(void* dev_ptr, void* ctx) { (void)dev_ptr; (void)ctx; return 0; }
int SNEPPX_npu_mem_htod(void* dev_dst, const void* host_src, size_t bytes, void* ctx) { (void)dev_dst; (void)host_src; (void)bytes; (void)ctx; return 0; }
int SNEPPX_npu_mem_dtoh(void* host_dst, const void* dev_src, size_t bytes, void* ctx) { (void)host_dst; (void)dev_src; (void)bytes; (void)ctx; return 0; }
int SNEPPX_npu_execute(void* exec, void** inputs, size_t num_inputs, void** outputs, size_t num_outputs, void* ctx) { (void)exec; (void)inputs; (void)num_inputs; (void)outputs; (void)num_outputs; (void)ctx; return 0; }
