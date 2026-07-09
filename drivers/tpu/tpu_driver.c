/*
 * TPU Driver Implementation — SKELETON
 * VERSION: v2.0
 */

#include "tpu_driver.h"
#include <stdlib.h>
#include <string.h>

int SNEPPX_tpu_register_driver(void) { return 0; }

int SNEPPX_tpu_get_device_count(int* count) { if (count) *count = 0; return 0; }

int SNEPPX_tpu_get_device_props(int dev_id, SNEPPXTPUDeviceProps* props) {
    (void)dev_id; if (props) memset(props, 0, sizeof(*props)); return 0;
}

SNEPPXTPUContext* SNEPPX_tpu_create_context(int device_id) {
    (void)device_id;
    SNEPPXTPUContext* ctx = (SNEPPXTPUContext*)calloc(1, sizeof(SNEPPXTPUContext));
    return ctx;
}

void SNEPPX_tpu_destroy_context(SNEPPXTPUContext* ctx) { free(ctx); }

int SNEPPX_tpu_mem_alloc(void** dev_ptr, size_t bytes, SNEPPXTPUContext* ctx) {
    (void)ctx; if (!dev_ptr) return -1; (void)bytes; *dev_ptr = NULL; return 0;
}

int SNEPPX_tpu_mem_free(void* dev_ptr, SNEPPXTPUContext* ctx) {
    (void)dev_ptr; (void)ctx; return 0;
}

int SNEPPX_tpu_mem_htod(void* dev_dst, const void* host_src, size_t bytes, SNEPPXTPUContext* ctx) {
    (void)dev_dst; (void)host_src; (void)bytes; (void)ctx; return 0;
}

int SNEPPX_tpu_mem_dtoh(void* host_dst, const void* dev_src, size_t bytes, SNEPPXTPUContext* ctx) {
    (void)host_dst; (void)dev_src; (void)bytes; (void)ctx; return 0;
}

SNEPPXTPUExecutable* SNEPPX_tpu_compile(const char* hlo_module, size_t hlo_len, SNEPPXTPUContext* ctx) {
    (void)hlo_module; (void)hlo_len; (void)ctx;
    return NULL;
}

void SNEPPX_tpu_executable_destroy(SNEPPXTPUExecutable* exec) { free(exec); }

int SNEPPX_tpu_execute(SNEPPXTPUExecutable* exec, SNEPPXTensor** inputs, size_t num_inputs,
                     SNEPPXTensor** outputs, size_t num_outputs, SNEPPXTPUContext* ctx) {
    (void)exec; (void)inputs; (void)num_inputs; (void)outputs; (void)num_outputs; (void)ctx;
    return 0;
}

int SNEPPX_tpu_all_reduce(void* send_buf, void* recv_buf, size_t count,
                        int dtype, int reduce_op, SNEPPXTPUContext* ctx) {
    (void)send_buf; (void)recv_buf; (void)count; (void)dtype; (void)reduce_op; (void)ctx;
    return 0;
}
