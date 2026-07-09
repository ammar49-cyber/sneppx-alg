/*
 * ROCm Driver Implementation — SKELETON
 *
 * Stub implementations matching rocm_driver.h declarations.
 * VERSION: v1.0
 */

#include "rocm_driver.h"
#include <stdlib.h>
#include <string.h>

int SNEPPX_rocm_register_driver(void) { return 0; }

int SNEPPX_rocm_get_device_count(int* count) { if (count) *count = 0; return 0; }

int SNEPPX_rocm_get_device_props(int dev_id, SNEPPXROCmDeviceProps* props) {
    (void)dev_id; if (props) memset(props, 0, sizeof(*props)); return 0;
}

int SNEPPX_rocm_set_device(int dev_id) { (void)dev_id; return 0; }

SNEPPXROCmContext* SNEPPX_rocm_create_context(int device_id) {
    (void)device_id;
    SNEPPXROCmContext* ctx = (SNEPPXROCmContext*)calloc(1, sizeof(SNEPPXROCmContext));
    return ctx;
}

void SNEPPX_rocm_destroy_context(SNEPPXROCmContext* ctx) { free(ctx); }

int SNEPPX_rocm_stream_create(SNEPPXROCmStream** stream) {
    if (!stream) return -1;
    *stream = (SNEPPXROCmStream*)calloc(1, sizeof(SNEPPXROCmStream));
    return *stream ? 0 : -1;
}

void SNEPPX_rocm_stream_destroy(SNEPPXROCmStream* stream) { free(stream); }

int SNEPPX_rocm_stream_synchronize(SNEPPXROCmStream* stream) { (void)stream; return 0; }

int SNEPPX_rocm_event_create(SNEPPXROCmEvent** event) {
    if (!event) return -1;
    *event = (SNEPPXROCmEvent*)calloc(1, sizeof(SNEPPXROCmEvent));
    return *event ? 0 : -1;
}

void SNEPPX_rocm_event_destroy(SNEPPXROCmEvent* event) { free(event); }

int SNEPPX_rocm_event_record(SNEPPXROCmEvent* event, SNEPPXROCmStream* stream) {
    (void)event; (void)stream; return 0;
}

int SNEPPX_rocm_event_synchronize(SNEPPXROCmEvent* event) { (void)event; return 0; }

int SNEPPX_rocm_mem_alloc(void** dev_ptr, size_t bytes) {
    if (!dev_ptr) return -1; (void)bytes; *dev_ptr = NULL; return 0;
}

int SNEPPX_rocm_mem_free(void* dev_ptr) { (void)dev_ptr; return 0; }

int SNEPPX_rocm_mem_htod(void* dev_dst, const void* host_src, size_t bytes) {
    (void)dev_dst; (void)host_src; (void)bytes; return 0;
}

int SNEPPX_rocm_mem_dtoh(void* host_dst, const void* dev_src, size_t bytes) {
    (void)host_dst; (void)dev_src; (void)bytes; return 0;
}

int SNEPPX_rocm_mem_dtod(void* dev_dst, const void* dev_src, size_t bytes) {
    (void)dev_dst; (void)dev_src; (void)bytes; return 0;
}

int SNEPPX_rocm_launch_kernel(const SNEPPXROCmKernelLaunch* launch) {
    (void)launch; return 0;
}

int SNEPPX_rocm_launch_kernel_async(const SNEPPXROCmKernelLaunch* launch) {
    (void)launch; return 0;
}
