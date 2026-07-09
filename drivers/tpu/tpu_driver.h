#ifndef SNEPPX_TPU_DRIVER_H
#define SNEPPX_TPU_DRIVER_H
/*
 * TPU Driver Interface — v2.0 (Google TPU / custom ASIC acceleration)
 *
 * PURPOSE: Abstract TPU and NPU devices behind the kernel's device ops.
 * TPUs use a different programming model (XLA / PjRt) than GPUs; this
 * interface wraps the PjRt client, executed programs, and device memory.
 *
 * DEPENDENCIES: polymorphic_memory_allocator.h, multidimensional_tensor_engine.h
 * VERSION: v2.0 — production-scale distributed training
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SNEPPXTensor SNEPPXTensor;

/* ---------- TPU device capabilities ---------- */
typedef struct {
    char     name[256];
    size_t   device_memory_bytes;
    int      num_cores;
    int      num_chips;
    int      topology_ring_size;       /* for all-reduce ring */
    int      supports_bfloat16;
    int      supports_int8;
    int      supports_sparse_core;
} SNEPPXTPUDeviceProps;

/* ---------- PjRt-style client / executable handles (opaque) ---------- */
typedef struct SNEPPXTPUClient {
    void*    pjrt_client;             /* PJRT_Client* */
    int      device_id;
} SNEPPXTPUClient;

typedef struct SNEPPXTPUExecutable {
    void*    pjrt_executable;         /* PJRT_Executable* */
    size_t   input_count;
    size_t   output_count;
} SNEPPXTPUExecutable;

typedef struct {
    int                device_id;
    SNEPPXTPUDeviceProps props;
    SNEPPXTPUClient*     client;
    size_t             alloc_bytes;
    int                error_state;
} SNEPPXTPUContext;

/* ---------- Driver lifecycle ---------- */
int  SNEPPX_tpu_register_driver(void);
int  SNEPPX_tpu_get_device_count(int* count);
int  SNEPPX_tpu_get_device_props(int dev_id, SNEPPXTPUDeviceProps* props);
SNEPPXTPUContext* SNEPPX_tpu_create_context(int device_id);
void            SNEPPX_tpu_destroy_context(SNEPPXTPUContext* ctx);

/* ---------- Memory ---------- */
int SNEPPX_tpu_mem_alloc(void** dev_ptr, size_t bytes, SNEPPXTPUContext* ctx);
int SNEPPX_tpu_mem_free(void* dev_ptr, SNEPPXTPUContext* ctx);
int SNEPPX_tpu_mem_htod(void* dev_dst, const void* host_src, size_t bytes, SNEPPXTPUContext* ctx);
int SNEPPX_tpu_mem_dtoh(void* host_dst, const void* dev_src, size_t bytes, SNEPPXTPUContext* ctx);

/* ---------- Compilation and execution (XLA) ---------- */
SNEPPXTPUExecutable* SNEPPX_tpu_compile(const char* hlo_module, size_t hlo_len, SNEPPXTPUContext* ctx);
void               SNEPPX_tpu_executable_destroy(SNEPPXTPUExecutable* exec);
int                SNEPPX_tpu_execute(SNEPPXTPUExecutable* exec,
                                    SNEPPXTensor** inputs, size_t num_inputs,
                                    SNEPPXTensor** outputs, size_t num_outputs,
                                    SNEPPXTPUContext* ctx);

/* ---------- Collective operations (v2.0) ---------- */
int SNEPPX_tpu_all_reduce(void* send_buf, void* recv_buf, size_t count,
                        int dtype, int reduce_op, SNEPPXTPUContext* ctx);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_TPU_DRIVER_H */
