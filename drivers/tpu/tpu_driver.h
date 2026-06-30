#ifndef ARIX_TPU_DRIVER_H
#define ARIX_TPU_DRIVER_H
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

typedef struct ArixTensor ArixTensor;

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
} ArixTPUDeviceProps;

/* ---------- PjRt-style client / executable handles (opaque) ---------- */
typedef struct ArixTPUClient {
    void*    pjrt_client;             /* PJRT_Client* */
    int      device_id;
} ArixTPUClient;

typedef struct ArixTPUExecutable {
    void*    pjrt_executable;         /* PJRT_Executable* */
    size_t   input_count;
    size_t   output_count;
} ArixTPUExecutable;

typedef struct {
    int                device_id;
    ArixTPUDeviceProps props;
    ArixTPUClient*     client;
    size_t             alloc_bytes;
    int                error_state;
} ArixTPUContext;

/* ---------- Driver lifecycle ---------- */
int  arix_tpu_register_driver(void);
int  arix_tpu_get_device_count(int* count);
int  arix_tpu_get_device_props(int dev_id, ArixTPUDeviceProps* props);
ArixTPUContext* arix_tpu_create_context(int device_id);
void            arix_tpu_destroy_context(ArixTPUContext* ctx);

/* ---------- Memory ---------- */
int arix_tpu_mem_alloc(void** dev_ptr, size_t bytes, ArixTPUContext* ctx);
int arix_tpu_mem_free(void* dev_ptr, ArixTPUContext* ctx);
int arix_tpu_mem_htod(void* dev_dst, const void* host_src, size_t bytes, ArixTPUContext* ctx);
int arix_tpu_mem_dtoh(void* host_dst, const void* dev_src, size_t bytes, ArixTPUContext* ctx);

/* ---------- Compilation and execution (XLA) ---------- */
ArixTPUExecutable* arix_tpu_compile(const char* hlo_module, size_t hlo_len, ArixTPUContext* ctx);
void               arix_tpu_executable_destroy(ArixTPUExecutable* exec);
int                arix_tpu_execute(ArixTPUExecutable* exec,
                                    ArixTensor** inputs, size_t num_inputs,
                                    ArixTensor** outputs, size_t num_outputs,
                                    ArixTPUContext* ctx);

/* ---------- Collective operations (v2.0) ---------- */
int arix_tpu_all_reduce(void* send_buf, void* recv_buf, size_t count,
                        int dtype, int reduce_op, ArixTPUContext* ctx);

#ifdef __cplusplus
}
#endif

#endif /* ARIX_TPU_DRIVER_H */
