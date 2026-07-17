#include "tpu_driver.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
  #include <windows.h>
  #define TPU_DLOPEN(path) LoadLibraryA(path)
  #define TPU_DLSYM(handle, name) GetProcAddress((HMODULE)handle, name)
  #define TPU_DLCLOSE(handle) FreeLibrary((HMODULE)handle)
#else
  #include <dlfcn.h>
  #define TPU_DLOPEN(path) dlopen(path, RTLD_LAZY | RTLD_LOCAL)
  #define TPU_DLSYM(handle, name) dlsym(handle, name)
  #define TPU_DLCLOSE(handle) dlclose(handle)
#endif

static void* pjrt_lib = NULL;
static int pjrt_loaded = 0;
static int tpu_device_count = 0;

static void* host_malloc(size_t bytes) {
#ifdef _WIN32
    return VirtualAlloc(NULL, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    void* p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) return NULL;
    return p;
#endif
}

static void host_free(void* ptr) {
#ifdef _WIN32
    VirtualFree(ptr, 0, MEM_RELEASE);
#else
    free(ptr);
#endif
}

typedef struct TPUBuffer {
    void* data;
    size_t size;
    int is_device;
} TPUBuffer;

static TPUBuffer* tpu_buffers = NULL;
static int tpu_buffer_count = 0;
static int tpu_buffer_capacity = 0;

static TPUBuffer* find_buffer(void* dev_ptr) {
    for (int i = 0; i < tpu_buffer_count; i++)
        if (tpu_buffers[i].data == dev_ptr) return &tpu_buffers[i];
    return NULL;
}

static TPUBuffer* add_buffer(void* data, size_t size, int is_device) {
    if (tpu_buffer_count >= tpu_buffer_capacity) {
        int new_cap = tpu_buffer_capacity == 0 ? 64 : tpu_buffer_capacity * 2;
        TPUBuffer* new_buf = (TPUBuffer*)realloc(tpu_buffers, new_cap * sizeof(TPUBuffer));
        if (!new_buf) return NULL;
        tpu_buffers = new_buf;
        tpu_buffer_capacity = new_cap;
    }
    TPUBuffer* b = &tpu_buffers[tpu_buffer_count++];
    b->data = data;
    b->size = size;
    b->is_device = is_device;
    return b;
}

int SNEPPX_tpu_register_driver(void) {
    (void)pjrt_lib;
    (void)pjrt_loaded;
    tpu_device_count = 1;
    return 0;
}

int SNEPPX_tpu_get_device_count(int* count) {
    if (!count) return -1;
    *count = tpu_device_count > 0 ? tpu_device_count : 1;
    return 0;
}

int SNEPPX_tpu_get_device_props(int dev_id, SNEPPXTPUDeviceProps* props) {
    if (!props) return -1;
    memset(props, 0, sizeof(*props));
    (void)dev_id;
    snprintf(props->name, sizeof(props->name), "TPU v4 (emulated, device %d)", dev_id);
    props->device_memory_bytes = 16ULL * 1024 * 1024 * 1024;
    props->num_cores = 8;
    props->num_chips = 4;
    props->topology_ring_size = 4;
    props->supports_bfloat16 = 1;
    props->supports_int8 = 1;
    props->supports_sparse_core = 0;
    return 0;
}

SNEPPXTPUContext* SNEPPX_tpu_create_context(int device_id) {
    SNEPPXTPUContext* ctx = (SNEPPXTPUContext*)calloc(1, sizeof(SNEPPXTPUContext));
    if (!ctx) return NULL;
    ctx->device_id = device_id;
    ctx->client = (SNEPPXTPUClient*)calloc(1, sizeof(SNEPPXTPUClient));
    if (ctx->client) {
        ctx->client->device_id = device_id;
        ctx->client->pjrt_client = NULL;
    }
    ctx->alloc_bytes = 0;
    ctx->error_state = 0;
    SNEPPX_tpu_get_device_props(device_id, &ctx->props);
    return ctx;
}

void SNEPPX_tpu_destroy_context(SNEPPXTPUContext* ctx) {
    if (!ctx) return;
    free(ctx->client);
    free(ctx);
}

int SNEPPX_tpu_mem_alloc(void** dev_ptr, size_t bytes, SNEPPXTPUContext* ctx) {
    if (!dev_ptr || !ctx) return -1;
    (void)ctx;
    void* ptr = host_malloc(bytes);
    if (!ptr) return -1;
    if (!add_buffer(ptr, bytes, 1)) {
        host_free(ptr);
        return -1;
    }
    ctx->alloc_bytes += bytes;
    *dev_ptr = ptr;
    return 0;
}

int SNEPPX_tpu_mem_free(void* dev_ptr, SNEPPXTPUContext* ctx) {
    if (!dev_ptr || !ctx) return -1;
    (void)ctx;
    TPUBuffer* buf = find_buffer(dev_ptr);
    if (buf) { ctx->alloc_bytes -= buf->size; buf->data = NULL; }
    host_free(dev_ptr);
    return 0;
}

int SNEPPX_tpu_mem_htod(void* dev_dst, const void* host_src, size_t bytes, SNEPPXTPUContext* ctx) {
    if (!dev_dst || !host_src || bytes == 0 || !ctx) return -1;
    (void)ctx;
    memcpy(dev_dst, host_src, bytes);
    return 0;
}

int SNEPPX_tpu_mem_dtoh(void* host_dst, const void* dev_src, size_t bytes, SNEPPXTPUContext* ctx) {
    if (!host_dst || !dev_src || bytes == 0 || !ctx) return -1;
    (void)ctx;
    memcpy(host_dst, dev_src, bytes);
    return 0;
}

SNEPPXTPUExecutable* SNEPPX_tpu_compile(const char* hlo_module, size_t hlo_len, SNEPPXTPUContext* ctx) {
    if (!hlo_module || hlo_len == 0 || !ctx) return NULL;
    (void)hlo_module;
    SNEPPXTPUExecutable* exec = (SNEPPXTPUExecutable*)calloc(1, sizeof(SNEPPXTPUExecutable));
    if (!exec) return NULL;
    exec->input_count = 2;
    exec->output_count = 1;
    exec->pjrt_executable = NULL;
    return exec;
}

void SNEPPX_tpu_executable_destroy(SNEPPXTPUExecutable* exec) {
    free(exec);
}

int SNEPPX_tpu_execute(SNEPPXTPUExecutable* exec, SNEPPXTensor** inputs, size_t num_inputs,
                     SNEPPXTensor** outputs, size_t num_outputs, SNEPPXTPUContext* ctx) {
    if (!exec || !inputs || num_inputs == 0 || !outputs || num_outputs == 0 || !ctx) return -1;
    (void)exec; (void)inputs; (void)num_inputs; (void)outputs; (void)num_outputs; (void)ctx;
    return 0;
}

int SNEPPX_tpu_all_reduce(void* send_buf, void* recv_buf, size_t count,
                        int dtype, int reduce_op, SNEPPXTPUContext* ctx) {
    if (!send_buf || !recv_buf || count == 0 || !ctx) return -1;
    (void)dtype; (void)reduce_op;
    size_t elem_size = 4;
    memcpy(recv_buf, send_buf, count * elem_size);
    return 0;
}
