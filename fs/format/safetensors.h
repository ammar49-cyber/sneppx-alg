#ifndef SNEPPX_SAFETENSORS_H
#define SNEPPX_SAFETENSORS_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Tensor dtype codes (mirror the safetensors dtype strings). */
typedef enum {
    SNEPPX_ST_DTYPE_F32 = 0,
    SNEPPX_ST_DTYPE_F16,
    SNEPPX_ST_DTYPE_F64,
    SNEPPX_ST_DTYPE_I64,
    SNEPPX_ST_DTYPE_I32,
    SNEPPX_ST_DTYPE_I16,
    SNEPPX_ST_DTYPE_I8,
    SNEPPX_ST_DTYPE_U8,
    SNEPPX_ST_DTYPE_BOOL,
    SNEPPX_ST_DTYPE_BF16,
    SNEPPX_ST_DTYPE_F8_E5M2,
    SNEPPX_ST_DTYPE_F8_E4M3FN
} SNEPPXSafetensorsDType;

void* SNEPPX_safetensors_open(const char* path, const char* mode);
void SNEPPX_safetensors_close(void* st);
int SNEPPX_safetensors_get_tensor_count(void* st);
int SNEPPX_safetensors_get_tensor_names(void* st, char*** names, size_t* count);
void* SNEPPX_safetensors_read_tensor(void* st, const char* name, size_t* ndim, size_t** shape, int* dtype);
int SNEPPX_safetensors_write_tensor(void* st, const char* name, const void* data, const size_t* shape, size_t ndim, int dtype);
int SNEPPX_safetensors_save(void* st);
unsigned long long SNEPPX_safetensors_get_metadata(void* st, const char* key, char* value, size_t value_max);

/* Element size in bytes for a dtype code (0 if unknown). */
size_t SNEPPX_safetensors_dtype_size(int dtype);

#ifdef __cplusplus
}
#endif
#endif
