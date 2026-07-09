#include "multidimensional_tensor_engine.h"
#include "polymorphic_memory_allocator.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>

static size_t compute_offset(const SNEPPXTensor* tensor, const size_t* indices) {
    size_t offset = 0;
    for (size_t i = 0; i < tensor->ndim; i++) {
        offset += indices[i] * tensor->strides[i];
    }
    return offset;
}

static size_t compute_offset_flat(const SNEPPXTensor* tensor, size_t flat_idx) {
    size_t offset = 0;
    size_t remaining = flat_idx;
    for (size_t i = 0; i < tensor->ndim; i++) {
        size_t idx = remaining / (tensor->size / (tensor->strides[i] * tensor->shape[i]));
        if (tensor->size > 0 && tensor->strides[i] > 0) {
            size_t dim_stride = tensor->strides[i];
            size_t dim_size = tensor->shape[i];
            idx = (flat_idx / dim_stride) % dim_size;
        }
        offset += idx * tensor->strides[i];
        if (tensor->strides[i] > 0) {
            remaining = flat_idx % tensor->strides[i];
        } else {
            remaining = 0;
        }
    }
    return offset;
}

static void* aligned_alloc_wrapper(size_t size, size_t alignment) {
    return SNEPPX_malloc(size, alignment);
}

/* ===== SNEPPXStorage implementation ===== */

SNEPPXStorage* SNEPPX_storage_create(size_t num_bytes) {
    SNEPPXStorage* s = (SNEPPXStorage*)aligned_alloc_wrapper(sizeof(SNEPPXStorage), 64);
    if (!s) return NULL;
    s->data = aligned_alloc_wrapper(num_bytes, 64);
    if (!s->data && num_bytes > 0) {
        SNEPPX_free(s, sizeof(SNEPPXStorage));
        return NULL;
    }
    s->num_bytes = num_bytes;
    s->ref_count = 1;
    return s;
}

void SNEPPX_storage_retain(SNEPPXStorage* s) {
    if (s) s->ref_count++;
}

void SNEPPX_storage_release(SNEPPXStorage* s) {
    if (!s) return;
    if (--s->ref_count <= 0) {
        if (s->data) SNEPPX_free(s->data, s->num_bytes);
        SNEPPX_free(s, sizeof(SNEPPXStorage));
    }
}

/* ===== View creation helpers ===== */

static SNEPPXTensor* tensor_view_alloc(const SNEPPXTensor* src) {
    SNEPPXTensor* t = (SNEPPXTensor*)aligned_alloc_wrapper(sizeof(SNEPPXTensor), 64);
    if (!t) return NULL;
    size_t ndim = src->ndim > 0 ? src->ndim : 1;
    t->shape = (size_t*)aligned_alloc_wrapper(ndim * sizeof(size_t), 64);
    t->strides = (size_t*)aligned_alloc_wrapper(ndim * sizeof(size_t), 64);
    if (!t->shape || !t->strides) {
        SNEPPX_free(t->shape, ndim * sizeof(size_t));
        SNEPPX_free(t->strides, ndim * sizeof(size_t));
        SNEPPX_free(t, sizeof(SNEPPXTensor));
        return NULL;
    }
    return t;
}

SNEPPXTensor* SNEPPX_tensor_as_strided(const SNEPPXTensor* src, size_t offset, const size_t* shape, size_t ndim, const size_t* strides) {
    if (!src) return NULL;
    SNEPPXTensor* t = tensor_view_alloc(src);
    if (!t) return NULL;
    SNEPPX_storage_retain(src->storage);
    t->storage = src->storage;
    t->offset = offset;
    t->ndim = ndim;
    t->size = 1;
    for (size_t i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->strides[i] = strides[i];
        t->size *= shape[i];
    }
    t->dtype = src->dtype;
    t->item_size = src->item_size;
    t->device = src->device;
    t->device_id = src->device_id;
    t->layout = src->layout;
    t->owns_data = 0;
    t->backend_handle = NULL;
    t->data = (unsigned char*)src->storage->data + offset * t->item_size;
    return t;
}

SNEPPXTensor* SNEPPX_tensor_narrow(const SNEPPXTensor* src, size_t dim, size_t start, size_t size) {
    if (!src || dim >= src->ndim || start + size > src->shape[dim]) return NULL;
    size_t new_shape[16], new_strides[16];
    for (size_t i = 0; i < src->ndim; i++) {
        new_shape[i] = src->shape[i];
        new_strides[i] = src->strides[i];
    }
    new_shape[dim] = size;
    size_t offset = src->offset + start * src->strides[dim];
    return SNEPPX_tensor_as_strided(src, offset, new_shape, src->ndim, new_strides);
}

/* ===== Backward-compat: ensure storage exists for legacy code paths ===== */
static SNEPPXStorage* ensure_storage(SNEPPXTensor* t) {
    if (!t->storage) {
        size_t nb = t->size * t->item_size;
        t->storage = SNEPPX_storage_create(nb);
        if (t->storage && t->data) {
            memcpy(t->storage->data, t->data, nb);
            /* keep data pointing to storage */
            t->data = t->storage->data;
            t->offset = 0;
        }
    }
    return t->storage;
}

static void SNEPPX_tensor_fill_scalar(SNEPPXTensor* t, double value) {
    unsigned char* data = (unsigned char*)t->data;
    size_t n = t->size;
    switch (t->dtype) {
        case SNEPPX_FLOAT32: { float v = (float)value; for (size_t i = 0; i < n; i++) ((float*)data)[i] = v; break; }
        case SNEPPX_FLOAT64: { double v = value; for (size_t i = 0; i < n; i++) ((double*)data)[i] = v; break; }
        case SNEPPX_FLOAT16:
        case SNEPPX_BFLOAT16:
        case SNEPPX_FLOAT8:  { float v = (float)value; for (size_t i = 0; i < n; i++) ((float*)data)[i] = v; break; }
        case SNEPPX_INT32:   { int32_t v = (int32_t)value; for (size_t i = 0; i < n; i++) ((int32_t*)data)[i] = v; break; }
        case SNEPPX_INT64:   { int64_t v = (int64_t)value; for (size_t i = 0; i < n; i++) ((int64_t*)data)[i] = v; break; }
        case SNEPPX_INT16:   { int16_t v = (int16_t)value; for (size_t i = 0; i < n; i++) ((int16_t*)data)[i] = v; break; }
        case SNEPPX_INT8:    { int8_t v = (int8_t)value; for (size_t i = 0; i < n; i++) ((int8_t*)data)[i] = v; break; }
        case SNEPPX_UINT8:   { uint8_t v = (uint8_t)value; for (size_t i = 0; i < n; i++) ((uint8_t*)data)[i] = v; break; }
        case SNEPPX_BOOL:    { uint8_t v = value != 0.0 ? 1 : 0; for (size_t i = 0; i < n; i++) ((uint8_t*)data)[i] = v; break; }
        case SNEPPX_COMPLEX64:
        case SNEPPX_COMPLEX128: { memset(data, 0, n * t->item_size); break; }
    }
}

SNEPPXTensor* SNEPPX_tensor_create(const size_t* shape, size_t ndim, SNEPPXDtype dtype) {
    if (ndim > 0 && !shape) return NULL;
    SNEPPXTensor* tensor = (SNEPPXTensor*)aligned_alloc_wrapper(sizeof(SNEPPXTensor), 64);
    if (!tensor) return NULL;

    tensor->ndim = ndim;
    tensor->dtype = dtype;
    tensor->item_size = SNEPPX_tensor_dtype_size(dtype);
    tensor->device = SNEPPX_DEVICE_CPU;
    tensor->device_id = 0;
    tensor->layout = SNEPPX_LAYOUT_ROW_MAJOR;
    tensor->owns_data = 1;
    tensor->backend_handle = NULL;
    tensor->storage = NULL;
    tensor->offset = 0;

    size_t safe_ndim = ndim > 0 ? ndim : 1;
    tensor->shape = (size_t*)aligned_alloc_wrapper(safe_ndim * sizeof(size_t), 64);
    tensor->strides = (size_t*)aligned_alloc_wrapper(safe_ndim * sizeof(size_t), 64);
    if (!tensor->shape || !tensor->strides) {
        SNEPPX_free(tensor->shape, safe_ndim * sizeof(size_t));
        SNEPPX_free(tensor->strides, safe_ndim * sizeof(size_t));
        SNEPPX_free(tensor, sizeof(SNEPPXTensor));
        return NULL;
    }

    size_t total = 1;
    for (size_t i = 0; i < ndim; i++) {
        tensor->shape[i] = shape[i];
        total *= shape[i];
    }
    if (ndim == 0) total = 1;
    tensor->size = total;

    size_t stride = 1;
    for (size_t i = ndim; i > 0; i--) {
        tensor->strides[i - 1] = stride;
        stride *= shape[i - 1];
    }

    size_t num_bytes = total * tensor->item_size;
    tensor->storage = SNEPPX_storage_create(num_bytes);
    if (!tensor->storage) {
        SNEPPX_free(tensor->shape, ndim * sizeof(size_t));
        SNEPPX_free(tensor->strides, ndim * sizeof(size_t));
        SNEPPX_free(tensor, sizeof(SNEPPXTensor));
        return NULL;
    }
    tensor->data = tensor->storage->data;

    return tensor;
}

void SNEPPX_tensor_destroy(SNEPPXTensor* tensor) {
    if (!tensor) return;
    SNEPPX_storage_release(tensor->storage);
    size_t ndim = tensor->ndim > 0 ? tensor->ndim : 1;
    SNEPPX_free(tensor->shape, ndim * sizeof(size_t));
    SNEPPX_free(tensor->strides, ndim * sizeof(size_t));
    SNEPPX_free(tensor, sizeof(SNEPPXTensor));
}

float SNEPPX_tensor_get_f32(const SNEPPXTensor* tensor, const size_t* indices) {
    if (tensor->dtype != SNEPPX_FLOAT32) return 0.0f;
    size_t offset = compute_offset(tensor, indices);
    return ((float*)tensor->data)[offset];
}

void SNEPPX_tensor_set_f32(SNEPPXTensor* tensor, const size_t* indices, float value) {
    if (tensor->dtype != SNEPPX_FLOAT32) return;
    size_t offset = compute_offset(tensor, indices);
    ((float*)tensor->data)[offset] = value;
}

double SNEPPX_tensor_get_f64(const SNEPPXTensor* tensor, const size_t* indices) {
    if (tensor->dtype != SNEPPX_FLOAT64) return 0.0;
    size_t offset = compute_offset(tensor, indices);
    return ((double*)tensor->data)[offset];
}

void SNEPPX_tensor_set_f64(SNEPPXTensor* tensor, const size_t* indices, double value) {
    if (tensor->dtype != SNEPPX_FLOAT64) return;
    size_t offset = compute_offset(tensor, indices);
    ((double*)tensor->data)[offset] = value;
}

int32_t SNEPPX_tensor_get_i32(const SNEPPXTensor* tensor, const size_t* indices) {
    if (tensor->dtype != SNEPPX_INT32) return 0;
    size_t offset = compute_offset(tensor, indices);
    return ((int32_t*)tensor->data)[offset];
}

void SNEPPX_tensor_set_i32(SNEPPXTensor* tensor, const size_t* indices, int32_t value) {
    if (tensor->dtype != SNEPPX_INT32) return;
    size_t offset = compute_offset(tensor, indices);
    ((int32_t*)tensor->data)[offset] = value;
}

int64_t SNEPPX_tensor_get_i64(const SNEPPXTensor* tensor, const size_t* indices) {
    if (tensor->dtype != SNEPPX_INT64) return 0;
    size_t offset = compute_offset(tensor, indices);
    return ((int64_t*)tensor->data)[offset];
}

void SNEPPX_tensor_set_i64(SNEPPXTensor* tensor, const size_t* indices, int64_t value) {
    if (tensor->dtype != SNEPPX_INT64) return;
    size_t offset = compute_offset(tensor, indices);
    ((int64_t*)tensor->data)[offset] = value;
}

uint8_t SNEPPX_tensor_get_bool(const SNEPPXTensor* tensor, const size_t* indices) {
    if (tensor->dtype != SNEPPX_BOOL) return 0;
    size_t offset = compute_offset(tensor, indices);
    return ((uint8_t*)tensor->data)[offset];
}

void SNEPPX_tensor_set_bool(SNEPPXTensor* tensor, const size_t* indices, uint8_t value) {
    if (tensor->dtype != SNEPPX_BOOL) return;
    size_t offset = compute_offset(tensor, indices);
    ((uint8_t*)tensor->data)[offset] = value ? 1 : 0;
}

void SNEPPX_tensor_fill_f32(SNEPPXTensor* tensor, float value) {
    if (!tensor || tensor->dtype != SNEPPX_FLOAT32) return;
    float* data = (float*)tensor->data;
    for (size_t i = 0; i < tensor->size; i++) data[i] = value;
}

void SNEPPX_tensor_fill_f64(SNEPPXTensor* tensor, double value) {
    if (!tensor || tensor->dtype != SNEPPX_FLOAT64) return;
    double* data = (double*)tensor->data;
    for (size_t i = 0; i < tensor->size; i++) data[i] = value;
}

SNEPPXTensor* SNEPPX_tensor_empty(const size_t* shape, size_t ndim, SNEPPXDtype dtype) {
    return SNEPPX_tensor_create(shape, ndim, dtype);
}

SNEPPXTensor* SNEPPX_tensor_zeros(const size_t* shape, size_t ndim, SNEPPXDtype dtype) {
    SNEPPXTensor* tensor = SNEPPX_tensor_create(shape, ndim, dtype);
    if (!tensor) return NULL;
    memset(tensor->data, 0, tensor->size * tensor->item_size);
    return tensor;
}

SNEPPXTensor* SNEPPX_tensor_ones(const size_t* shape, size_t ndim, SNEPPXDtype dtype) {
    SNEPPXTensor* tensor = SNEPPX_tensor_create(shape, ndim, dtype);
    if (!tensor) return NULL;
    size_t n = tensor->size;
    if (dtype == SNEPPX_FLOAT32 || dtype == SNEPPX_FLOAT16 || dtype == SNEPPX_BFLOAT16 || dtype == SNEPPX_FLOAT8) {
        float* data = (float*)tensor->data;
        for (size_t i = 0; i < n; i++) data[i] = 1.0f;
    } else if (dtype == SNEPPX_FLOAT64) {
        double* data = (double*)tensor->data;
        for (size_t i = 0; i < n; i++) data[i] = 1.0;
    } else if (dtype == SNEPPX_INT32 || dtype == SNEPPX_INT64 || dtype == SNEPPX_INT16 || dtype == SNEPPX_INT8 || dtype == SNEPPX_UINT8 || dtype == SNEPPX_BOOL) {
        memset(tensor->data, 1, n * tensor->item_size);
    }
    return tensor;
}

SNEPPXTensor* SNEPPX_tensor_full(const size_t* shape, size_t ndim, SNEPPXDtype dtype, const void* value) {
    SNEPPXTensor* tensor = SNEPPX_tensor_create(shape, ndim, dtype);
    if (!tensor || !value) return tensor;
    size_t item_size = tensor->item_size;
    unsigned char* data = (unsigned char*)tensor->data;
    for (size_t i = 0; i < tensor->size; i++) {
        memcpy(data + i * item_size, value, item_size);
    }
    return tensor;
}

SNEPPXTensor* SNEPPX_tensor_arange(float start, float stop, float step, SNEPPXDtype dtype) {
    if (step == 0.0f) return NULL;
    if ((step > 0.0f && start >= stop) || (step < 0.0f && start <= stop)) return NULL;
    size_t n = (size_t)((stop - start) / step);
    size_t shape[] = {n};
    SNEPPXTensor* tensor = SNEPPX_tensor_create(shape, 1, dtype);
    if (!tensor) return NULL;
    if (dtype == SNEPPX_FLOAT32) {
        float* data = (float*)tensor->data;
        for (size_t i = 0; i < n; i++) data[i] = start + i * step;
    } else if (dtype == SNEPPX_FLOAT64) {
        double* data = (double*)tensor->data;
        for (size_t i = 0; i < n; i++) data[i] = (double)(start + i * step);
    } else if (dtype == SNEPPX_INT32) {
        int32_t* data = (int32_t*)tensor->data;
        for (size_t i = 0; i < n; i++) data[i] = (int32_t)(start + i * step);
    } else if (dtype == SNEPPX_INT64) {
        int64_t* data = (int64_t*)tensor->data;
        for (size_t i = 0; i < n; i++) data[i] = (int64_t)(start + i * step);
    }
    return tensor;
}

SNEPPXTensor* SNEPPX_tensor_linspace(float start, float stop, size_t steps, SNEPPXDtype dtype) {
    if (steps == 0) return NULL;
    size_t shape[] = {steps};
    SNEPPXTensor* tensor = SNEPPX_tensor_create(shape, 1, dtype);
    if (!tensor) return NULL;
    float step = (steps > 1) ? (stop - start) / (float)(steps - 1) : 0.0f;
    if (dtype == SNEPPX_FLOAT32) {
        float* data = (float*)tensor->data;
        for (size_t i = 0; i < steps; i++) data[i] = start + i * step;
    } else if (dtype == SNEPPX_FLOAT64) {
        double* data = (double*)tensor->data;
        for (size_t i = 0; i < steps; i++) data[i] = (double)(start + i * step);
    }
    return tensor;
}

SNEPPXTensor* SNEPPX_tensor_eye(size_t n, SNEPPXDtype dtype) {
    size_t shape[] = {n, n};
    SNEPPXTensor* tensor = SNEPPX_tensor_zeros(shape, 2, dtype);
    if (!tensor) return NULL;
    unsigned char* data = (unsigned char*)tensor->data;
    size_t is = tensor->item_size;
    for (size_t i = 0; i < n; i++) {
        if (dtype == SNEPPX_FLOAT64) {
            *(double*)(data + (i * n + i) * is) = 1.0;
        } else if (dtype == SNEPPX_INT32 || dtype == SNEPPX_INT64) {
            memset(data + (i * n + i) * is, 1, is);
        } else {
            *(float*)(data + (i * n + i) * is) = 1.0f;
        }
    }
    return tensor;
}

static unsigned long lcg_state = 123456789;

static float uniform_01(void) {
    lcg_state = lcg_state * 1103515245UL + 12345UL;
    return (float)((lcg_state >> 16) & 0x7FFF) / 32767.0f;
}

SNEPPXTensor* SNEPPX_tensor_randn(const size_t* shape, size_t ndim, SNEPPXDtype dtype) {
    SNEPPXTensor* tensor = SNEPPX_tensor_create(shape, ndim, dtype);
    if (!tensor) return NULL;
    float* data = (float*)tensor->data;
    for (size_t i = 0; i < tensor->size; i += 2) {
        float u1 = uniform_01();
        float u2 = uniform_01();
        float r = sqrtf(-2.0f * logf(u1 + 1e-10f));
        float theta = 2.0f * 3.14159265f * u2;
        data[i] = r * cosf(theta);
        if (i + 1 < tensor->size) {
            data[i + 1] = r * sinf(theta);
        }
    }
    return tensor;
}

SNEPPXTensor* SNEPPX_tensor_copy(const SNEPPXTensor* src) {
    if (!src) return NULL;
    SNEPPXTensor* dst = SNEPPX_tensor_create(src->shape, src->ndim, src->dtype);
    if (!dst) return NULL;
    memcpy(dst->data, src->data, src->size * src->item_size);
    return dst;
}

SNEPPXTensor* SNEPPX_tensor_clone(const SNEPPXTensor* src) {
    return SNEPPX_tensor_copy(src);
}

SNEPPXTensor* SNEPPX_tensor_slice(const SNEPPXTensor* src, size_t dim, size_t start, size_t end) {
    if (!src || dim >= src->ndim || start >= end || end > src->shape[dim]) return NULL;
    return SNEPPX_tensor_narrow(src, dim, start, end - start);
}

SNEPPXTensor* SNEPPX_tensor_reshape(const SNEPPXTensor* src, const size_t* new_shape, size_t new_ndim) {
    if (!src) return NULL;
    size_t new_size = 1;
    size_t auto_idx = new_ndim;
    for (size_t i = 0; i < new_ndim; i++) {
        if (new_shape[i] == (size_t)-1) {
            auto_idx = i;
        } else {
            new_size *= new_shape[i];
        }
    }
    size_t resolved_shape[16];
    for (size_t i = 0; i < new_ndim; i++) {
        resolved_shape[i] = (i == auto_idx) ? (src->size / new_size) : new_shape[i];
    }
    if (SNEPPX_tensor_is_contiguous(src) && src->ndim > 0) {
        size_t new_strides[16];
        size_t stride = 1;
        for (size_t i = new_ndim; i > 0; i--) {
            new_strides[i - 1] = stride;
            stride *= resolved_shape[i - 1];
        }
        return SNEPPX_tensor_as_strided(src, src->offset, resolved_shape, new_ndim, new_strides);
    }
    SNEPPXTensor* result = SNEPPX_tensor_empty(resolved_shape, new_ndim, src->dtype);
    if (!result) return NULL;
    memcpy(result->data, src->data, src->size * src->item_size);
    return result;
}

SNEPPXTensor* SNEPPX_tensor_permute(const SNEPPXTensor* src, const size_t* axes) {
    if (!src) return NULL;
    size_t new_shape[16], new_strides[16];
    for (size_t i = 0; i < src->ndim; i++) {
        new_shape[i] = src->shape[axes[i]];
        new_strides[i] = src->strides[axes[i]];
    }
    return SNEPPX_tensor_as_strided(src, src->offset, new_shape, src->ndim, new_strides);
}

SNEPPXTensor* SNEPPX_tensor_expand(const SNEPPXTensor* src, const size_t* new_shape, size_t new_ndim) {
    if (!src || new_ndim < src->ndim) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_empty(new_shape, new_ndim, src->dtype);
    if (!result) return NULL;
    size_t* src_indices = (size_t*)aligned_alloc_wrapper(src->ndim * sizeof(size_t), 64);
    size_t* dst_indices = (size_t*)aligned_alloc_wrapper(new_ndim * sizeof(size_t), 64);
    if (!src_indices || !dst_indices) { SNEPPX_free(src_indices, src->ndim * sizeof(size_t)); SNEPPX_free(dst_indices, new_ndim * sizeof(size_t)); SNEPPX_tensor_destroy(result); return NULL; }
    memset(src_indices, 0, src->ndim * sizeof(size_t));
    memset(dst_indices, 0, new_ndim * sizeof(size_t));
    unsigned char* src_data = (unsigned char*)src->data;
    unsigned char* dst_data = (unsigned char*)result->data;
    size_t is = src->item_size;
    size_t offset_src = new_ndim - src->ndim;
    for (size_t flat = 0; flat < result->size; flat++) {
        size_t tmp = flat;
        for (size_t i = new_ndim; i > 0; i--) {
            dst_indices[i - 1] = (result->strides[i - 1] > 0) ? (tmp / result->strides[i - 1]) % result->shape[i - 1] : 0;
            if (result->strides[i - 1] > 0) tmp %= result->strides[i - 1];
        }
        for (size_t i = 0; i < src->ndim; i++) {
            size_t sdim = dst_indices[i + offset_src];
            src_indices[i] = (sdim >= src->shape[i]) ? 0 : sdim;
        }
        size_t soff = compute_offset(src, src_indices);
        memcpy(dst_data + flat * is, src_data + soff * is, is);
    }
    SNEPPX_free(src_indices, src->ndim * sizeof(size_t));
    SNEPPX_free(dst_indices, new_ndim * sizeof(size_t));
    return result;
}

SNEPPXTensor* SNEPPX_tensor_squeeze(const SNEPPXTensor* src, size_t dim) {
    if (!src || dim >= src->ndim || src->shape[dim] != 1) return SNEPPX_tensor_copy(src);
    size_t new_ndim = src->ndim - 1;
    size_t new_shape[16], new_strides[16];
    size_t j = 0;
    for (size_t i = 0; i < src->ndim; i++) {
        if (i != dim) {
            new_shape[j] = src->shape[i];
            new_strides[j] = src->strides[i];
            j++;
        }
    }
    return SNEPPX_tensor_as_strided(src, src->offset, new_shape, new_ndim, new_strides);
}

SNEPPXTensor* SNEPPX_tensor_unsqueeze(const SNEPPXTensor* src, size_t dim) {
    if (!src || dim > src->ndim) return NULL;
    size_t new_ndim = src->ndim + 1;
    size_t new_shape[16], new_strides[16];
    size_t j = 0;
    for (size_t i = 0; i < new_ndim; i++) {
        if (i == dim) {
            new_shape[i] = 1;
            new_strides[i] = 0;
        } else {
            new_shape[i] = src->shape[j];
            new_strides[i] = src->strides[j];
            j++;
        }
    }
    return SNEPPX_tensor_as_strided(src, src->offset, new_shape, new_ndim, new_strides);
}

SNEPPXTensor* SNEPPX_tensor_concat(const SNEPPXTensor** tensors, size_t num_tensors, size_t dim) {
    if (!tensors || num_tensors == 0) return NULL;
    const SNEPPXTensor* first = tensors[0];
    if (!first || dim >= first->ndim) return NULL;
    size_t total_dim = 0;
    for (size_t t = 0; t < num_tensors; t++) {
        if (!tensors[t] || tensors[t]->ndim != first->ndim) return NULL;
        for (size_t i = 0; i < first->ndim; i++) {
            if (i != dim && tensors[t]->shape[i] != first->shape[i]) return NULL;
        }
        total_dim += tensors[t]->shape[dim];
    }
    size_t* new_shape = (size_t*)aligned_alloc_wrapper(first->ndim * sizeof(size_t), 64);
    if (!new_shape) return NULL;
    for (size_t i = 0; i < first->ndim; i++) new_shape[i] = first->shape[i];
    new_shape[dim] = total_dim;
    SNEPPXTensor* result = SNEPPX_tensor_empty(new_shape, first->ndim, first->dtype);
    SNEPPX_free(new_shape, first->ndim * sizeof(size_t));
    if (!result) return NULL;
    unsigned char* dst = (unsigned char*)result->data;
    size_t item_size = first->item_size;
    size_t offset = 0;
    size_t slice_size = first->size / first->shape[dim];
    for (size_t t = 0; t < num_tensors; t++) {
        size_t n = tensors[t]->shape[dim] * slice_size;
        memcpy(dst + offset * item_size, tensors[t]->data, n * item_size);
        offset += n;
    }
    return result;
}

SNEPPXTensor** SNEPPX_tensor_split(const SNEPPXTensor* src, size_t num_splits, size_t dim) {
    if (!src || num_splits == 0 || dim >= src->ndim || src->shape[dim] % num_splits != 0) return NULL;
    size_t split_size = src->shape[dim] / num_splits;
    SNEPPXTensor** results = (SNEPPXTensor**)aligned_alloc_wrapper(num_splits * sizeof(SNEPPXTensor*), 64);
    if (!results) return NULL;
    memset(results, 0, num_splits * sizeof(SNEPPXTensor*));
    for (size_t s = 0; s < num_splits; s++) {
        results[s] = SNEPPX_tensor_slice(src, dim, s * split_size, (s + 1) * split_size);
        if (!results[s]) {
            for (size_t k = 0; k < s; k++) SNEPPX_tensor_destroy(results[k]);
            SNEPPX_free(results, num_splits * sizeof(SNEPPXTensor*));
            return NULL;
        }
    }
    return results;
}

SNEPPXTensor* SNEPPX_tensor_tile(const SNEPPXTensor* src, const size_t* reps, size_t reps_ndim) {
    if (!src || !reps) return NULL;
    size_t out_ndim = (src->ndim > reps_ndim) ? src->ndim : reps_ndim;
    size_t out_shape[16];
    size_t offset = out_ndim - src->ndim;
    for (size_t i = 0; i < out_ndim; i++) {
        size_t s = (i < offset) ? 1 : src->shape[i - offset];
        size_t r = (i < reps_ndim) ? reps[i] : 1;
        out_shape[i] = s * r;
    }
    SNEPPXTensor* result = SNEPPX_tensor_empty(out_shape, out_ndim, src->dtype);
    if (!result) return NULL;
    unsigned char* src_data = (unsigned char*)src->data;
    unsigned char* dst_data = (unsigned char*)result->data;
    size_t is = src->item_size;
    size_t* indices = (size_t*)aligned_alloc_wrapper(out_ndim * sizeof(size_t), 64);
    if (!indices) { SNEPPX_tensor_destroy(result); return NULL; }
    for (size_t flat = 0; flat < result->size; flat++) {
        size_t tmp = flat;
        for (size_t i = out_ndim; i > 0; i--) {
            indices[i - 1] = (result->strides[i - 1] > 0) ? (tmp / result->strides[i - 1]) % result->shape[i - 1] : 0;
            if (result->strides[i - 1] > 0) tmp %= result->strides[i - 1];
        }
        size_t src_flat = 0;
        for (size_t i = 0; i < src->ndim; i++) {
            size_t si = indices[offset + i] % src->shape[i];
            src_flat += si * src->strides[i];
        }
        memcpy(dst_data + flat * is, src_data + src_flat * is, is);
    }
    SNEPPX_free(indices, out_ndim * sizeof(size_t));
    return result;
}

SNEPPXTensor* SNEPPX_tensor_repeat(const SNEPPXTensor* src, size_t repeats, size_t dim) {
    if (!src || dim >= src->ndim) return NULL;
    size_t* reps = (size_t*)aligned_alloc_wrapper(src->ndim * sizeof(size_t), 64);
    if (!reps) return NULL;
    for (size_t i = 0; i < src->ndim; i++) reps[i] = (i == dim) ? repeats : 1;
    SNEPPXTensor* result = SNEPPX_tensor_tile(src, reps, src->ndim);
    SNEPPX_free(reps, src->ndim * sizeof(size_t));
    return result;
}

SNEPPXTensor* SNEPPX_tensor_gather(const SNEPPXTensor* src, size_t dim, const SNEPPXTensor* indices) {
    if (!src || !indices || dim >= src->ndim || indices->ndim != src->ndim) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_empty(indices->shape, indices->ndim, src->dtype);
    if (!result) return NULL;
    unsigned char* src_data = (unsigned char*)src->data;
    unsigned char* dst_data = (unsigned char*)result->data;
    size_t is = src->item_size;
    size_t* idx = (size_t*)aligned_alloc_wrapper(src->ndim * sizeof(size_t), 64);
    if (!idx) { SNEPPX_tensor_destroy(result); return NULL; }
    int32_t* idx_data = (int32_t*)indices->data;
    for (size_t flat = 0; flat < result->size; flat++) {
        size_t tmp = flat;
        for (size_t i = result->ndim; i > 0; i--) {
            idx[i - 1] = (result->strides[i - 1] > 0) ? (tmp / result->strides[i - 1]) % result->shape[i - 1] : 0;
            if (result->strides[i - 1] > 0) tmp %= result->strides[i - 1];
        }
        size_t gather_idx = idx_data[flat];
        if (gather_idx >= src->shape[dim]) gather_idx = src->shape[dim] - 1;
        idx[dim] = gather_idx;
        size_t soff = compute_offset(src, idx);
        memcpy(dst_data + flat * is, src_data + soff * is, is);
    }
    SNEPPX_free(idx, src->ndim * sizeof(size_t));
    return result;
}

SNEPPXTensor* SNEPPX_tensor_scatter(SNEPPXTensor* dest, size_t dim, const SNEPPXTensor* indices, const SNEPPXTensor* src) {
    if (!dest || !indices || !src || dim >= dest->ndim) return NULL;
    unsigned char* src_data = (unsigned char*)src->data;
    size_t is = dest->item_size;
    size_t* idx = (size_t*)aligned_alloc_wrapper(dest->ndim * sizeof(size_t), 64);
    if (!idx) return NULL;
    int32_t* idx_data = (int32_t*)indices->data;
    for (size_t flat = 0; flat < src->size; flat++) {
        size_t tmp = flat;
        for (size_t i = src->ndim; i > 0; i--) {
            idx[i - 1] = (src->strides[i - 1] > 0) ? (tmp / src->strides[i - 1]) % src->shape[i - 1] : 0;
            if (src->strides[i - 1] > 0) tmp %= src->strides[i - 1];
        }
        size_t scatter_idx = idx_data[flat];
        if (scatter_idx >= dest->shape[dim]) scatter_idx = dest->shape[dim] - 1;
        idx[dim] = scatter_idx;
        size_t doff = compute_offset(dest, idx);
        memcpy(((unsigned char*)dest->data) + doff * is, src_data + flat * is, is);
    }
    SNEPPX_free(idx, dest->ndim * sizeof(size_t));
    return dest;
}

SNEPPXTensor* SNEPPX_tensor_masked_select(const SNEPPXTensor* src, const SNEPPXTensor* mask) {
    if (!src || !mask || src->size != mask->size) return NULL;
    unsigned char* mask_data = (unsigned char*)mask->data;
    size_t count = 0;
    for (size_t i = 0; i < mask->size; i++) {
        if (mask_data[i]) count++;
    }
    size_t shape[] = {count};
    SNEPPXTensor* result = SNEPPX_tensor_empty(shape, 1, src->dtype);
    if (!result) return NULL;
    unsigned char* src_data = (unsigned char*)src->data;
    unsigned char* dst_data = (unsigned char*)result->data;
    size_t is = src->item_size;
    size_t j = 0;
    for (size_t i = 0; i < src->size; i++) {
        if (mask_data[i]) {
            memcpy(dst_data + j * is, src_data + i * is, is);
            j++;
        }
    }
    return result;
}

SNEPPXTensor* SNEPPX_tensor_masked_fill(SNEPPXTensor* src, const SNEPPXTensor* mask, const void* value) {
    if (!src || !mask || !value || src->size != mask->size) return src;
    unsigned char* mask_data = (unsigned char*)mask->data;
    unsigned char* src_data = (unsigned char*)src->data;
    size_t is = src->item_size;
    for (size_t i = 0; i < src->size; i++) {
        if (mask_data[i]) {
            memcpy(src_data + i * is, value, is);
        }
    }
    return src;
}

SNEPPXTensor* SNEPPX_tensor_where(const SNEPPXTensor* condition, const SNEPPXTensor* x, const SNEPPXTensor* y) {
    if (!condition || !x || !y) return NULL;
    size_t n = x->size < y->size ? x->size : y->size;
    if (condition->size < n) n = condition->size;
    SNEPPXTensor* result = SNEPPX_tensor_copy(x);
    if (!result) return NULL;
    unsigned char* cond_data = (unsigned char*)condition->data;
    unsigned char* y_data = (unsigned char*)y->data;
    unsigned char* dst_data = (unsigned char*)result->data;
    size_t is = x->item_size;
    for (size_t i = 0; i < n; i++) {
        if (!cond_data[i]) {
            memcpy(dst_data + i * is, y_data + i * is, is);
        }
    }
    return result;
}

SNEPPXTensor* SNEPPX_tensor_cast(const SNEPPXTensor* src, SNEPPXDtype dtype) {
    if (!src || dtype == src->dtype) return SNEPPX_tensor_copy(src);
    SNEPPXTensor* result = SNEPPX_tensor_empty(src->shape, src->ndim, dtype);
    if (!result) return NULL;
    size_t n = src->size;
    if (src->dtype == SNEPPX_FLOAT32 && dtype == SNEPPX_FLOAT64) {
        float* s = (float*)src->data; double* d = (double*)result->data;
        for (size_t i = 0; i < n; i++) d[i] = (double)s[i];
    } else if (src->dtype == SNEPPX_FLOAT64 && dtype == SNEPPX_FLOAT32) {
        double* s = (double*)src->data; float* d = (float*)result->data;
        for (size_t i = 0; i < n; i++) d[i] = (float)s[i];
    } else if (src->dtype == SNEPPX_FLOAT32 && dtype == SNEPPX_INT32) {
        float* s = (float*)src->data; int32_t* d = (int32_t*)result->data;
        for (size_t i = 0; i < n; i++) d[i] = (int32_t)s[i];
    } else if (src->dtype == SNEPPX_INT32 && dtype == SNEPPX_FLOAT32) {
        int32_t* s = (int32_t*)src->data; float* d = (float*)result->data;
        for (size_t i = 0; i < n; i++) d[i] = (float)s[i];
    } else if (dtype == SNEPPX_FLOAT32) {
        float* d = (float*)result->data;
        for (size_t i = 0; i < n; i++) {
            d[i] = SNEPPX_tensor_get_f32(src, (size_t[]){i});
        }
    } else {
        memcpy(result->data, src->data, n * result->item_size);
    }
    return result;
}

SNEPPXTensor* SNEPPX_tensor_to_device(const SNEPPXTensor* src, SNEPPXDevice device) {
    (void)device;
    if (!src) return NULL;
    SNEPPXTensor* dst = SNEPPX_tensor_copy(src);
    if (!dst) return NULL;
    dst->device = device;
    return dst;
}

SNEPPXTensor* SNEPPX_tensor_to_layout(const SNEPPXTensor* src, SNEPPXLayout layout) {
    if (!src) return NULL;
    if (src->layout == layout) return SNEPPX_tensor_copy(src);
    SNEPPXTensor* dst = SNEPPX_tensor_copy(src);
    if (!dst) return NULL;
    if (layout == SNEPPX_LAYOUT_COL_MAJOR && src->ndim == 2) {
        size_t m = src->shape[0], n = src->shape[1];
        float* s = (float*)src->data;
        float* d = (float*)dst->data;
        for (size_t i = 0; i < m; i++)
            for (size_t j = 0; j < n; j++)
                d[j * m + i] = s[i * n + j];
        dst->strides[0] = 1;
        dst->strides[1] = m;
    }
    dst->layout = layout;
    return dst;
}

int SNEPPX_tensor_save(const SNEPPXTensor* src, const char* path) {
    if (!src || !path) return -1;
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t magic = 0x41524958;
    uint32_t version = 1;
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&version, sizeof(version), 1, f);
    uint64_t ndim = (uint64_t)src->ndim;
    fwrite(&ndim, sizeof(ndim), 1, f);
    for (size_t i = 0; i < src->ndim; i++) {
        uint64_t dim = (uint64_t)src->shape[i];
        fwrite(&dim, sizeof(dim), 1, f);
    }
    uint32_t dtype = (uint32_t)src->dtype;
    fwrite(&dtype, sizeof(dtype), 1, f);
    fwrite(src->data, src->size * src->item_size, 1, f);
    fclose(f);
    return 0;
}

SNEPPXTensor* SNEPPX_tensor_load(const char* path) {
    if (!path) return NULL;
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    uint32_t magic, version;
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != 0x41524958) { fclose(f); return NULL; }
    if (fread(&version, sizeof(version), 1, f) != 1 || version != 1) { fclose(f); return NULL; }
    uint64_t ndim;
    if (fread(&ndim, sizeof(ndim), 1, f) != 1) { fclose(f); return NULL; }
    size_t shape[16];
    for (uint64_t i = 0; i < ndim; i++) {
        uint64_t dim;
        if (fread(&dim, sizeof(dim), 1, f) != 1) { fclose(f); return NULL; }
        shape[i] = (size_t)dim;
    }
    uint32_t dtype;
    if (fread(&dtype, sizeof(dtype), 1, f) != 1) { fclose(f); return NULL; }
    SNEPPXTensor* tensor = SNEPPX_tensor_create(shape, (size_t)ndim, (SNEPPXDtype)dtype);
    if (!tensor) { fclose(f); return NULL; }
    size_t bytes = tensor->size * tensor->item_size;
    if (fread(tensor->data, bytes, 1, f) != 1) { SNEPPX_tensor_destroy(tensor); fclose(f); return NULL; }
    fclose(f);
    return tensor;
}

/* Forward declaration of contiguity helper */
static const SNEPPXTensor* tensor_prep_contiguous(const SNEPPXTensor* src, SNEPPXTensor** out);

static SNEPPXTensor* compare_op(const SNEPPXTensor* a, const SNEPPXTensor* b, int (*cmp)(float, float)) {
    if (!a || !b) return NULL;
    SNEPPXTensor* ta = NULL, *tb = NULL;
    a = tensor_prep_contiguous(a, &ta);
    b = tensor_prep_contiguous(b, &tb);
    size_t n = a->size < b->size ? a->size : b->size;
    SNEPPXTensor* result = SNEPPX_tensor_empty(a->shape, a->ndim, SNEPPX_BOOL);
    if (!result) { if (ta) SNEPPX_tensor_destroy(ta); if (tb) SNEPPX_tensor_destroy(tb); return NULL; }
    float* ad = (float*)a->data;
    float* bd = (float*)b->data;
    uint8_t* rd = (uint8_t*)result->data;
    for (size_t i = 0; i < n; i++) rd[i] = cmp(ad[i], bd[i]) ? 1 : 0;
    if (ta) SNEPPX_tensor_destroy(ta); if (tb) SNEPPX_tensor_destroy(tb);
    return result;
}

static int cmp_eq(float a, float b) { return fabsf(a - b) < 1e-6f; }
static int cmp_ne(float a, float b) { return fabsf(a - b) >= 1e-6f; }
static int cmp_lt(float a, float b) { return a < b; }
static int cmp_le(float a, float b) { return a <= b; }
static int cmp_gt(float a, float b) { return a > b; }
static int cmp_ge(float a, float b) { return a >= b; }

SNEPPXTensor* SNEPPX_tensor_eq(const SNEPPXTensor* a, const SNEPPXTensor* b) { return compare_op(a, b, cmp_eq); }
SNEPPXTensor* SNEPPX_tensor_ne(const SNEPPXTensor* a, const SNEPPXTensor* b) { return compare_op(a, b, cmp_ne); }
SNEPPXTensor* SNEPPX_tensor_lt(const SNEPPXTensor* a, const SNEPPXTensor* b) { return compare_op(a, b, cmp_lt); }
SNEPPXTensor* SNEPPX_tensor_le(const SNEPPXTensor* a, const SNEPPXTensor* b) { return compare_op(a, b, cmp_le); }
SNEPPXTensor* SNEPPX_tensor_gt(const SNEPPXTensor* a, const SNEPPXTensor* b) { return compare_op(a, b, cmp_gt); }
SNEPPXTensor* SNEPPX_tensor_ge(const SNEPPXTensor* a, const SNEPPXTensor* b) { return compare_op(a, b, cmp_ge); }

/* Helper: binary op with contiguity support */
#define BINARY_OP_F32(name, body) \
SNEPPXTensor* SNEPPX_tensor_##name(const SNEPPXTensor* a, const SNEPPXTensor* b) { \
    if (!a || !b) return NULL; \
    SNEPPXTensor* ta = NULL, *tb = NULL; \
    a = tensor_prep_contiguous(a, &ta); \
    b = tensor_prep_contiguous(b, &tb); \
    size_t sz = a->size < b->size ? a->size : b->size; \
    SNEPPXTensor* result = SNEPPX_tensor_create(a->shape, a->ndim, SNEPPX_FLOAT32); \
    if (!result) { if (ta) SNEPPX_tensor_destroy(ta); if (tb) SNEPPX_tensor_destroy(tb); return NULL; } \
    float* rd = (float*)result->data; \
    float* ad = (float*)a->data; \
    float* bd = (float*)b->data; \
    for (size_t i = 0; i < sz; i++) { body; } \
    if (ta) SNEPPX_tensor_destroy(ta); if (tb) SNEPPX_tensor_destroy(tb); \
    return result; \
}

/* add has broadcasting support */
SNEPPXTensor* SNEPPX_tensor_add(const SNEPPXTensor* a, const SNEPPXTensor* b) {
    if (!a || !b) return NULL;
    SNEPPXTensor* ta = NULL, *tb = NULL;
    a = tensor_prep_contiguous(a, &ta);
    b = tensor_prep_contiguous(b, &tb);
    size_t sz = a->size < b->size ? a->size : b->size;
    SNEPPXTensor* result = SNEPPX_tensor_create(a->shape, a->ndim, SNEPPX_FLOAT32);
    if (!result) { if (ta) SNEPPX_tensor_destroy(ta); if (tb) SNEPPX_tensor_destroy(tb); return NULL; }
    float* rd = (float*)result->data;
    float* ad = (float*)a->data;
    float* bd = (float*)b->data;
    if (a->size == b->size) {
        for (size_t i = 0; i < sz; i++) rd[i] = ad[i] + bd[i];
    } else {
        size_t last = a->shape[a->ndim - 1];
        for (size_t i = 0; i < a->size; i++) rd[i] = ad[i] + bd[i % last];
    }
    if (ta) SNEPPX_tensor_destroy(ta); if (tb) SNEPPX_tensor_destroy(tb);
    return result;
}
BINARY_OP_F32(sub, rd[i] = ad[i] - bd[i])
BINARY_OP_F32(mul, rd[i] = ad[i] * bd[i])
BINARY_OP_F32(div, rd[i] = ad[i] / (bd[i] + 1e-10f))
BINARY_OP_F32(minimum, rd[i] = ad[i] < bd[i] ? ad[i] : bd[i])
BINARY_OP_F32(maximum, rd[i] = ad[i] > bd[i] ? ad[i] : bd[i])
BINARY_OP_F32(pow, rd[i] = powf(ad[i], bd[i]))

/* ===== Contiguity helper =====
 * If src is non-contiguous, creates a contiguous copy and returns it via *out.
 * The caller must destroy *out after use (if non-NULL).
 * Returns a pointer usable for flat-indexed access (either src or the copy).
 */
static const SNEPPXTensor* tensor_prep_contiguous(const SNEPPXTensor* src, SNEPPXTensor** out) {
    *out = NULL;
    if (!src || SNEPPX_tensor_is_contiguous(src)) return src;
    *out = SNEPPX_tensor_contiguous(src);
    return *out ? *out : src;
}

static SNEPPXTensor* unary_op_f32(const SNEPPXTensor* src, float (*op)(float)) {
    if (!src) return NULL;
    SNEPPXTensor* tmp = NULL;
    src = tensor_prep_contiguous(src, &tmp);
    SNEPPXTensor* result = SNEPPX_tensor_create(src->shape, src->ndim, SNEPPX_FLOAT32);
    if (!result) { if (tmp) SNEPPX_tensor_destroy(tmp); return NULL; }
    float* sd = (float*)src->data;
    float* rd = (float*)result->data;
    for (size_t i = 0; i < src->size; i++) rd[i] = op(sd[i]);
    if (tmp) SNEPPX_tensor_destroy(tmp);
    return result;
}

static float negate_f32(float x) { return -x; }
static float sign_f32(float x) { return (x > 0) ? 1.0f : (x < 0) ? -1.0f : 0.0f; }

SNEPPXTensor* SNEPPX_tensor_neg(const SNEPPXTensor* src) { return unary_op_f32(src, negate_f32); }
SNEPPXTensor* SNEPPX_tensor_abs(const SNEPPXTensor* src) { return unary_op_f32(src, fabsf); }
SNEPPXTensor* SNEPPX_tensor_sign(const SNEPPXTensor* src) { return unary_op_f32(src, sign_f32); }
SNEPPXTensor* SNEPPX_tensor_floor(const SNEPPXTensor* src) { return unary_op_f32(src, floorf); }
SNEPPXTensor* SNEPPX_tensor_ceil(const SNEPPXTensor* src) { return unary_op_f32(src, ceilf); }
SNEPPXTensor* SNEPPX_tensor_round(const SNEPPXTensor* src) { return unary_op_f32(src, roundf); }
SNEPPXTensor* SNEPPX_tensor_trunc(const SNEPPXTensor* src) { return unary_op_f32(src, truncf); }
SNEPPXTensor* SNEPPX_tensor_exp(const SNEPPXTensor* src) { return unary_op_f32(src, expf); }
SNEPPXTensor* SNEPPX_tensor_log(const SNEPPXTensor* src) { return unary_op_f32(src, logf); }
SNEPPXTensor* SNEPPX_tensor_sqrt(const SNEPPXTensor* src) { return unary_op_f32(src, sqrtf); }
SNEPPXTensor* SNEPPX_tensor_sin(const SNEPPXTensor* src) { return unary_op_f32(src, sinf); }
SNEPPXTensor* SNEPPX_tensor_cos(const SNEPPXTensor* src) { return unary_op_f32(src, cosf); }
SNEPPXTensor* SNEPPX_tensor_tan(const SNEPPXTensor* src) { return unary_op_f32(src, tanf); }
SNEPPXTensor* SNEPPX_tensor_asin(const SNEPPXTensor* src) { return unary_op_f32(src, asinf); }
SNEPPXTensor* SNEPPX_tensor_acos(const SNEPPXTensor* src) { return unary_op_f32(src, acosf); }
SNEPPXTensor* SNEPPX_tensor_atan(const SNEPPXTensor* src) { return unary_op_f32(src, atanf); }
SNEPPXTensor* SNEPPX_tensor_sinh(const SNEPPXTensor* src) { return unary_op_f32(src, sinhf); }
SNEPPXTensor* SNEPPX_tensor_cosh(const SNEPPXTensor* src) { return unary_op_f32(src, coshf); }
SNEPPXTensor* SNEPPX_tensor_tanh(const SNEPPXTensor* src) { return unary_op_f32(src, tanhf); }

SNEPPXTensor* SNEPPX_tensor_sum(const SNEPPXTensor* src, size_t dim) {
    if (!src || dim >= src->ndim) return NULL;
    SNEPPXTensor* tmp = NULL;
    src = tensor_prep_contiguous(src, &tmp);
    size_t out_ndim = src->ndim;
    size_t out_shape[16];
    for (size_t i = 0; i < src->ndim; i++) out_shape[i] = (i == dim) ? 1 : src->shape[i];
    SNEPPXTensor* result = SNEPPX_tensor_zeros(out_shape, out_ndim, SNEPPX_FLOAT32);
    if (!result) { if (tmp) SNEPPX_tensor_destroy(tmp); return NULL; }
    float* sd = (float*)src->data;
    float* rd = (float*)result->data;
    size_t inner = 1, outer = 1, reduce = src->shape[dim];
    for (size_t i = 0; i < dim; i++) outer *= src->shape[i];
    for (size_t i = dim + 1; i < src->ndim; i++) inner *= src->shape[i];
    for (size_t o = 0; o < outer; o++) {
        for (size_t r = 0; r < reduce; r++) {
            for (size_t i = 0; i < inner; i++) {
                rd[o * inner + i] += sd[o * reduce * inner + r * inner + i];
            }
        }
    }
    if (tmp) SNEPPX_tensor_destroy(tmp);
    return result;
}

SNEPPXTensor* SNEPPX_tensor_mean(const SNEPPXTensor* src, size_t dim) {
    if (!src || dim >= src->ndim) return NULL;
    SNEPPXTensor* sum = SNEPPX_tensor_sum(src, dim);
    if (!sum) return NULL;
    float* rd = (float*)sum->data;
    float inv_n = 1.0f / (float)src->shape[dim];
    for (size_t i = 0; i < sum->size; i++) rd[i] *= inv_n;
    return sum;
}

SNEPPXTensor* SNEPPX_tensor_var(const SNEPPXTensor* src, size_t dim) {
    if (!src || dim >= src->ndim) return NULL;
    SNEPPXTensor* tmp = NULL;
    src = tensor_prep_contiguous(src, &tmp);
    SNEPPXTensor* mean = SNEPPX_tensor_mean(src, dim);
    if (!mean) { if (tmp) SNEPPX_tensor_destroy(tmp); return NULL; }
    size_t inner = 1, outer = 1, reduce = src->shape[dim];
    for (size_t i = 0; i < dim; i++) outer *= src->shape[i];
    for (size_t i = dim + 1; i < src->ndim; i++) inner *= src->shape[i];
    SNEPPXTensor* result = SNEPPX_tensor_zeros(mean->shape, mean->ndim, SNEPPX_FLOAT32);
    if (!result) { SNEPPX_tensor_destroy(mean); if (tmp) SNEPPX_tensor_destroy(tmp); return NULL; }
    float* sd = (float*)src->data;
    float* md = (float*)mean->data;
    float* rd = (float*)result->data;
    for (size_t o = 0; o < outer; o++) {
        for (size_t r = 0; r < reduce; r++) {
            for (size_t i = 0; i < inner; i++) {
                float d = sd[o * reduce * inner + r * inner + i] - md[o * inner + i];
                rd[o * inner + i] += d * d;
            }
        }
    }
    float inv_n = 1.0f / (float)reduce;
    for (size_t i = 0; i < result->size; i++) rd[i] *= inv_n;
    SNEPPX_tensor_destroy(mean);
    if (tmp) SNEPPX_tensor_destroy(tmp);
    return result;
}

SNEPPXTensor* SNEPPX_tensor_std(const SNEPPXTensor* src, size_t dim) {
    SNEPPXTensor* v = SNEPPX_tensor_var(src, dim);
    if (!v) return NULL;
    float* d = (float*)v->data;
    for (size_t i = 0; i < v->size; i++) d[i] = sqrtf(d[i]);
    return v;
}

float SNEPPX_tensor_min(const SNEPPXTensor* src) {
    if (!src || src->size == 0) return 0.0f;
    SNEPPXTensor* tmp = NULL;
    src = tensor_prep_contiguous(src, &tmp);
    float* d = (float*)src->data;
    float val = d[0];
    for (size_t i = 1; i < src->size; i++) if (d[i] < val) val = d[i];
    if (tmp) SNEPPX_tensor_destroy(tmp);
    return val;
}

float SNEPPX_tensor_max(const SNEPPXTensor* src) {
    if (!src || src->size == 0) return 0.0f;
    SNEPPXTensor* tmp = NULL;
    src = tensor_prep_contiguous(src, &tmp);
    float* d = (float*)src->data;
    float val = d[0];
    for (size_t i = 1; i < src->size; i++) if (d[i] > val) val = d[i];
    if (tmp) SNEPPX_tensor_destroy(tmp);
    return val;
}

size_t SNEPPX_tensor_argmin(const SNEPPXTensor* src) {
    if (!src || src->size == 0) return 0;
    SNEPPXTensor* tmp = NULL;
    src = tensor_prep_contiguous(src, &tmp);
    float* d = (float*)src->data;
    size_t idx = 0;
    for (size_t i = 1; i < src->size; i++) if (d[i] < d[idx]) idx = i;
    if (tmp) SNEPPX_tensor_destroy(tmp);
    return idx;
}

size_t SNEPPX_tensor_argmax(const SNEPPXTensor* src) {
    if (!src || src->size == 0) return 0;
    SNEPPXTensor* tmp = NULL;
    src = tensor_prep_contiguous(src, &tmp);
    float* d = (float*)src->data;
    size_t idx = 0;
    for (size_t i = 1; i < src->size; i++) if (d[i] > d[idx]) idx = i;
    if (tmp) SNEPPX_tensor_destroy(tmp);
    return idx;
}

SNEPPXTensor* SNEPPX_tensor_cumsum(const SNEPPXTensor* src, size_t dim) {
    if (!src || dim >= src->ndim) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_copy(src);
    if (!result) return NULL;
    size_t inner = 1, outer = 1;
    for (size_t i = 0; i < dim; i++) outer *= src->shape[i];
    for (size_t i = dim + 1; i < src->ndim; i++) inner *= src->shape[i];
    size_t reduce = src->shape[dim];
    float* rd = (float*)result->data;
    for (size_t o = 0; o < outer; o++) {
        for (size_t i = 0; i < inner; i++) {
            for (size_t r = 1; r < reduce; r++) {
                rd[o * reduce * inner + r * inner + i] += rd[o * reduce * inner + (r - 1) * inner + i];
            }
        }
    }
    return result;
}

SNEPPXTensor* SNEPPX_tensor_cumprod(const SNEPPXTensor* src, size_t dim) {
    if (!src || dim >= src->ndim) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_copy(src);
    if (!result) return NULL;
    size_t inner = 1, outer = 1;
    for (size_t i = 0; i < dim; i++) outer *= src->shape[i];
    for (size_t i = dim + 1; i < src->ndim; i++) inner *= src->shape[i];
    size_t reduce = src->shape[dim];
    float* rd = (float*)result->data;
    for (size_t o = 0; o < outer; o++) {
        for (size_t i = 0; i < inner; i++) {
            for (size_t r = 1; r < reduce; r++) {
                rd[o * reduce * inner + r * inner + i] *= rd[o * reduce * inner + (r - 1) * inner + i];
            }
        }
    }
    return result;
}

float SNEPPX_tensor_dot(const SNEPPXTensor* a, const SNEPPXTensor* b) {
    if (!a || !b || a->size != b->size) return 0.0f;
    float* ad = (float*)a->data;
    float* bd = (float*)b->data;
    float sum = 0.0f;
    for (size_t i = 0; i < a->size; i++) sum += ad[i] * bd[i];
    return sum;
}

SNEPPXTensor* SNEPPX_tensor_matmul(const SNEPPXTensor* a, const SNEPPXTensor* b) {
    if (!a || !b || a->ndim != 2 || b->ndim != 2) return NULL;
    SNEPPXTensor* ta = NULL, *tb = NULL;
    a = tensor_prep_contiguous(a, &ta);
    b = tensor_prep_contiguous(b, &tb);
    size_t m = a->shape[0], k = a->shape[1], n = b->shape[1];
    if (k != b->shape[0]) { if (ta) SNEPPX_tensor_destroy(ta); if (tb) SNEPPX_tensor_destroy(tb); return NULL; }
    size_t shape_c[] = {m, n};
    SNEPPXTensor* result = SNEPPX_tensor_zeros(shape_c, 2, SNEPPX_FLOAT32);
    if (!result) { if (ta) SNEPPX_tensor_destroy(ta); if (tb) SNEPPX_tensor_destroy(tb); return NULL; }
    float* ad = (float*)a->data;
    float* bd = (float*)b->data;
    float* rd = (float*)result->data;
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++)
            for (size_t l = 0; l < k; l++)
                rd[i * n + j] += ad[i * k + l] * bd[l * n + j];
    if (ta) SNEPPX_tensor_destroy(ta); if (tb) SNEPPX_tensor_destroy(tb);
    return result;
}

SNEPPXTensor* SNEPPX_tensor_transpose(const SNEPPXTensor* src, size_t dim1, size_t dim2) {
    if (!src || dim1 >= src->ndim || dim2 >= src->ndim) return NULL;
    size_t* axes = (size_t*)aligned_alloc_wrapper(src->ndim * sizeof(size_t), 64);
    if (!axes) return NULL;
    for (size_t i = 0; i < src->ndim; i++) axes[i] = i;
    axes[dim1] = dim2;
    axes[dim2] = dim1;
    SNEPPXTensor* result = SNEPPX_tensor_permute(src, axes);
    SNEPPX_free(axes, src->ndim * sizeof(size_t));
    return result;
}

SNEPPXTensor* SNEPPX_tensor_inverse(const SNEPPXTensor* src) {
    if (!src || src->ndim != 2 || src->shape[0] != src->shape[1]) return NULL;
    size_t n = src->shape[0];
    SNEPPXTensor* result = SNEPPX_tensor_create(src->shape, 2, SNEPPX_FLOAT32);
    if (!result) return NULL;
    float* a = (float*)aligned_alloc_wrapper(n * n * 2 * sizeof(float), 64);
    if (!a) { SNEPPX_tensor_destroy(result); return NULL; }
    float* s = (float*)src->data;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) a[i * 2 * n + j] = s[i * n + j];
        for (size_t j = 0; j < n; j++) a[i * 2 * n + n + j] = (i == j) ? 1.0f : 0.0f;
    }
    for (size_t i = 0; i < n; i++) {
        size_t pivot = i;
        for (size_t r = i + 1; r < n; r++)
            if (fabsf(a[r * 2 * n + i]) > fabsf(a[pivot * 2 * n + i])) pivot = r;
        if (pivot != i)
            for (size_t j = 0; j < 2 * n; j++) { float t = a[i * 2 * n + j]; a[i * 2 * n + j] = a[pivot * 2 * n + j]; a[pivot * 2 * n + j] = t; }
        float piv = a[i * 2 * n + i];
        if (fabsf(piv) < 1e-10f) { SNEPPX_free(a, n * n * 2 * sizeof(float)); SNEPPX_tensor_destroy(result); return NULL; }
        for (size_t j = 0; j < 2 * n; j++) a[i * 2 * n + j] /= piv;
        for (size_t k = 0; k < n; k++) {
            if (k == i) continue;
            float factor = a[k * 2 * n + i];
            for (size_t j = 0; j < 2 * n; j++) a[k * 2 * n + j] -= factor * a[i * 2 * n + j];
        }
    }
    float* rd = (float*)result->data;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            rd[i * n + j] = a[i * 2 * n + n + j];
    SNEPPX_free(a, n * n * 2 * sizeof(float));
    return result;
}

float SNEPPX_tensor_det(const SNEPPXTensor* src) {
    if (!src || src->ndim != 2 || src->shape[0] != src->shape[1]) return 0.0f;
    size_t n = src->shape[0];
    float* m = (float*)aligned_alloc_wrapper(n * n * sizeof(float), 64);
    if (!m) return 0.0f;
    float* s = (float*)src->data;
    for (size_t i = 0; i < n * n; i++) m[i] = s[i];
    float det = 1.0f;
    for (size_t i = 0; i < n; i++) {
        size_t pivot = i;
        for (size_t j = i + 1; j < n; j++)
            if (fabsf(m[j * n + i]) > fabsf(m[pivot * n + i])) pivot = j;
        if (pivot != i) {
            for (size_t j = 0; j < n; j++) { float t = m[i * n + j]; m[i * n + j] = m[pivot * n + j]; m[pivot * n + j] = t; }
            det = -det;
        }
        if (fabsf(m[i * n + i]) < 1e-10f) { SNEPPX_free(m, n * n * sizeof(float)); return 0.0f; }
        det *= m[i * n + i];
        for (size_t j = i + 1; j < n; j++) {
            float f = m[j * n + i] / m[i * n + i];
            for (size_t k = i + 1; k < n; k++)
                m[j * n + k] -= f * m[i * n + k];
        }
    }
    SNEPPX_free(m, n * n * sizeof(float));
    return det;
}

SNEPPXTensor* SNEPPX_tensor_conv1d(const SNEPPXTensor* input, const SNEPPXTensor* kernel, size_t stride, size_t padding) {
    if (!input || !kernel || stride == 0) return NULL;
    size_t L = input->shape[input->ndim - 1];
    size_t K = kernel->shape[kernel->ndim - 1];
    size_t out_len = (L + 2 * padding - K) / stride + 1;
    if (out_len == 0) return NULL;
    size_t out_ndim = input->ndim;
    size_t out_shape[16];
    for (size_t i = 0; i < out_ndim - 1; i++) out_shape[i] = input->shape[i];
    out_shape[out_ndim - 1] = out_len;
    SNEPPXTensor* result = SNEPPX_tensor_zeros(out_shape, out_ndim, SNEPPX_FLOAT32);
    if (!result) return NULL;
    float* id = (float*)input->data;
    float* kd = (float*)kernel->data;
    float* rd = (float*)result->data;
    size_t in_channels = input->shape[input->ndim - 2];
    size_t kernel_channels = kernel->shape[kernel->ndim - 2];
    size_t batch = input->size / (in_channels * L);
    for (size_t b = 0; b < batch; b++) {
        for (size_t c_in = 0; c_in < in_channels; c_in++) {
            for (size_t c_out = 0; c_out < kernel_channels; c_out++) {
                for (size_t pos = 0; pos < out_len; pos++) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < K; k++) {
                        int inp_pos = (int)(pos * stride + k) - (int)padding;
                        if (inp_pos >= 0 && inp_pos < (int)L)
                            sum += id[b * in_channels * L + c_in * L + inp_pos] * kd[c_out * in_channels * K + c_in * K + k];
                    }
                    rd[b * kernel_channels * out_len + c_out * out_len + pos] += sum;
                }
            }
        }
    }
    return result;
}

SNEPPXTensor* SNEPPX_tensor_conv2d(const SNEPPXTensor* input, const SNEPPXTensor* kernel, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w) {
    if (!input || !kernel || stride_h == 0 || stride_w == 0) return NULL;
    size_t H = input->shape[input->ndim - 2], W = input->shape[input->ndim - 1];
    size_t KH = kernel->shape[kernel->ndim - 2], KW = kernel->shape[kernel->ndim - 1];
    size_t out_h = (H + 2 * pad_h - KH) / stride_h + 1;
    size_t out_w = (W + 2 * pad_w - KW) / stride_w + 1;
    if (out_h == 0 || out_w == 0) return NULL;
    size_t out_ndim = input->ndim;
    size_t out_shape[16];
    for (size_t i = 0; i < out_ndim - 2; i++) out_shape[i] = input->shape[i];
    out_shape[out_ndim - 2] = out_h;
    out_shape[out_ndim - 1] = out_w;
    SNEPPXTensor* result = SNEPPX_tensor_zeros(out_shape, out_ndim, SNEPPX_FLOAT32);
    if (!result) return NULL;
    float* id = (float*)input->data;
    float* kd = (float*)kernel->data;
    float* rd = (float*)result->data;
    size_t in_channels = input->shape[input->ndim - 3];
    size_t out_channels = kernel->shape[kernel->ndim - 3];
    size_t batch = input->size / (in_channels * H * W);
    for (size_t b = 0; b < batch; b++) {
        for (size_t co = 0; co < out_channels; co++) {
            for (size_t ci = 0; ci < in_channels; ci++) {
                for (size_t oh = 0; oh < out_h; oh++) {
                    for (size_t ow = 0; ow < out_w; ow++) {
                        float sum = 0.0f;
                        for (size_t kh = 0; kh < KH; kh++) {
                            int ih = (int)(oh * stride_h + kh) - (int)pad_h;
                            for (size_t kw = 0; kw < KW; kw++) {
                                int iw = (int)(ow * stride_w + kw) - (int)pad_w;
                                if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W)
                                    sum += id[b * in_channels * H * W + ci * H * W + (size_t)ih * W + (size_t)iw]
                                         * kd[co * in_channels * KH * KW + ci * KH * KW + kh * KW + kw];
                            }
                        }
                        rd[b * out_channels * out_h * out_w + co * out_h * out_w + oh * out_w + ow] += sum;
                    }
                }
            }
        }
    }
    return result;
}

SNEPPXTensor* SNEPPX_tensor_pool1d(const SNEPPXTensor* src, size_t kernel_size, size_t stride) {
    if (!src || kernel_size == 0 || stride == 0) return NULL;
    size_t L = src->shape[src->ndim - 1];
    size_t out_len = (L - kernel_size) / stride + 1;
    if (out_len == 0) return NULL;
    size_t out_ndim = src->ndim;
    size_t out_shape[16];
    for (size_t i = 0; i < out_ndim - 1; i++) out_shape[i] = src->shape[i];
    out_shape[out_ndim - 1] = out_len;
    SNEPPXTensor* result = SNEPPX_tensor_zeros(out_shape, out_ndim, SNEPPX_FLOAT32);
    if (!result) return NULL;
    float* sd = (float*)src->data;
    float* rd = (float*)result->data;
    size_t channels = src->shape[src->ndim - 2];
    size_t batch = src->size / (channels * L);
    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t pos = 0; pos < out_len; pos++) {
                float max_val = -FLT_MAX;
                for (size_t k = 0; k < kernel_size; k++)
                    if (sd[b * channels * L + c * L + pos * stride + k] > max_val)
                        max_val = sd[b * channels * L + c * L + pos * stride + k];
                rd[b * channels * out_len + c * out_len + pos] = max_val;
            }
        }
    }
    return result;
}

SNEPPXTensor* SNEPPX_tensor_pool2d(const SNEPPXTensor* src, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w) {
    if (!src || kernel_h == 0 || kernel_w == 0 || stride_h == 0 || stride_w == 0) return NULL;
    size_t H = src->shape[src->ndim - 2], W = src->shape[src->ndim - 1];
    size_t out_h = (H - kernel_h) / stride_h + 1;
    size_t out_w = (W - kernel_w) / stride_w + 1;
    if (out_h == 0 || out_w == 0) return NULL;
    size_t out_ndim = src->ndim;
    size_t out_shape[16];
    for (size_t i = 0; i < out_ndim - 2; i++) out_shape[i] = src->shape[i];
    out_shape[out_ndim - 2] = out_h;
    out_shape[out_ndim - 1] = out_w;
    SNEPPXTensor* result = SNEPPX_tensor_zeros(out_shape, out_ndim, SNEPPX_FLOAT32);
    if (!result) return NULL;
    float* sd = (float*)src->data;
    float* rd = (float*)result->data;
    size_t channels = src->shape[src->ndim - 3];
    size_t batch = src->size / (channels * H * W);
    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t oh = 0; oh < out_h; oh++) {
                for (size_t ow = 0; ow < out_w; ow++) {
                    float max_val = -FLT_MAX;
                    for (size_t kh = 0; kh < kernel_h; kh++)
                        for (size_t kw = 0; kw < kernel_w; kw++)
                            if (sd[b * channels * H * W + c * H * W + (oh * stride_h + kh) * W + (ow * stride_w + kw)] > max_val)
                                max_val = sd[b * channels * H * W + c * H * W + (oh * stride_h + kh) * W + (ow * stride_w + kw)];
                    rd[b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow] = max_val;
                }
            }
        }
    }
    return result;
}

SNEPPXTensor* SNEPPX_tensor_softmax(const SNEPPXTensor* src, size_t dim) {
    if (!src || dim >= src->ndim) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_copy(src);
    if (!result) return NULL;
    size_t inner = 1, outer = 1, reduce = src->shape[dim];
    for (size_t i = 0; i < dim; i++) outer *= src->shape[i];
    for (size_t i = dim + 1; i < src->ndim; i++) inner *= src->shape[i];
    float* rd = (float*)result->data;
    for (size_t o = 0; o < outer; o++) {
        for (size_t i = 0; i < inner; i++) {
            float max_val = rd[o * reduce * inner + i];
            for (size_t r = 1; r < reduce; r++) {
                float v = rd[o * reduce * inner + r * inner + i];
                if (v > max_val) max_val = v;
            }
            float sum = 0.0f;
            for (size_t r = 0; r < reduce; r++) {
                float ex = expf(rd[o * reduce * inner + r * inner + i] - max_val);
                rd[o * reduce * inner + r * inner + i] = ex;
                sum += ex;
            }
            float inv_sum = 1.0f / (sum + 1e-10f);
            for (size_t r = 0; r < reduce; r++) {
                rd[o * reduce * inner + r * inner + i] *= inv_sum;
            }
        }
    }
    return result;
}

SNEPPXTensor* SNEPPX_tensor_log_softmax(const SNEPPXTensor* src, size_t dim) {
    if (!src || dim >= src->ndim) return NULL;
    SNEPPXTensor* sm = SNEPPX_tensor_softmax(src, dim);
    if (!sm) return NULL;
    float* d = (float*)sm->data;
    for (size_t i = 0; i < sm->size; i++) d[i] = logf(d[i] + 1e-10f);
    return sm;
}

SNEPPXTensor* SNEPPX_tensor_relu(const SNEPPXTensor* src) {
    if (!src) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_copy(src);
    if (!result) return NULL;
    float* rd = (float*)result->data;
    for (size_t i = 0; i < result->size; i++) if (rd[i] < 0) rd[i] = 0.0f;
    return result;
}

SNEPPXTensor* SNEPPX_tensor_gelu(const SNEPPXTensor* src) {
    if (!src) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_copy(src);
    if (!result) return NULL;
    float* rd = (float*)result->data;
    for (size_t i = 0; i < result->size; i++) {
        float x = rd[i];
        rd[i] = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    }
    return result;
}

SNEPPXTensor* SNEPPX_tensor_silu(const SNEPPXTensor* src) {
    if (!src) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_copy(src);
    if (!result) return NULL;
    float* rd = (float*)result->data;
    for (size_t i = 0; i < result->size; i++) {
        float x = rd[i];
        rd[i] = x / (1.0f + expf(-x));
    }
    return result;
}

SNEPPXTensor* SNEPPX_tensor_sigmoid(const SNEPPXTensor* src) {
    if (!src) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_copy(src);
    if (!result) return NULL;
    float* rd = (float*)result->data;
    for (size_t i = 0; i < result->size; i++) {
        rd[i] = 1.0f / (1.0f + expf(-rd[i]));
    }
    return result;
}

SNEPPXTensor* SNEPPX_tensor_dropout(const SNEPPXTensor* src, float rate, unsigned int seed) {
    (void)seed;
    if (!src) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_copy(src);
    if (!result) return NULL;
    float* rd = (float*)result->data;
    unsigned long s = (seed) ? (unsigned long)seed : 123456789;
    float scale = 1.0f / (1.0f - rate);
    for (size_t i = 0; i < result->size; i++) {
        s = s * 1103515245UL + 12345UL;
        float r = (float)((s >> 16) & 0x7FFF) / 32767.0f;
        if (r < rate) rd[i] = 0.0f;
        else rd[i] *= scale;
    }
    return result;
}

SNEPPXTensor* SNEPPX_tensor_layer_norm(const SNEPPXTensor* src, const SNEPPXTensor* gamma, const SNEPPXTensor* beta, float eps) {
    if (!src) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_copy(src);
    if (!result) return NULL;
    size_t d = src->shape[src->ndim - 1];
    size_t n = src->size / d;
    float* rd = (float*)result->data;
    float* gd = gamma ? (float*)gamma->data : NULL;
    float* bd = beta ? (float*)beta->data : NULL;
    (void)gd; (void)bd;
    for (size_t i = 0; i < n; i++) {
        float sum = 0.0f, sum2 = 0.0f;
        for (size_t j = 0; j < d; j++) { float v = rd[i * d + j]; sum += v; sum2 += v * v; }
        float mean = sum / (float)d;
        float var = sum2 / (float)d - mean * mean;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (size_t j = 0; j < d; j++) {
            float x = (rd[i * d + j] - mean) * inv_std;
            if (gd) x *= gd[j];
            if (bd) x += bd[j];
            rd[i * d + j] = x;
        }
    }
    return result;
}

SNEPPXTensor* SNEPPX_tensor_batch_norm(const SNEPPXTensor* src, const SNEPPXTensor* gamma, const SNEPPXTensor* beta, const SNEPPXTensor* running_mean, const SNEPPXTensor* running_var, float eps) {
    if (!src) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_copy(src);
    if (!result) return NULL;
    float* rd = (float*)result->data;
    float* gd = gamma ? (float*)gamma->data : NULL;
    float* bd = beta ? (float*)beta->data : NULL;
    float* rm = running_mean ? (float*)running_mean->data : NULL;
    float* rv = running_var ? (float*)running_var->data : NULL;
    size_t c = src->shape[1];
    size_t hw = src->size / c;
    (void)gd; (void)bd;
    for (size_t ch = 0; ch < c; ch++) {
        float mean = rm ? rm[ch] : 0.0f;
        float var = rv ? rv[ch] : 1.0f;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (size_t p = 0; p < hw; p++) {
            size_t idx = p * c + ch;
            float x = (rd[idx] - mean) * inv_std;
            if (gd) x *= gd[ch];
            if (bd) x += bd[ch];
            rd[idx] = x;
        }
    }
    return result;
}

SNEPPXTensor* SNEPPX_tensor_group_norm(const SNEPPXTensor* src, const SNEPPXTensor* gamma, const SNEPPXTensor* beta, size_t num_groups, float eps) {
    if (!src) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_copy(src);
    if (!result) return NULL;
    float* rd = (float*)result->data;
    size_t c = src->shape[1];
    size_t group_size = c / num_groups;
    size_t hw = src->size / (src->shape[0] * c);
    size_t n = src->shape[0];
    float* gd = gamma ? (float*)gamma->data : NULL;
    float* bd = beta ? (float*)beta->data : NULL;
    for (size_t b = 0; b < n; b++) {
        for (size_t g = 0; g < num_groups; g++) {
            size_t start_c = g * group_size;
            float sum = 0.0f, sum2 = 0.0f;
            size_t count = 0;
            for (size_t ch = start_c; ch < start_c + group_size && ch < c; ch++) {
                for (size_t p = 0; p < hw; p++) {
                    float v = rd[b * c * hw + ch * hw + p];
                    sum += v; sum2 += v * v; count++;
                }
            }
            float mean = sum / (float)count;
            float var = sum2 / (float)count - mean * mean;
            float inv_std = 1.0f / sqrtf(var + eps);
            for (size_t ch = start_c; ch < start_c + group_size && ch < c; ch++) {
                for (size_t p = 0; p < hw; p++) {
                    size_t idx = b * c * hw + ch * hw + p;
                    float x = (rd[idx] - mean) * inv_std;
                    if (gd) x *= gd[ch];
                    if (bd) x += bd[ch];
                    rd[idx] = x;
                }
            }
        }
    }
    return result;
}

SNEPPXTensor* SNEPPX_tensor_instance_norm(const SNEPPXTensor* src, const SNEPPXTensor* gamma, const SNEPPXTensor* beta, float eps) {
    if (!src) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_copy(src);
    if (!result) return NULL;
    size_t n = src->shape[0], c = src->shape[1];
    size_t hw = src->size / (n * c);
    float* rd = (float*)result->data;
    float* gd = gamma ? (float*)gamma->data : NULL;
    float* bd = beta ? (float*)beta->data : NULL;
    for (size_t b = 0; b < n; b++) {
        for (size_t ch = 0; ch < c; ch++) {
            float sum = 0.0f, sum2 = 0.0f;
            for (size_t p = 0; p < hw; p++) {
                float v = rd[b * c * hw + ch * hw + p];
                sum += v; sum2 += v * v;
            }
            float mean = sum / (float)hw;
            float var = sum2 / (float)hw - mean * mean;
            float inv_std = 1.0f / sqrtf(var + eps);
            for (size_t p = 0; p < hw; p++) {
                size_t idx = b * c * hw + ch * hw + p;
                float x = (rd[idx] - mean) * inv_std;
                if (gd) x *= gd[ch];
                if (bd) x += bd[ch];
                rd[idx] = x;
            }
        }
    }
    return result;
}

static size_t read_index(const SNEPPXTensor* t, size_t i) {
    switch (t->dtype) {
        case SNEPPX_INT32:  return (size_t)((int32_t*)t->data)[i];
        case SNEPPX_INT64:  return (size_t)((int64_t*)t->data)[i];
        case SNEPPX_FLOAT32: return (size_t)((float*)t->data)[i];
        default: return ((size_t*)t->data)[i];
    }
}

SNEPPXTensor* SNEPPX_tensor_embedding(const SNEPPXTensor* weight, const SNEPPXTensor* indices) {
    if (!weight || !indices || weight->ndim != 2) return NULL;
    size_t num_embeddings = weight->shape[0];
    size_t embedding_dim = weight->shape[1];
    size_t n = indices->size;
    size_t out_shape[] = {n, embedding_dim};
    SNEPPXTensor* result = SNEPPX_tensor_empty(out_shape, 2, weight->dtype);
    if (!result) return NULL;
    unsigned char* wd = (unsigned char*)weight->data;
    unsigned char* rd = (unsigned char*)result->data;
    size_t is = weight->item_size;
    for (size_t i = 0; i < n; i++) {
        size_t emb_idx = read_index(indices, i);
        if (emb_idx >= num_embeddings) emb_idx = 0;
        memcpy(rd + i * embedding_dim * is, wd + emb_idx * embedding_dim * is, embedding_dim * is);
    }
    return result;
}

SNEPPXTensor* SNEPPX_tensor_cross_entropy(const SNEPPXTensor* pred, const SNEPPXTensor* target) {
    if (!pred || !target) return NULL;
    size_t n = pred->size / pred->shape[pred->ndim - 1];
    size_t c = pred->shape[pred->ndim - 1];
    size_t shape[] = {1};
    SNEPPXTensor* result = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    if (!result) return NULL;
    float* pd = (float*)pred->data;
    float* td = (float*)target->data;
    float* rd = (float*)result->data;
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < c; j++) {
            size_t idx = i * c + j;
            sum -= td[idx] * logf(pd[idx] + 1e-10f);
        }
    }
    rd[0] = sum / (float)n;
    return result;
}

SNEPPXTensor* SNEPPX_tensor_mse_loss(const SNEPPXTensor* pred, const SNEPPXTensor* target) {
    if (!pred || !target || pred->size != target->size) return NULL;
    size_t shape[] = {1};
    SNEPPXTensor* result = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    if (!result) return NULL;
    float* pd = (float*)pred->data;
    float* td = (float*)target->data;
    float* rd = (float*)result->data;
    float sum = 0.0f;
    for (size_t i = 0; i < pred->size; i++) { float d = pd[i] - td[i]; sum += d * d; }
    rd[0] = sum / (float)pred->size;
    return result;
}

SNEPPXTensor* SNEPPX_tensor_mae_loss(const SNEPPXTensor* pred, const SNEPPXTensor* target) {
    if (!pred || !target || pred->size != target->size) return NULL;
    size_t shape[] = {1};
    SNEPPXTensor* result = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    if (!result) return NULL;
    float* pd = (float*)pred->data;
    float* td = (float*)target->data;
    float* rd = (float*)result->data;
    float sum = 0.0f;
    for (size_t i = 0; i < pred->size; i++) sum += fabsf(pd[i] - td[i]);
    rd[0] = sum / (float)pred->size;
    return result;
}

SNEPPXTensor* SNEPPX_tensor_nll_loss(const SNEPPXTensor* pred, const SNEPPXTensor* target) {
    if (!pred || !target) return NULL;
    size_t n = pred->size / pred->shape[pred->ndim - 1];
    size_t c = pred->shape[pred->ndim - 1];
    size_t shape[] = {1};
    SNEPPXTensor* result = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    if (!result) return NULL;
    float* pd = (float*)pred->data;
    float* td = (float*)target->data;
    float* rd = (float*)result->data;
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        size_t target_idx = (size_t)td[i];
        if (target_idx >= c) target_idx = 0;
        sum -= logf(pd[i * c + target_idx] + 1e-10f);
    }
    rd[0] = sum / (float)n;
    return result;
}

SNEPPXTensor* SNEPPX_tensor_kl_div(const SNEPPXTensor* pred, const SNEPPXTensor* target) {
    if (!pred || !target || pred->size != target->size) return NULL;
    size_t shape[] = {1};
    SNEPPXTensor* result = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    if (!result) return NULL;
    float* pd = (float*)pred->data;
    float* td = (float*)target->data;
    float* rd = (float*)result->data;
    float sum = 0.0f;
    for (size_t i = 0; i < pred->size; i++) {
        sum += td[i] * logf((td[i] + 1e-10f) / (pd[i] + 1e-10f));
    }
    rd[0] = sum;
    return result;
}

SNEPPXTensor* SNEPPX_tensor_binary_cross_entropy(const SNEPPXTensor* pred, const SNEPPXTensor* target) {
    if (!pred || !target || pred->size != target->size) return NULL;
    size_t shape[] = {1};
    SNEPPXTensor* result = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    if (!result) return NULL;
    float* pd = (float*)pred->data;
    float* td = (float*)target->data;
    float* rd = (float*)result->data;
    float sum = 0.0f;
    for (size_t i = 0; i < pred->size; i++) {
        sum += -td[i] * logf(pd[i] + 1e-10f) - (1.0f - td[i]) * logf(1.0f - pd[i] + 1e-10f);
    }
    rd[0] = sum / (float)pred->size;
    return result;
}

void SNEPPX_tensor_print(const SNEPPXTensor* tensor) {
    if (!tensor) return;
    printf("Tensor shape: [");
    for (size_t i = 0; i < tensor->ndim; i++) {
        printf("%zu", tensor->shape[i]);
        if (i < tensor->ndim - 1) printf(", ");
    }
    printf("]\n");
    printf("Dtype: %s\n", SNEPPX_tensor_dtype_name(tensor->dtype));
    printf("Size: %zu elements\n", tensor->size);
    printf("Device: %s\n", tensor->device == SNEPPX_DEVICE_CPU ? "cpu" : "cuda");
    if (tensor->dtype == SNEPPX_FLOAT32) {
        size_t n = tensor->size < 20 ? tensor->size : 20;
        printf("Data (first %zu): [", n);
        float* data = (float*)tensor->data;
        for (size_t i = 0; i < n; i++) {
            printf("%f", data[i]);
            if (i < n - 1) printf(", ");
        }
        printf("]\n");
    } else if (tensor->dtype == SNEPPX_INT32) {
        size_t n = tensor->size < 20 ? tensor->size : 20;
        printf("Data (first %zu): [", n);
        int32_t* data = (int32_t*)tensor->data;
        for (size_t i = 0; i < n; i++) {
            printf("%d", data[i]);
            if (i < n - 1) printf(", ");
        }
        printf("]\n");
    }
}

SNEPPXTensor* SNEPPX_tensor_contiguous(const SNEPPXTensor* src) {
    if (!src) return NULL;
    if (SNEPPX_tensor_is_contiguous(src)) {
        SNEPPX_storage_retain(src->storage);
        return SNEPPX_tensor_as_strided(src, src->offset, src->shape, src->ndim, src->strides);
    }
    SNEPPXTensor* result = SNEPPX_tensor_create(src->shape, src->ndim, src->dtype);
    if (!result) return NULL;
    /* Strided copy: iterate over all indices */
    unsigned char* dst = (unsigned char*)result->data;
    unsigned char* src_data = (unsigned char*)src->data;
    size_t is = src->item_size;
    size_t* indices = (size_t*)aligned_alloc_wrapper(src->ndim * sizeof(size_t), 64);
    if (!indices) { SNEPPX_tensor_destroy(result); return NULL; }
    memset(indices, 0, src->ndim * sizeof(size_t));
    for (size_t flat = 0; flat < src->size; flat++) {
        size_t tmp = flat;
        for (size_t i = src->ndim; i > 0; i--) {
            indices[i - 1] = (src->strides[i - 1] > 0) ? (tmp / src->strides[i - 1]) % src->shape[i - 1] : 0;
            if (src->strides[i - 1] > 0) tmp %= src->strides[i - 1];
        }
        size_t offset = 0;
        for (size_t i = 0; i < src->ndim; i++)
            offset += indices[i] * src->strides[i];
        memcpy(dst + flat * is, src_data + offset * is, is);
    }
    SNEPPX_free(indices, src->ndim * sizeof(size_t));
    return result;
}

size_t SNEPPX_tensor_dtype_size(SNEPPXDtype dtype) {
    switch (dtype) {
        case SNEPPX_FLOAT8:    return 1;
        case SNEPPX_FLOAT16:   return 2;
        case SNEPPX_BFLOAT16:  return 2;
        case SNEPPX_FLOAT32:   return sizeof(float);
        case SNEPPX_FLOAT64:   return sizeof(double);
        case SNEPPX_INT8:      return 1;
        case SNEPPX_INT16:     return 2;
        case SNEPPX_INT32:     return sizeof(int32_t);
        case SNEPPX_INT64:     return sizeof(int64_t);
        case SNEPPX_UINT8:     return 1;
        case SNEPPX_BOOL:      return 1;
        case SNEPPX_COMPLEX64: return 8;
        case SNEPPX_COMPLEX128: return 16;
    }
    return 0;
}

size_t SNEPPX_tensor_numel(const SNEPPXTensor* tensor) {
    if (!tensor) return 0;
    return tensor->size;
}

int SNEPPX_tensor_is_contiguous(const SNEPPXTensor* tensor) {
    if (!tensor || tensor->ndim == 0) return 1;
    size_t expected = 1;
    for (size_t i = tensor->ndim; i > 0; i--) {
        if (tensor->strides[i - 1] != expected) return 0;
        expected *= tensor->shape[i - 1];
    }
    return 1;
}

const char* SNEPPX_tensor_dtype_name(SNEPPXDtype dtype) {
    switch (dtype) {
        case SNEPPX_FLOAT8:    return "float8";
        case SNEPPX_FLOAT16:   return "float16";
        case SNEPPX_BFLOAT16:  return "bfloat16";
        case SNEPPX_FLOAT32:   return "float32";
        case SNEPPX_FLOAT64:   return "float64";
        case SNEPPX_INT8:      return "int8";
        case SNEPPX_INT16:     return "int16";
        case SNEPPX_INT32:     return "int32";
        case SNEPPX_INT64:     return "int64";
        case SNEPPX_UINT8:     return "uint8";
        case SNEPPX_BOOL:      return "bool";
        case SNEPPX_COMPLEX64: return "complex64";
        case SNEPPX_COMPLEX128:return "complex128";
    }
    return "unknown";
}
