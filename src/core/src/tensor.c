#include "arix_tensor.h"
#include "arix_memory.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>

static size_t compute_offset(const ArixTensor* tensor, const size_t* indices) {
    size_t offset = 0;
    for (size_t i = 0; i < tensor->ndim; i++) {
        offset += indices[i] * tensor->strides[i];
    }
    return offset;
}

static size_t compute_offset_flat(const ArixTensor* tensor, size_t flat_idx) {
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
    return arix_malloc(size, alignment);
}

ArixTensor* arix_tensor_create(const size_t* shape, size_t ndim, ArixDtype dtype) {
    ArixTensor* tensor = (ArixTensor*)aligned_alloc_wrapper(sizeof(ArixTensor), 64);
    if (!tensor) return NULL;

    tensor->ndim = ndim;
    tensor->dtype = dtype;
    tensor->item_size = arix_tensor_dtype_size(dtype);
    tensor->device = ARIX_DEVICE_CPU;
    tensor->device_id = 0;
    tensor->layout = ARIX_LAYOUT_ROW_MAJOR;
    tensor->owns_data = 1;
    tensor->backend_handle = NULL;

    tensor->shape = (size_t*)aligned_alloc_wrapper(ndim * sizeof(size_t), 64);
    tensor->strides = (size_t*)aligned_alloc_wrapper(ndim * sizeof(size_t), 64);
    if (!tensor->shape || !tensor->strides) {
        arix_free(tensor->shape, ndim * sizeof(size_t));
        arix_free(tensor->strides, ndim * sizeof(size_t));
        arix_free(tensor, sizeof(ArixTensor));
        return NULL;
    }

    size_t total = 1;
    for (size_t i = 0; i < ndim; i++) {
        tensor->shape[i] = shape[i];
        total *= shape[i];
    }
    tensor->size = total;

    size_t stride = 1;
    for (size_t i = ndim; i > 0; i--) {
        tensor->strides[i - 1] = stride;
        stride *= shape[i - 1];
    }

    tensor->data = aligned_alloc_wrapper(total * tensor->item_size, 64);
    if (!tensor->data) {
        arix_free(tensor->shape, ndim * sizeof(size_t));
        arix_free(tensor->strides, ndim * sizeof(size_t));
        arix_free(tensor, sizeof(ArixTensor));
        return NULL;
    }

    return tensor;
}

void arix_tensor_destroy(ArixTensor* tensor) {
    if (!tensor) return;
    if (tensor->owns_data) {
        arix_free(tensor->data, tensor->size * tensor->item_size);
    }
    arix_free(tensor->shape, tensor->ndim * sizeof(size_t));
    arix_free(tensor->strides, tensor->ndim * sizeof(size_t));
    arix_free(tensor, sizeof(ArixTensor));
}

float arix_tensor_get_f32(const ArixTensor* tensor, const size_t* indices) {
    if (tensor->dtype != ARIX_FLOAT32) return 0.0f;
    size_t offset = compute_offset(tensor, indices);
    return ((float*)tensor->data)[offset];
}

void arix_tensor_set_f32(ArixTensor* tensor, const size_t* indices, float value) {
    if (tensor->dtype != ARIX_FLOAT32) return;
    size_t offset = compute_offset(tensor, indices);
    ((float*)tensor->data)[offset] = value;
}

double arix_tensor_get_f64(const ArixTensor* tensor, const size_t* indices) {
    if (tensor->dtype != ARIX_FLOAT64) return 0.0;
    size_t offset = compute_offset(tensor, indices);
    return ((double*)tensor->data)[offset];
}

void arix_tensor_set_f64(ArixTensor* tensor, const size_t* indices, double value) {
    if (tensor->dtype != ARIX_FLOAT64) return;
    size_t offset = compute_offset(tensor, indices);
    ((double*)tensor->data)[offset] = value;
}

int32_t arix_tensor_get_i32(const ArixTensor* tensor, const size_t* indices) {
    if (tensor->dtype != ARIX_INT32) return 0;
    size_t offset = compute_offset(tensor, indices);
    return ((int32_t*)tensor->data)[offset];
}

void arix_tensor_set_i32(ArixTensor* tensor, const size_t* indices, int32_t value) {
    if (tensor->dtype != ARIX_INT32) return;
    size_t offset = compute_offset(tensor, indices);
    ((int32_t*)tensor->data)[offset] = value;
}

int64_t arix_tensor_get_i64(const ArixTensor* tensor, const size_t* indices) {
    if (tensor->dtype != ARIX_INT64) return 0;
    size_t offset = compute_offset(tensor, indices);
    return ((int64_t*)tensor->data)[offset];
}

void arix_tensor_set_i64(ArixTensor* tensor, const size_t* indices, int64_t value) {
    if (tensor->dtype != ARIX_INT64) return;
    size_t offset = compute_offset(tensor, indices);
    ((int64_t*)tensor->data)[offset] = value;
}

uint8_t arix_tensor_get_bool(const ArixTensor* tensor, const size_t* indices) {
    if (tensor->dtype != ARIX_BOOL) return 0;
    size_t offset = compute_offset(tensor, indices);
    return ((uint8_t*)tensor->data)[offset];
}

void arix_tensor_set_bool(ArixTensor* tensor, const size_t* indices, uint8_t value) {
    if (tensor->dtype != ARIX_BOOL) return;
    size_t offset = compute_offset(tensor, indices);
    ((uint8_t*)tensor->data)[offset] = value ? 1 : 0;
}

void arix_tensor_fill_f32(ArixTensor* tensor, float value) {
    if (!tensor || tensor->dtype != ARIX_FLOAT32) return;
    float* data = (float*)tensor->data;
    for (size_t i = 0; i < tensor->size; i++) data[i] = value;
}

void arix_tensor_fill_f64(ArixTensor* tensor, double value) {
    if (!tensor || tensor->dtype != ARIX_FLOAT64) return;
    double* data = (double*)tensor->data;
    for (size_t i = 0; i < tensor->size; i++) data[i] = value;
}

ArixTensor* arix_tensor_empty(const size_t* shape, size_t ndim, ArixDtype dtype) {
    return arix_tensor_create(shape, ndim, dtype);
}

ArixTensor* arix_tensor_zeros(const size_t* shape, size_t ndim, ArixDtype dtype) {
    ArixTensor* tensor = arix_tensor_create(shape, ndim, dtype);
    if (!tensor) return NULL;
    memset(tensor->data, 0, tensor->size * tensor->item_size);
    return tensor;
}

ArixTensor* arix_tensor_ones(const size_t* shape, size_t ndim, ArixDtype dtype) {
    ArixTensor* tensor = arix_tensor_create(shape, ndim, dtype);
    if (!tensor) return NULL;
    size_t n = tensor->size;
    if (dtype == ARIX_FLOAT32 || dtype == ARIX_FLOAT16 || dtype == ARIX_BFLOAT16 || dtype == ARIX_FLOAT8) {
        float* data = (float*)tensor->data;
        for (size_t i = 0; i < n; i++) data[i] = 1.0f;
    } else if (dtype == ARIX_FLOAT64) {
        double* data = (double*)tensor->data;
        for (size_t i = 0; i < n; i++) data[i] = 1.0;
    } else if (dtype == ARIX_INT32 || dtype == ARIX_INT64 || dtype == ARIX_INT16 || dtype == ARIX_INT8 || dtype == ARIX_UINT8 || dtype == ARIX_BOOL) {
        memset(tensor->data, 1, n * tensor->item_size);
    }
    return tensor;
}

ArixTensor* arix_tensor_full(const size_t* shape, size_t ndim, ArixDtype dtype, const void* value) {
    ArixTensor* tensor = arix_tensor_create(shape, ndim, dtype);
    if (!tensor || !value) return tensor;
    size_t item_size = tensor->item_size;
    unsigned char* data = (unsigned char*)tensor->data;
    for (size_t i = 0; i < tensor->size; i++) {
        memcpy(data + i * item_size, value, item_size);
    }
    return tensor;
}

ArixTensor* arix_tensor_arange(float start, float stop, float step, ArixDtype dtype) {
    if (step == 0.0f) return NULL;
    size_t n = (size_t)((stop - start) / step);
    if ((stop - start) / step < 0) return NULL;
    size_t shape[] = {n};
    ArixTensor* tensor = arix_tensor_create(shape, 1, dtype);
    if (!tensor) return NULL;
    if (dtype == ARIX_FLOAT32) {
        float* data = (float*)tensor->data;
        for (size_t i = 0; i < n; i++) data[i] = start + i * step;
    } else if (dtype == ARIX_FLOAT64) {
        double* data = (double*)tensor->data;
        for (size_t i = 0; i < n; i++) data[i] = (double)(start + i * step);
    } else if (dtype == ARIX_INT32) {
        int32_t* data = (int32_t*)tensor->data;
        for (size_t i = 0; i < n; i++) data[i] = (int32_t)(start + i * step);
    } else if (dtype == ARIX_INT64) {
        int64_t* data = (int64_t*)tensor->data;
        for (size_t i = 0; i < n; i++) data[i] = (int64_t)(start + i * step);
    }
    return tensor;
}

ArixTensor* arix_tensor_linspace(float start, float stop, size_t steps, ArixDtype dtype) {
    if (steps == 0) return NULL;
    size_t shape[] = {steps};
    ArixTensor* tensor = arix_tensor_create(shape, 1, dtype);
    if (!tensor) return NULL;
    float step = (steps > 1) ? (stop - start) / (float)(steps - 1) : 0.0f;
    if (dtype == ARIX_FLOAT32) {
        float* data = (float*)tensor->data;
        for (size_t i = 0; i < steps; i++) data[i] = start + i * step;
    } else if (dtype == ARIX_FLOAT64) {
        double* data = (double*)tensor->data;
        for (size_t i = 0; i < steps; i++) data[i] = (double)(start + i * step);
    }
    return tensor;
}

ArixTensor* arix_tensor_eye(size_t n, ArixDtype dtype) {
    size_t shape[] = {n, n};
    ArixTensor* tensor = arix_tensor_zeros(shape, 2, dtype);
    if (!tensor) return NULL;
    float* data = (float*)tensor->data;
    for (size_t i = 0; i < n; i++) {
        data[i * n + i] = 1.0f;
    }
    return tensor;
}

static unsigned long lcg_state = 123456789;

static float uniform_01(void) {
    lcg_state = lcg_state * 1103515245UL + 12345UL;
    return (float)((lcg_state >> 16) & 0x7FFF) / 32767.0f;
}

ArixTensor* arix_tensor_randn(const size_t* shape, size_t ndim, ArixDtype dtype) {
    ArixTensor* tensor = arix_tensor_create(shape, ndim, dtype);
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

ArixTensor* arix_tensor_copy(const ArixTensor* src) {
    if (!src) return NULL;
    ArixTensor* dst = arix_tensor_create(src->shape, src->ndim, src->dtype);
    if (!dst) return NULL;
    memcpy(dst->data, src->data, src->size * src->item_size);
    return dst;
}

ArixTensor* arix_tensor_clone(const ArixTensor* src) {
    return arix_tensor_copy(src);
}

ArixTensor* arix_tensor_slice(const ArixTensor* src, size_t dim, size_t start, size_t end) {
    if (!src || dim >= src->ndim || start >= end || end > src->shape[dim]) return NULL;
    size_t new_ndim = src->ndim;
    size_t* new_shape = (size_t*)aligned_alloc_wrapper(new_ndim * sizeof(size_t), 64);
    if (!new_shape) return NULL;
    size_t new_size = 1;
    for (size_t i = 0; i < new_ndim; i++) {
        new_shape[i] = (i == dim) ? (end - start) : src->shape[i];
        new_size *= new_shape[i];
    }
    ArixTensor* result = arix_tensor_empty(new_shape, new_ndim, src->dtype);
    arix_free(new_shape, new_ndim * sizeof(size_t));
    if (!result) return NULL;
    size_t* src_indices = (size_t*)aligned_alloc_wrapper(src->ndim * sizeof(size_t), 64);
    if (!src_indices) { arix_tensor_destroy(result); return NULL; }
    memset(src_indices, 0, src->ndim * sizeof(size_t));
    unsigned char* src_data = (unsigned char*)src->data;
    unsigned char* dst_data = (unsigned char*)result->data;
    size_t item_size = src->item_size;
    size_t dst_idx = 0;
    for (size_t flat = 0; flat < src->size; flat++) {
        size_t tmp = flat;
        for (size_t i = 0; i < src->ndim; i++) {
            size_t dim_stride = src->strides[i];
            size_t dim_size = src->shape[i];
            src_indices[i] = (dim_stride > 0) ? (tmp / dim_stride) % dim_size : 0;
            if (dim_stride > 0) tmp = tmp % dim_stride;
        }
        if (src_indices[dim] >= start && src_indices[dim] < end) {
            size_t offset = compute_offset(src, src_indices);
            memcpy(dst_data + dst_idx * item_size, src_data + offset * item_size, item_size);
            dst_idx++;
        }
    }
    arix_free(src_indices, src->ndim * sizeof(size_t));
    return result;
}

ArixTensor* arix_tensor_reshape(const ArixTensor* src, const size_t* new_shape, size_t new_ndim) {
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
    ArixTensor* result = arix_tensor_empty(resolved_shape, new_ndim, src->dtype);
    if (!result) return NULL;
    memcpy(result->data, src->data, src->size * src->item_size);
    return result;
}

ArixTensor* arix_tensor_permute(const ArixTensor* src, const size_t* axes) {
    if (!src) return NULL;
    size_t* new_shape = (size_t*)aligned_alloc_wrapper(src->ndim * sizeof(size_t), 64);
    size_t* new_strides = (size_t*)aligned_alloc_wrapper(src->ndim * sizeof(size_t), 64);
    if (!new_shape || !new_strides) {
        arix_free(new_shape, src->ndim * sizeof(size_t));
        arix_free(new_strides, src->ndim * sizeof(size_t));
        return NULL;
    }
    for (size_t i = 0; i < src->ndim; i++) {
        new_shape[i] = src->shape[axes[i]];
        new_strides[i] = src->strides[axes[i]];
    }
    ArixTensor* result = arix_tensor_empty(new_shape, src->ndim, src->dtype);
    if (!result) { arix_free(new_shape, src->ndim * sizeof(size_t)); arix_free(new_strides, src->ndim * sizeof(size_t)); return NULL; }
    for (size_t i = 0; i < src->ndim; i++) {
        result->strides[i] = new_strides[i];
    }
    memcpy(result->data, src->data, src->size * src->item_size);
    arix_free(new_shape, src->ndim * sizeof(size_t));
    arix_free(new_strides, src->ndim * sizeof(size_t));
    return result;
}

ArixTensor* arix_tensor_expand(const ArixTensor* src, const size_t* new_shape, size_t new_ndim) {
    if (!src || new_ndim < src->ndim) return NULL;
    ArixTensor* result = arix_tensor_empty(new_shape, new_ndim, src->dtype);
    if (!result) return NULL;
    size_t* src_indices = (size_t*)aligned_alloc_wrapper(src->ndim * sizeof(size_t), 64);
    size_t* dst_indices = (size_t*)aligned_alloc_wrapper(new_ndim * sizeof(size_t), 64);
    if (!src_indices || !dst_indices) { arix_free(src_indices, src->ndim * sizeof(size_t)); arix_free(dst_indices, new_ndim * sizeof(size_t)); arix_tensor_destroy(result); return NULL; }
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
    arix_free(src_indices, src->ndim * sizeof(size_t));
    arix_free(dst_indices, new_ndim * sizeof(size_t));
    return result;
}

ArixTensor* arix_tensor_squeeze(const ArixTensor* src, size_t dim) {
    if (!src || dim >= src->ndim || src->shape[dim] != 1) return arix_tensor_copy(src);
    size_t new_ndim = src->ndim - 1;
    size_t new_shape[16];
    size_t j = 0;
    for (size_t i = 0; i < src->ndim; i++) {
        if (i != dim) { new_shape[j++] = src->shape[i]; }
    }
    return arix_tensor_reshape(src, new_shape, new_ndim);
}

ArixTensor* arix_tensor_unsqueeze(const ArixTensor* src, size_t dim) {
    if (!src || dim > src->ndim) return NULL;
    size_t new_ndim = src->ndim + 1;
    size_t new_shape[16];
    size_t j = 0;
    for (size_t i = 0; i < new_ndim; i++) {
        new_shape[i] = (i == dim) ? 1 : src->shape[j++];
    }
    return arix_tensor_reshape(src, new_shape, new_ndim);
}

ArixTensor* arix_tensor_concat(const ArixTensor** tensors, size_t num_tensors, size_t dim) {
    if (!tensors || num_tensors == 0) return NULL;
    const ArixTensor* first = tensors[0];
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
    ArixTensor* result = arix_tensor_empty(new_shape, first->ndim, first->dtype);
    arix_free(new_shape, first->ndim * sizeof(size_t));
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

ArixTensor** arix_tensor_split(const ArixTensor* src, size_t num_splits, size_t dim) {
    if (!src || num_splits == 0 || dim >= src->ndim || src->shape[dim] % num_splits != 0) return NULL;
    size_t split_size = src->shape[dim] / num_splits;
    ArixTensor** results = (ArixTensor**)aligned_alloc_wrapper(num_splits * sizeof(ArixTensor*), 64);
    if (!results) return NULL;
    memset(results, 0, num_splits * sizeof(ArixTensor*));
    for (size_t s = 0; s < num_splits; s++) {
        results[s] = arix_tensor_slice(src, dim, s * split_size, (s + 1) * split_size);
        if (!results[s]) {
            for (size_t k = 0; k < s; k++) arix_tensor_destroy(results[k]);
            arix_free(results, num_splits * sizeof(ArixTensor*));
            return NULL;
        }
    }
    return results;
}

ArixTensor* arix_tensor_tile(const ArixTensor* src, const size_t* reps, size_t reps_ndim) {
    if (!src || !reps) return NULL;
    size_t out_ndim = (src->ndim > reps_ndim) ? src->ndim : reps_ndim;
    size_t out_shape[16];
    size_t offset = out_ndim - src->ndim;
    for (size_t i = 0; i < out_ndim; i++) {
        size_t s = (i < offset) ? 1 : src->shape[i - offset];
        size_t r = (i < reps_ndim) ? reps[i] : 1;
        out_shape[i] = s * r;
    }
    ArixTensor* result = arix_tensor_empty(out_shape, out_ndim, src->dtype);
    if (!result) return NULL;
    unsigned char* src_data = (unsigned char*)src->data;
    unsigned char* dst_data = (unsigned char*)result->data;
    size_t is = src->item_size;
    size_t* indices = (size_t*)aligned_alloc_wrapper(out_ndim * sizeof(size_t), 64);
    if (!indices) { arix_tensor_destroy(result); return NULL; }
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
    arix_free(indices, out_ndim * sizeof(size_t));
    return result;
}

ArixTensor* arix_tensor_repeat(const ArixTensor* src, size_t repeats, size_t dim) {
    if (!src || dim >= src->ndim) return NULL;
    size_t* reps = (size_t*)aligned_alloc_wrapper(src->ndim * sizeof(size_t), 64);
    if (!reps) return NULL;
    for (size_t i = 0; i < src->ndim; i++) reps[i] = (i == dim) ? repeats : 1;
    ArixTensor* result = arix_tensor_tile(src, reps, src->ndim);
    arix_free(reps, src->ndim * sizeof(size_t));
    return result;
}

ArixTensor* arix_tensor_gather(const ArixTensor* src, size_t dim, const ArixTensor* indices) {
    if (!src || !indices || dim >= src->ndim || indices->ndim != src->ndim) return NULL;
    ArixTensor* result = arix_tensor_empty(indices->shape, indices->ndim, src->dtype);
    if (!result) return NULL;
    unsigned char* src_data = (unsigned char*)src->data;
    unsigned char* dst_data = (unsigned char*)result->data;
    size_t is = src->item_size;
    size_t* idx = (size_t*)aligned_alloc_wrapper(src->ndim * sizeof(size_t), 64);
    if (!idx) { arix_tensor_destroy(result); return NULL; }
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
    arix_free(idx, src->ndim * sizeof(size_t));
    return result;
}

ArixTensor* arix_tensor_scatter(ArixTensor* dest, size_t dim, const ArixTensor* indices, const ArixTensor* src) {
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
    arix_free(idx, dest->ndim * sizeof(size_t));
    return dest;
}

ArixTensor* arix_tensor_masked_select(const ArixTensor* src, const ArixTensor* mask) {
    if (!src || !mask || src->size != mask->size) return NULL;
    unsigned char* mask_data = (unsigned char*)mask->data;
    size_t count = 0;
    for (size_t i = 0; i < mask->size; i++) {
        if (mask_data[i]) count++;
    }
    size_t shape[] = {count};
    ArixTensor* result = arix_tensor_empty(shape, 1, src->dtype);
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

ArixTensor* arix_tensor_masked_fill(ArixTensor* src, const ArixTensor* mask, const void* value) {
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

ArixTensor* arix_tensor_where(const ArixTensor* condition, const ArixTensor* x, const ArixTensor* y) {
    if (!condition || !x || !y) return NULL;
    size_t n = x->size < y->size ? x->size : y->size;
    if (condition->size < n) n = condition->size;
    ArixTensor* result = arix_tensor_copy(x);
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

ArixTensor* arix_tensor_cast(const ArixTensor* src, ArixDtype dtype) {
    if (!src || dtype == src->dtype) return arix_tensor_copy(src);
    ArixTensor* result = arix_tensor_empty(src->shape, src->ndim, dtype);
    if (!result) return NULL;
    size_t n = src->size;
    if (src->dtype == ARIX_FLOAT32 && dtype == ARIX_FLOAT64) {
        float* s = (float*)src->data; double* d = (double*)result->data;
        for (size_t i = 0; i < n; i++) d[i] = (double)s[i];
    } else if (src->dtype == ARIX_FLOAT64 && dtype == ARIX_FLOAT32) {
        double* s = (double*)src->data; float* d = (float*)result->data;
        for (size_t i = 0; i < n; i++) d[i] = (float)s[i];
    } else if (src->dtype == ARIX_FLOAT32 && dtype == ARIX_INT32) {
        float* s = (float*)src->data; int32_t* d = (int32_t*)result->data;
        for (size_t i = 0; i < n; i++) d[i] = (int32_t)s[i];
    } else if (src->dtype == ARIX_INT32 && dtype == ARIX_FLOAT32) {
        int32_t* s = (int32_t*)src->data; float* d = (float*)result->data;
        for (size_t i = 0; i < n; i++) d[i] = (float)s[i];
    } else if (dtype == ARIX_FLOAT32) {
        float* d = (float*)result->data;
        for (size_t i = 0; i < n; i++) {
            d[i] = arix_tensor_get_f32(src, (size_t[]){i});
        }
    } else {
        memcpy(result->data, src->data, n * result->item_size);
    }
    return result;
}

ArixTensor* arix_tensor_to_device(const ArixTensor* src, ArixDevice device) {
    (void)device;
    if (!src) return NULL;
    ArixTensor* dst = arix_tensor_copy(src);
    if (!dst) return NULL;
    dst->device = device;
    return dst;
}

ArixTensor* arix_tensor_to_layout(const ArixTensor* src, ArixLayout layout) {
    if (!src) return NULL;
    if (src->layout == layout) return arix_tensor_copy(src);
    ArixTensor* dst = arix_tensor_copy(src);
    if (!dst) return NULL;
    if (layout == ARIX_LAYOUT_COL_MAJOR && src->ndim == 2) {
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

static ArixTensor* compare_op(const ArixTensor* a, const ArixTensor* b, int (*cmp)(float, float)) {
    if (!a || !b) return NULL;
    size_t n = a->size < b->size ? a->size : b->size;
    ArixTensor* result = arix_tensor_empty(a->shape, a->ndim, ARIX_BOOL);
    if (!result) return NULL;
    float* ad = (float*)a->data;
    float* bd = (float*)b->data;
    uint8_t* rd = (uint8_t*)result->data;
    for (size_t i = 0; i < n; i++) rd[i] = cmp(ad[i], bd[i]) ? 1 : 0;
    return result;
}

static int cmp_eq(float a, float b) { return fabsf(a - b) < 1e-6f; }
static int cmp_ne(float a, float b) { return fabsf(a - b) >= 1e-6f; }
static int cmp_lt(float a, float b) { return a < b; }
static int cmp_le(float a, float b) { return a <= b; }
static int cmp_gt(float a, float b) { return a > b; }
static int cmp_ge(float a, float b) { return a >= b; }

ArixTensor* arix_tensor_eq(const ArixTensor* a, const ArixTensor* b) { return compare_op(a, b, cmp_eq); }
ArixTensor* arix_tensor_ne(const ArixTensor* a, const ArixTensor* b) { return compare_op(a, b, cmp_ne); }
ArixTensor* arix_tensor_lt(const ArixTensor* a, const ArixTensor* b) { return compare_op(a, b, cmp_lt); }
ArixTensor* arix_tensor_le(const ArixTensor* a, const ArixTensor* b) { return compare_op(a, b, cmp_le); }
ArixTensor* arix_tensor_gt(const ArixTensor* a, const ArixTensor* b) { return compare_op(a, b, cmp_gt); }
ArixTensor* arix_tensor_ge(const ArixTensor* a, const ArixTensor* b) { return compare_op(a, b, cmp_ge); }

ArixTensor* arix_tensor_add(const ArixTensor* a, const ArixTensor* b) {
    if (!a || !b) return NULL;
    size_t sz = a->size < b->size ? a->size : b->size;
    ArixTensor* result = arix_tensor_create(a->shape, a->ndim, ARIX_FLOAT32);
    if (!result) return NULL;
    float* rd = (float*)result->data;
    float* ad = (float*)a->data;
    float* bd = (float*)b->data;
    if (a->size == b->size) {
        for (size_t i = 0; i < sz; i++) rd[i] = ad[i] + bd[i];
    } else {
        size_t last = a->shape[a->ndim - 1];
        for (size_t i = 0; i < a->size; i++)
            rd[i] = ad[i] + bd[i % last];
    }
    return result;
}

ArixTensor* arix_tensor_sub(const ArixTensor* a, const ArixTensor* b) {
    if (!a || !b) return NULL;
    size_t sz = a->size < b->size ? a->size : b->size;
    ArixTensor* result = arix_tensor_create(a->shape, a->ndim, ARIX_FLOAT32);
    if (!result) return NULL;
    float* rd = (float*)result->data;
    float* ad = (float*)a->data;
    float* bd = (float*)b->data;
    for (size_t i = 0; i < sz; i++) rd[i] = ad[i] - bd[i];
    return result;
}

ArixTensor* arix_tensor_mul(const ArixTensor* a, const ArixTensor* b) {
    if (!a || !b) return NULL;
    size_t sz = a->size < b->size ? a->size : b->size;
    ArixTensor* result = arix_tensor_create(a->shape, a->ndim, ARIX_FLOAT32);
    if (!result) return NULL;
    float* rd = (float*)result->data;
    float* ad = (float*)a->data;
    float* bd = (float*)b->data;
    for (size_t i = 0; i < sz; i++) rd[i] = ad[i] * bd[i];
    return result;
}

ArixTensor* arix_tensor_div(const ArixTensor* a, const ArixTensor* b) {
    if (!a || !b) return NULL;
    size_t sz = a->size < b->size ? a->size : b->size;
    ArixTensor* result = arix_tensor_create(a->shape, a->ndim, ARIX_FLOAT32);
    if (!result) return NULL;
    float* rd = (float*)result->data;
    float* ad = (float*)a->data;
    float* bd = (float*)b->data;
    for (size_t i = 0; i < sz; i++) rd[i] = ad[i] / (bd[i] + 1e-10f);
    return result;
}

ArixTensor* arix_tensor_pow(const ArixTensor* a, const ArixTensor* b) {
    if (!a || !b) return NULL;
    size_t sz = a->size < b->size ? a->size : b->size;
    ArixTensor* result = arix_tensor_create(a->shape, a->ndim, ARIX_FLOAT32);
    if (!result) return NULL;
    float* rd = (float*)result->data;
    float* ad = (float*)a->data;
    float* bd = (float*)b->data;
    for (size_t i = 0; i < sz; i++) rd[i] = powf(ad[i], bd[i]);
    return result;
}

static ArixTensor* unary_op_f32(const ArixTensor* src, float (*op)(float)) {
    if (!src) return NULL;
    ArixTensor* result = arix_tensor_create(src->shape, src->ndim, ARIX_FLOAT32);
    if (!result) return NULL;
    float* sd = (float*)src->data;
    float* rd = (float*)result->data;
    for (size_t i = 0; i < src->size; i++) rd[i] = op(sd[i]);
    return result;
}

static float negate_f32(float x) { return -x; }
static float sign_f32(float x) { return (x > 0) ? 1.0f : (x < 0) ? -1.0f : 0.0f; }

ArixTensor* arix_tensor_neg(const ArixTensor* src) { return unary_op_f32(src, negate_f32); }
ArixTensor* arix_tensor_abs(const ArixTensor* src) { return unary_op_f32(src, fabsf); }
ArixTensor* arix_tensor_sign(const ArixTensor* src) { return unary_op_f32(src, sign_f32); }
ArixTensor* arix_tensor_floor(const ArixTensor* src) { return unary_op_f32(src, floorf); }
ArixTensor* arix_tensor_ceil(const ArixTensor* src) { return unary_op_f32(src, ceilf); }
ArixTensor* arix_tensor_round(const ArixTensor* src) { return unary_op_f32(src, roundf); }
ArixTensor* arix_tensor_trunc(const ArixTensor* src) { return unary_op_f32(src, truncf); }
ArixTensor* arix_tensor_exp(const ArixTensor* src) { return unary_op_f32(src, expf); }
ArixTensor* arix_tensor_log(const ArixTensor* src) { return unary_op_f32(src, logf); }
ArixTensor* arix_tensor_sqrt(const ArixTensor* src) { return unary_op_f32(src, sqrtf); }
ArixTensor* arix_tensor_sin(const ArixTensor* src) { return unary_op_f32(src, sinf); }
ArixTensor* arix_tensor_cos(const ArixTensor* src) { return unary_op_f32(src, cosf); }
ArixTensor* arix_tensor_tan(const ArixTensor* src) { return unary_op_f32(src, tanf); }
ArixTensor* arix_tensor_asin(const ArixTensor* src) { return unary_op_f32(src, asinf); }
ArixTensor* arix_tensor_acos(const ArixTensor* src) { return unary_op_f32(src, acosf); }
ArixTensor* arix_tensor_atan(const ArixTensor* src) { return unary_op_f32(src, atanf); }
ArixTensor* arix_tensor_sinh(const ArixTensor* src) { return unary_op_f32(src, sinhf); }
ArixTensor* arix_tensor_cosh(const ArixTensor* src) { return unary_op_f32(src, coshf); }
ArixTensor* arix_tensor_tanh(const ArixTensor* src) { return unary_op_f32(src, tanhf); }

ArixTensor* arix_tensor_sum(const ArixTensor* src, size_t dim) {
    if (!src || dim >= src->ndim) return NULL;
    size_t out_ndim = src->ndim;
    size_t out_shape[16];
    for (size_t i = 0; i < src->ndim; i++) out_shape[i] = (i == dim) ? 1 : src->shape[i];
    ArixTensor* result = arix_tensor_zeros(out_shape, out_ndim, ARIX_FLOAT32);
    if (!result) return NULL;
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
    return result;
}

ArixTensor* arix_tensor_mean(const ArixTensor* src, size_t dim) {
    if (!src || dim >= src->ndim) return NULL;
    ArixTensor* sum = arix_tensor_sum(src, dim);
    if (!sum) return NULL;
    float* rd = (float*)sum->data;
    float inv_n = 1.0f / (float)src->shape[dim];
    for (size_t i = 0; i < sum->size; i++) rd[i] *= inv_n;
    return sum;
}

ArixTensor* arix_tensor_var(const ArixTensor* src, size_t dim) {
    if (!src || dim >= src->ndim) return NULL;
    ArixTensor* mean = arix_tensor_mean(src, dim);
    if (!mean) return NULL;
    size_t inner = 1, outer = 1, reduce = src->shape[dim];
    for (size_t i = 0; i < dim; i++) outer *= src->shape[i];
    for (size_t i = dim + 1; i < src->ndim; i++) inner *= src->shape[i];
    ArixTensor* result = arix_tensor_zeros(mean->shape, mean->ndim, ARIX_FLOAT32);
    if (!result) { arix_tensor_destroy(mean); return NULL; }
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
    arix_tensor_destroy(mean);
    return result;
}

ArixTensor* arix_tensor_std(const ArixTensor* src, size_t dim) {
    ArixTensor* v = arix_tensor_var(src, dim);
    if (!v) return NULL;
    float* d = (float*)v->data;
    for (size_t i = 0; i < v->size; i++) d[i] = sqrtf(d[i]);
    return v;
}

float arix_tensor_min(const ArixTensor* src) {
    if (!src || src->size == 0) return 0.0f;
    float* d = (float*)src->data;
    float val = d[0];
    for (size_t i = 1; i < src->size; i++) if (d[i] < val) val = d[i];
    return val;
}

float arix_tensor_max(const ArixTensor* src) {
    if (!src || src->size == 0) return 0.0f;
    float* d = (float*)src->data;
    float val = d[0];
    for (size_t i = 1; i < src->size; i++) if (d[i] > val) val = d[i];
    return val;
}

size_t arix_tensor_argmin(const ArixTensor* src) {
    if (!src || src->size == 0) return 0;
    float* d = (float*)src->data;
    size_t idx = 0;
    for (size_t i = 1; i < src->size; i++) if (d[i] < d[idx]) idx = i;
    return idx;
}

size_t arix_tensor_argmax(const ArixTensor* src) {
    if (!src || src->size == 0) return 0;
    float* d = (float*)src->data;
    size_t idx = 0;
    for (size_t i = 1; i < src->size; i++) if (d[i] > d[idx]) idx = i;
    return idx;
}

ArixTensor* arix_tensor_cumsum(const ArixTensor* src, size_t dim) {
    if (!src || dim >= src->ndim) return NULL;
    ArixTensor* result = arix_tensor_copy(src);
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

ArixTensor* arix_tensor_cumprod(const ArixTensor* src, size_t dim) {
    if (!src || dim >= src->ndim) return NULL;
    ArixTensor* result = arix_tensor_copy(src);
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

float arix_tensor_dot(const ArixTensor* a, const ArixTensor* b) {
    if (!a || !b || a->size != b->size) return 0.0f;
    float* ad = (float*)a->data;
    float* bd = (float*)b->data;
    float sum = 0.0f;
    for (size_t i = 0; i < a->size; i++) sum += ad[i] * bd[i];
    return sum;
}

ArixTensor* arix_tensor_matmul(const ArixTensor* a, const ArixTensor* b) {
    if (!a || !b || a->ndim != 2 || b->ndim != 2) return NULL;
    size_t m = a->shape[0], k = a->shape[1], n = b->shape[1];
    if (k != b->shape[0]) return NULL;
    size_t shape_c[] = {m, n};
    ArixTensor* result = arix_tensor_zeros(shape_c, 2, ARIX_FLOAT32);
    if (!result) return NULL;
    float* ad = (float*)a->data;
    float* bd = (float*)b->data;
    float* rd = (float*)result->data;
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++)
            for (size_t l = 0; l < k; l++)
                rd[i * n + j] += ad[i * k + l] * bd[l * n + j];
    return result;
}

ArixTensor* arix_tensor_transpose(const ArixTensor* src, size_t dim1, size_t dim2) {
    if (!src || dim1 >= src->ndim || dim2 >= src->ndim) return NULL;
    size_t* axes = (size_t*)aligned_alloc_wrapper(src->ndim * sizeof(size_t), 64);
    if (!axes) return NULL;
    for (size_t i = 0; i < src->ndim; i++) axes[i] = i;
    axes[dim1] = dim2;
    axes[dim2] = dim1;
    ArixTensor* result = arix_tensor_permute(src, axes);
    arix_free(axes, src->ndim * sizeof(size_t));
    return result;
}

ArixTensor* arix_tensor_inverse(const ArixTensor* src) {
    (void)src;
    return NULL;
}

float arix_tensor_det(const ArixTensor* src) {
    (void)src;
    return 0.0f;
}

ArixTensor* arix_tensor_conv1d(const ArixTensor* input, const ArixTensor* kernel, size_t stride, size_t padding) {
    (void)input; (void)kernel; (void)stride; (void)padding;
    return NULL;
}

ArixTensor* arix_tensor_conv2d(const ArixTensor* input, const ArixTensor* kernel, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w) {
    (void)input; (void)kernel; (void)stride_h; (void)stride_w; (void)pad_h; (void)pad_w;
    return NULL;
}

ArixTensor* arix_tensor_pool1d(const ArixTensor* src, size_t kernel_size, size_t stride) {
    (void)src; (void)kernel_size; (void)stride;
    return NULL;
}

ArixTensor* arix_tensor_pool2d(const ArixTensor* src, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w) {
    (void)src; (void)kernel_h; (void)kernel_w; (void)stride_h; (void)stride_w;
    return NULL;
}

ArixTensor* arix_tensor_softmax(const ArixTensor* src, size_t dim) {
    if (!src || dim >= src->ndim) return NULL;
    ArixTensor* result = arix_tensor_copy(src);
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

ArixTensor* arix_tensor_log_softmax(const ArixTensor* src, size_t dim) {
    if (!src || dim >= src->ndim) return NULL;
    ArixTensor* sm = arix_tensor_softmax(src, dim);
    if (!sm) return NULL;
    float* d = (float*)sm->data;
    for (size_t i = 0; i < sm->size; i++) d[i] = logf(d[i] + 1e-10f);
    return sm;
}

ArixTensor* arix_tensor_relu(const ArixTensor* src) {
    if (!src) return NULL;
    ArixTensor* result = arix_tensor_copy(src);
    if (!result) return NULL;
    float* rd = (float*)result->data;
    for (size_t i = 0; i < result->size; i++) if (rd[i] < 0) rd[i] = 0.0f;
    return result;
}

ArixTensor* arix_tensor_gelu(const ArixTensor* src) {
    if (!src) return NULL;
    ArixTensor* result = arix_tensor_copy(src);
    if (!result) return NULL;
    float* rd = (float*)result->data;
    for (size_t i = 0; i < result->size; i++) {
        float x = rd[i];
        rd[i] = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    }
    return result;
}

ArixTensor* arix_tensor_silu(const ArixTensor* src) {
    if (!src) return NULL;
    ArixTensor* result = arix_tensor_copy(src);
    if (!result) return NULL;
    float* rd = (float*)result->data;
    for (size_t i = 0; i < result->size; i++) {
        float x = rd[i];
        rd[i] = x / (1.0f + expf(-x));
    }
    return result;
}

ArixTensor* arix_tensor_sigmoid(const ArixTensor* src) {
    if (!src) return NULL;
    ArixTensor* result = arix_tensor_copy(src);
    if (!result) return NULL;
    float* rd = (float*)result->data;
    for (size_t i = 0; i < result->size; i++) {
        rd[i] = 1.0f / (1.0f + expf(-rd[i]));
    }
    return result;
}

ArixTensor* arix_tensor_dropout(const ArixTensor* src, float rate, unsigned int seed) {
    (void)seed;
    if (!src) return NULL;
    ArixTensor* result = arix_tensor_copy(src);
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

ArixTensor* arix_tensor_layer_norm(const ArixTensor* src, const ArixTensor* gamma, const ArixTensor* beta, float eps) {
    if (!src) return NULL;
    ArixTensor* result = arix_tensor_copy(src);
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

ArixTensor* arix_tensor_batch_norm(const ArixTensor* src, const ArixTensor* gamma, const ArixTensor* beta, const ArixTensor* running_mean, const ArixTensor* running_var, float eps) {
    if (!src) return NULL;
    ArixTensor* result = arix_tensor_copy(src);
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

ArixTensor* arix_tensor_group_norm(const ArixTensor* src, const ArixTensor* gamma, const ArixTensor* beta, size_t num_groups, float eps) {
    if (!src) return NULL;
    ArixTensor* result = arix_tensor_copy(src);
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

ArixTensor* arix_tensor_instance_norm(const ArixTensor* src, const ArixTensor* gamma, const ArixTensor* beta, float eps) {
    if (!src) return NULL;
    ArixTensor* result = arix_tensor_copy(src);
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

ArixTensor* arix_tensor_embedding(const ArixTensor* weight, const ArixTensor* indices) {
    if (!weight || !indices || weight->ndim != 2) return NULL;
    size_t num_embeddings = weight->shape[0];
    size_t embedding_dim = weight->shape[1];
    size_t* idx_data = (size_t*)indices->data;
    size_t n = indices->size;
    size_t out_shape[] = {n, embedding_dim};
    ArixTensor* result = arix_tensor_empty(out_shape, 2, weight->dtype);
    if (!result) return NULL;
    unsigned char* wd = (unsigned char*)weight->data;
    unsigned char* rd = (unsigned char*)result->data;
    size_t is = weight->item_size;
    for (size_t i = 0; i < n; i++) {
        size_t emb_idx = idx_data[i];
        if (emb_idx >= num_embeddings) emb_idx = 0;
        memcpy(rd + i * embedding_dim * is, wd + emb_idx * embedding_dim * is, embedding_dim * is);
    }
    return result;
}

ArixTensor* arix_tensor_cross_entropy(const ArixTensor* pred, const ArixTensor* target) {
    if (!pred || !target) return NULL;
    size_t n = pred->size / pred->shape[pred->ndim - 1];
    size_t c = pred->shape[pred->ndim - 1];
    size_t shape[] = {1};
    ArixTensor* result = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
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

ArixTensor* arix_tensor_mse_loss(const ArixTensor* pred, const ArixTensor* target) {
    if (!pred || !target || pred->size != target->size) return NULL;
    size_t shape[] = {1};
    ArixTensor* result = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    if (!result) return NULL;
    float* pd = (float*)pred->data;
    float* td = (float*)target->data;
    float* rd = (float*)result->data;
    float sum = 0.0f;
    for (size_t i = 0; i < pred->size; i++) { float d = pd[i] - td[i]; sum += d * d; }
    rd[0] = sum / (float)pred->size;
    return result;
}

ArixTensor* arix_tensor_mae_loss(const ArixTensor* pred, const ArixTensor* target) {
    if (!pred || !target || pred->size != target->size) return NULL;
    size_t shape[] = {1};
    ArixTensor* result = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
    if (!result) return NULL;
    float* pd = (float*)pred->data;
    float* td = (float*)target->data;
    float* rd = (float*)result->data;
    float sum = 0.0f;
    for (size_t i = 0; i < pred->size; i++) sum += fabsf(pd[i] - td[i]);
    rd[0] = sum / (float)pred->size;
    return result;
}

ArixTensor* arix_tensor_nll_loss(const ArixTensor* pred, const ArixTensor* target) {
    if (!pred || !target) return NULL;
    size_t n = pred->size / pred->shape[pred->ndim - 1];
    size_t c = pred->shape[pred->ndim - 1];
    size_t shape[] = {1};
    ArixTensor* result = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
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

ArixTensor* arix_tensor_kl_div(const ArixTensor* pred, const ArixTensor* target) {
    if (!pred || !target || pred->size != target->size) return NULL;
    size_t shape[] = {1};
    ArixTensor* result = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
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

ArixTensor* arix_tensor_binary_cross_entropy(const ArixTensor* pred, const ArixTensor* target) {
    if (!pred || !target || pred->size != target->size) return NULL;
    size_t shape[] = {1};
    ArixTensor* result = arix_tensor_zeros(shape, 1, ARIX_FLOAT32);
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

void arix_tensor_print(const ArixTensor* tensor) {
    if (!tensor) return;
    printf("Tensor shape: [");
    for (size_t i = 0; i < tensor->ndim; i++) {
        printf("%zu", tensor->shape[i]);
        if (i < tensor->ndim - 1) printf(", ");
    }
    printf("]\n");
    printf("Dtype: %s\n", arix_tensor_dtype_name(tensor->dtype));
    printf("Size: %zu elements\n", tensor->size);
    printf("Device: %s\n", tensor->device == ARIX_DEVICE_CPU ? "cpu" : "cuda");
    if (tensor->dtype == ARIX_FLOAT32) {
        size_t n = tensor->size < 20 ? tensor->size : 20;
        printf("Data (first %zu): [", n);
        float* data = (float*)tensor->data;
        for (size_t i = 0; i < n; i++) {
            printf("%f", data[i]);
            if (i < n - 1) printf(", ");
        }
        printf("]\n");
    } else if (tensor->dtype == ARIX_INT32) {
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

size_t arix_tensor_dtype_size(ArixDtype dtype) {
    switch (dtype) {
        case ARIX_FLOAT8:    return 1;
        case ARIX_FLOAT16:   return 2;
        case ARIX_BFLOAT16:  return 2;
        case ARIX_FLOAT32:   return sizeof(float);
        case ARIX_FLOAT64:   return sizeof(double);
        case ARIX_INT8:      return 1;
        case ARIX_INT16:     return 2;
        case ARIX_INT32:     return sizeof(int32_t);
        case ARIX_INT64:     return sizeof(int64_t);
        case ARIX_UINT8:     return 1;
        case ARIX_BOOL:      return 1;
        case ARIX_COMPLEX64: return 8;
        case ARIX_COMPLEX128: return 16;
    }
    return 0;
}

size_t arix_tensor_numel(const ArixTensor* tensor) {
    if (!tensor) return 0;
    return tensor->size;
}

int arix_tensor_is_contiguous(const ArixTensor* tensor) {
    if (!tensor || tensor->ndim == 0) return 1;
    size_t expected = 1;
    for (size_t i = tensor->ndim; i > 0; i--) {
        if (tensor->strides[i - 1] != expected) return 0;
        expected *= tensor->shape[i - 1];
    }
    return 1;
}

const char* arix_tensor_dtype_name(ArixDtype dtype) {
    switch (dtype) {
        case ARIX_FLOAT8:    return "float8";
        case ARIX_FLOAT16:   return "float16";
        case ARIX_BFLOAT16:  return "bfloat16";
        case ARIX_FLOAT32:   return "float32";
        case ARIX_FLOAT64:   return "float64";
        case ARIX_INT8:      return "int8";
        case ARIX_INT16:     return "int16";
        case ARIX_INT32:     return "int32";
        case ARIX_INT64:     return "int64";
        case ARIX_UINT8:     return "uint8";
        case ARIX_BOOL:      return "bool";
        case ARIX_COMPLEX64: return "complex64";
        case ARIX_COMPLEX128:return "complex128";
    }
    return "unknown";
}
