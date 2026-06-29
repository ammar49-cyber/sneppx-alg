#ifndef ARIX_TENSOR_INTERNAL_H
#define ARIX_TENSOR_INTERNAL_H
/*
 * Tensor Operations Internal — v0.5
 *
 * PURPOSE: Low-level tensor operation kernels: strided memory copy,
 * broadcasting logic, reduction kernels, and type conversion routines.
 * These are called by the public arix_tensor_* API but are factored
 * out for reuse by the autodiff and driver layers.
 *
 * DEPENDENCIES: arix_tensor.h, arix_thread.h
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ArixTensor ArixTensor;

/* ---------- Strided memory ---------- */
int arix_tensor_strided_copy(void* dst, const void* src,
                              const size_t* dst_strides, const size_t* src_strides,
                              const size_t* shape, size_t ndim, size_t elem_size);

int arix_tensor_broadcast_strides(const size_t* src_shape, size_t src_ndim,
                                  const size_t* dst_shape, size_t dst_ndim,
                                  size_t* out_strides);

/* ---------- Reduction kernels (v0.5) ---------- */
int arix_tensor_reduce_sum_f32(const float* src, float* dst,
                                const size_t* src_shape, size_t src_ndim,
                                const size_t* reduce_dims, size_t num_dims);

int arix_tensor_reduce_mean_f32(const float* src, float* dst,
                                 const size_t* src_shape, size_t src_ndim,
                                 const size_t* reduce_dims, size_t num_dims);

int arix_tensor_reduce_max_f32(const float* src, float* dst,
                                const size_t* src_shape, size_t src_ndim,
                                const size_t* reduce_dims, size_t num_dims);

/* ---------- Type conversion ---------- */
int arix_tensor_convert_dtype(void* dst, int dst_dtype,
                               const void* src, int src_dtype,
                               size_t num_elements);

/* ---------- Element-wise operations (v0.5) ---------- */
int arix_tensor_add_f32(const float* a, const float* b, float* out, size_t n);
int arix_tensor_mul_f32(const float* a, const float* b, float* out, size_t n);
int arix_tensor_relu_f32(const float* a, float* out, size_t n);
int arix_tensor_sigmoid_f32(const float* a, float* out, size_t n);

/* ---------- Parallel dispatch helpers ---------- */
typedef void (*ArixTensorKernel1D)(void* ctx, size_t begin, size_t end);
int arix_tensor_parallel_for(ArixTensorKernel1D kernel, void* ctx,
                              size_t total_work, size_t min_grain);

#ifdef __cplusplus
}
#endif

#endif /* ARIX_TENSOR_INTERNAL_H */
