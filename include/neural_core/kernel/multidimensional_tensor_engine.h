#ifndef SNEPPX_TENSOR_H
#define SNEPPX_TENSOR_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    SNEPPX_FLOAT32,
    SNEPPX_FLOAT64,
    SNEPPX_FLOAT16,
    SNEPPX_BFLOAT16,
    SNEPPX_FLOAT8,
    SNEPPX_INT32,
    SNEPPX_INT64,
    SNEPPX_INT16,
    SNEPPX_INT8,
    SNEPPX_UINT8,
    SNEPPX_BOOL,
    SNEPPX_COMPLEX64,
    SNEPPX_COMPLEX128,
} SNEPPXDtype;

#define SNEPPX_DTYPE_SIZE(dtype) SNEPPX_tensor_dtype_size(dtype)
#define SNEPPX_DTYPE_IS_FLOAT(dtype) ((dtype) == SNEPPX_FLOAT32 || (dtype) == SNEPPX_FLOAT64 || (dtype) == SNEPPX_FLOAT16 || (dtype) == SNEPPX_BFLOAT16 || (dtype) == SNEPPX_FLOAT8)
#define SNEPPX_DTYPE_IS_INT(dtype) ((dtype) == SNEPPX_INT32 || (dtype) == SNEPPX_INT64 || (dtype) == SNEPPX_INT16 || (dtype) == SNEPPX_INT8 || (dtype) == SNEPPX_UINT8 || (dtype) == SNEPPX_BOOL)
#define SNEPPX_DTYPE_IS_COMPLEX(dtype) ((dtype) == SNEPPX_COMPLEX64 || (dtype) == SNEPPX_COMPLEX128)

typedef enum {
    SNEPPX_LAYOUT_ROW_MAJOR,
    SNEPPX_LAYOUT_COL_MAJOR,
    SNEPPX_LAYOUT_CHANNELS_LAST,
    SNEPPX_LAYOUT_TILED,
} SNEPPXLayout;

typedef enum {
    SNEPPX_DEVICE_CPU,
    SNEPPX_DEVICE_CUDA,
    SNEPPX_DEVICE_METAL,
    SNEPPX_DEVICE_VULKAN,
    SNEPPX_DEVICE_TPU,
    SNEPPX_DEVICE_NPU,
} SNEPPXDevice;

typedef struct SNEPPXStorage {
    void* data;               /* raw data buffer */
    size_t num_bytes;         /* total bytes allocated */
    int ref_count;            /* how many tensors reference this storage */
} SNEPPXStorage;

SNEPPXStorage* SNEPPX_storage_create(size_t num_bytes);
void         SNEPPX_storage_retain(SNEPPXStorage* s);
void         SNEPPX_storage_release(SNEPPXStorage* s);

typedef struct {
    SNEPPXStorage* storage;     /* ref-counted storage (NULL for unmanaged) */
    void* data;               /* convenience pointer: storage->data + offset * item_size */
    size_t offset;            /* element offset into storage->data */
    size_t* shape;
    size_t* strides;
    size_t ndim;
    size_t size;
    size_t item_size;
    SNEPPXDtype dtype;
    SNEPPXDevice device;
    int device_id;
    SNEPPXLayout layout;
    int owns_data;
    void* backend_handle;
} SNEPPXTensor;

SNEPPXTensor* SNEPPX_tensor_create(const size_t* shape, size_t ndim, SNEPPXDtype dtype);
void SNEPPX_tensor_destroy(SNEPPXTensor* tensor);

float SNEPPX_tensor_get_f32(const SNEPPXTensor* tensor, const size_t* indices);
void SNEPPX_tensor_set_f32(SNEPPXTensor* tensor, const size_t* indices, float value);
double SNEPPX_tensor_get_f64(const SNEPPXTensor* tensor, const size_t* indices);
void SNEPPX_tensor_set_f64(SNEPPXTensor* tensor, const size_t* indices, double value);
int32_t SNEPPX_tensor_get_i32(const SNEPPXTensor* tensor, const size_t* indices);
void SNEPPX_tensor_set_i32(SNEPPXTensor* tensor, const size_t* indices, int32_t value);
int64_t SNEPPX_tensor_get_i64(const SNEPPXTensor* tensor, const size_t* indices);
void SNEPPX_tensor_set_i64(SNEPPXTensor* tensor, const size_t* indices, int64_t value);
uint8_t SNEPPX_tensor_get_bool(const SNEPPXTensor* tensor, const size_t* indices);
void SNEPPX_tensor_set_bool(SNEPPXTensor* tensor, const size_t* indices, uint8_t value);

void SNEPPX_tensor_fill_f32(SNEPPXTensor* tensor, float value);
void SNEPPX_tensor_fill_f64(SNEPPXTensor* tensor, double value);

SNEPPXTensor* SNEPPX_tensor_empty(const size_t* shape, size_t ndim, SNEPPXDtype dtype);
SNEPPXTensor* SNEPPX_tensor_zeros(const size_t* shape, size_t ndim, SNEPPXDtype dtype);
SNEPPXTensor* SNEPPX_tensor_ones(const size_t* shape, size_t ndim, SNEPPXDtype dtype);
SNEPPXTensor* SNEPPX_tensor_full(const size_t* shape, size_t ndim, SNEPPXDtype dtype, const void* value);
SNEPPXTensor* SNEPPX_tensor_arange(float start, float stop, float step, SNEPPXDtype dtype);
SNEPPXTensor* SNEPPX_tensor_linspace(float start, float stop, size_t steps, SNEPPXDtype dtype);
SNEPPXTensor* SNEPPX_tensor_eye(size_t n, SNEPPXDtype dtype);
SNEPPXTensor* SNEPPX_tensor_randn(const size_t* shape, size_t ndim, SNEPPXDtype dtype);

SNEPPXTensor* SNEPPX_tensor_copy(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_clone(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_slice(const SNEPPXTensor* src, size_t dim, size_t start, size_t end);
SNEPPXTensor* SNEPPX_tensor_reshape(const SNEPPXTensor* src, const size_t* new_shape, size_t new_ndim);
SNEPPXTensor* SNEPPX_tensor_permute(const SNEPPXTensor* src, const size_t* axes);
SNEPPXTensor* SNEPPX_tensor_expand(const SNEPPXTensor* src, const size_t* new_shape, size_t new_ndim);
SNEPPXTensor* SNEPPX_tensor_squeeze(const SNEPPXTensor* src, size_t dim);
SNEPPXTensor* SNEPPX_tensor_unsqueeze(const SNEPPXTensor* src, size_t dim);
SNEPPXTensor* SNEPPX_tensor_contiguous(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_as_strided(const SNEPPXTensor* src, size_t offset, const size_t* shape, size_t ndim, const size_t* strides);
SNEPPXTensor* SNEPPX_tensor_narrow(const SNEPPXTensor* src, size_t dim, size_t start, size_t size);

SNEPPXTensor* SNEPPX_tensor_concat(const SNEPPXTensor** tensors, size_t num_tensors, size_t dim);
SNEPPXTensor** SNEPPX_tensor_split(const SNEPPXTensor* src, size_t num_splits, size_t dim);
SNEPPXTensor* SNEPPX_tensor_tile(const SNEPPXTensor* src, const size_t* reps, size_t reps_ndim);
SNEPPXTensor* SNEPPX_tensor_repeat(const SNEPPXTensor* src, size_t repeats, size_t dim);
SNEPPXTensor* SNEPPX_tensor_gather(const SNEPPXTensor* src, size_t dim, const SNEPPXTensor* indices);
SNEPPXTensor* SNEPPX_tensor_scatter(SNEPPXTensor* dest, size_t dim, const SNEPPXTensor* indices, const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_masked_select(const SNEPPXTensor* src, const SNEPPXTensor* mask);
SNEPPXTensor* SNEPPX_tensor_masked_fill(SNEPPXTensor* src, const SNEPPXTensor* mask, const void* value);
SNEPPXTensor* SNEPPX_tensor_where(const SNEPPXTensor* condition, const SNEPPXTensor* x, const SNEPPXTensor* y);

SNEPPXTensor* SNEPPX_tensor_cast(const SNEPPXTensor* src, SNEPPXDtype dtype);
SNEPPXTensor* SNEPPX_tensor_to_device(const SNEPPXTensor* src, SNEPPXDevice device);
SNEPPXTensor* SNEPPX_tensor_to_layout(const SNEPPXTensor* src, SNEPPXLayout layout);
int SNEPPX_tensor_save(const SNEPPXTensor* src, const char* path);
SNEPPXTensor* SNEPPX_tensor_load(const char* path);

SNEPPXTensor* SNEPPX_tensor_eq(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_ne(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_lt(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_le(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_gt(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_ge(const SNEPPXTensor* a, const SNEPPXTensor* b);

SNEPPXTensor* SNEPPX_tensor_add(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_sub(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_mul(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_div(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_minimum(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_maximum(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_pow(const SNEPPXTensor* a, const SNEPPXTensor* b);

SNEPPXTensor* SNEPPX_tensor_neg(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_abs(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_sign(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_floor(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_ceil(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_round(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_trunc(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_exp(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_log(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_sqrt(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_sin(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_cos(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_tan(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_asin(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_acos(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_atan(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_sinh(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_cosh(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_tanh(const SNEPPXTensor* src);

SNEPPXTensor* SNEPPX_tensor_sum(const SNEPPXTensor* src, size_t dim);
SNEPPXTensor* SNEPPX_tensor_mean(const SNEPPXTensor* src, size_t dim);
SNEPPXTensor* SNEPPX_tensor_std(const SNEPPXTensor* src, size_t dim);
SNEPPXTensor* SNEPPX_tensor_var(const SNEPPXTensor* src, size_t dim);
float SNEPPX_tensor_min(const SNEPPXTensor* src);
float SNEPPX_tensor_max(const SNEPPXTensor* src);
size_t SNEPPX_tensor_argmin(const SNEPPXTensor* src);
size_t SNEPPX_tensor_argmax(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_cumsum(const SNEPPXTensor* src, size_t dim);
SNEPPXTensor* SNEPPX_tensor_cumprod(const SNEPPXTensor* src, size_t dim);

float SNEPPX_tensor_dot(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_matmul(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_transpose(const SNEPPXTensor* src, size_t dim1, size_t dim2);
SNEPPXTensor* SNEPPX_tensor_inverse(const SNEPPXTensor* src);
float SNEPPX_tensor_det(const SNEPPXTensor* src);

SNEPPXTensor* SNEPPX_tensor_conv1d(const SNEPPXTensor* input, const SNEPPXTensor* kernel, size_t stride, size_t padding);
SNEPPXTensor* SNEPPX_tensor_conv2d(const SNEPPXTensor* input, const SNEPPXTensor* kernel, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w);
SNEPPXTensor* SNEPPX_tensor_pool1d(const SNEPPXTensor* src, size_t kernel_size, size_t stride);
SNEPPXTensor* SNEPPX_tensor_pool2d(const SNEPPXTensor* src, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w);

SNEPPXTensor* SNEPPX_tensor_softmax(const SNEPPXTensor* src, size_t dim);
SNEPPXTensor* SNEPPX_tensor_log_softmax(const SNEPPXTensor* src, size_t dim);
SNEPPXTensor* SNEPPX_tensor_relu(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_gelu(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_silu(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_sigmoid(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_dropout(const SNEPPXTensor* src, float rate, unsigned int seed);
SNEPPXTensor* SNEPPX_tensor_layer_norm(const SNEPPXTensor* src, const SNEPPXTensor* gamma, const SNEPPXTensor* beta, float eps);
SNEPPXTensor* SNEPPX_tensor_batch_norm(const SNEPPXTensor* src, const SNEPPXTensor* gamma, const SNEPPXTensor* beta, const SNEPPXTensor* running_mean, const SNEPPXTensor* running_var, float eps);
SNEPPXTensor* SNEPPX_tensor_group_norm(const SNEPPXTensor* src, const SNEPPXTensor* gamma, const SNEPPXTensor* beta, size_t num_groups, float eps);
SNEPPXTensor* SNEPPX_tensor_instance_norm(const SNEPPXTensor* src, const SNEPPXTensor* gamma, const SNEPPXTensor* beta, float eps);
SNEPPXTensor* SNEPPX_tensor_embedding(const SNEPPXTensor* weight, const SNEPPXTensor* indices);

SNEPPXTensor* SNEPPX_tensor_cross_entropy(const SNEPPXTensor* pred, const SNEPPXTensor* target);
SNEPPXTensor* SNEPPX_tensor_mse_loss(const SNEPPXTensor* pred, const SNEPPXTensor* target);
SNEPPXTensor* SNEPPX_tensor_mae_loss(const SNEPPXTensor* pred, const SNEPPXTensor* target);
SNEPPXTensor* SNEPPX_tensor_nll_loss(const SNEPPXTensor* pred, const SNEPPXTensor* target);
SNEPPXTensor* SNEPPX_tensor_kl_div(const SNEPPXTensor* pred, const SNEPPXTensor* target);
SNEPPXTensor* SNEPPX_tensor_binary_cross_entropy(const SNEPPXTensor* pred, const SNEPPXTensor* target);

void SNEPPX_tensor_print(const SNEPPXTensor* tensor);
size_t SNEPPX_tensor_dtype_size(SNEPPXDtype dtype);
size_t SNEPPX_tensor_numel(const SNEPPXTensor* tensor);
int SNEPPX_tensor_is_contiguous(const SNEPPXTensor* tensor);
const char* SNEPPX_tensor_dtype_name(SNEPPXDtype dtype);

#endif
