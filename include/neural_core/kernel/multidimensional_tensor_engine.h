#ifndef ARIX_TENSOR_H
#define ARIX_TENSOR_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    ARIX_FLOAT32,
    ARIX_FLOAT64,
    ARIX_FLOAT16,
    ARIX_BFLOAT16,
    ARIX_FLOAT8,
    ARIX_INT32,
    ARIX_INT64,
    ARIX_INT16,
    ARIX_INT8,
    ARIX_UINT8,
    ARIX_BOOL,
    ARIX_COMPLEX64,
    ARIX_COMPLEX128,
} ArixDtype;

#define ARIX_DTYPE_SIZE(dtype) arix_tensor_dtype_size(dtype)
#define ARIX_DTYPE_IS_FLOAT(dtype) ((dtype) == ARIX_FLOAT32 || (dtype) == ARIX_FLOAT64 || (dtype) == ARIX_FLOAT16 || (dtype) == ARIX_BFLOAT16 || (dtype) == ARIX_FLOAT8)
#define ARIX_DTYPE_IS_INT(dtype) ((dtype) == ARIX_INT32 || (dtype) == ARIX_INT64 || (dtype) == ARIX_INT16 || (dtype) == ARIX_INT8 || (dtype) == ARIX_UINT8 || (dtype) == ARIX_BOOL)
#define ARIX_DTYPE_IS_COMPLEX(dtype) ((dtype) == ARIX_COMPLEX64 || (dtype) == ARIX_COMPLEX128)

typedef enum {
    ARIX_LAYOUT_ROW_MAJOR,
    ARIX_LAYOUT_COL_MAJOR,
    ARIX_LAYOUT_CHANNELS_LAST,
    ARIX_LAYOUT_TILED,
} ArixLayout;

typedef enum {
    ARIX_DEVICE_CPU,
    ARIX_DEVICE_CUDA,
    ARIX_DEVICE_METAL,
    ARIX_DEVICE_VULKAN,
    ARIX_DEVICE_TPU,
    ARIX_DEVICE_NPU,
} ArixDevice;

typedef struct ArixStorage {
    void* data;               /* raw data buffer */
    size_t num_bytes;         /* total bytes allocated */
    int ref_count;            /* how many tensors reference this storage */
} ArixStorage;

ArixStorage* arix_storage_create(size_t num_bytes);
void         arix_storage_retain(ArixStorage* s);
void         arix_storage_release(ArixStorage* s);

typedef struct {
    ArixStorage* storage;     /* ref-counted storage (NULL for unmanaged) */
    void* data;               /* convenience pointer: storage->data + offset * item_size */
    size_t offset;            /* element offset into storage->data */
    size_t* shape;
    size_t* strides;
    size_t ndim;
    size_t size;
    size_t item_size;
    ArixDtype dtype;
    ArixDevice device;
    int device_id;
    ArixLayout layout;
    int owns_data;
    void* backend_handle;
} ArixTensor;

ArixTensor* arix_tensor_create(const size_t* shape, size_t ndim, ArixDtype dtype);
void arix_tensor_destroy(ArixTensor* tensor);

float arix_tensor_get_f32(const ArixTensor* tensor, const size_t* indices);
void arix_tensor_set_f32(ArixTensor* tensor, const size_t* indices, float value);
double arix_tensor_get_f64(const ArixTensor* tensor, const size_t* indices);
void arix_tensor_set_f64(ArixTensor* tensor, const size_t* indices, double value);
int32_t arix_tensor_get_i32(const ArixTensor* tensor, const size_t* indices);
void arix_tensor_set_i32(ArixTensor* tensor, const size_t* indices, int32_t value);
int64_t arix_tensor_get_i64(const ArixTensor* tensor, const size_t* indices);
void arix_tensor_set_i64(ArixTensor* tensor, const size_t* indices, int64_t value);
uint8_t arix_tensor_get_bool(const ArixTensor* tensor, const size_t* indices);
void arix_tensor_set_bool(ArixTensor* tensor, const size_t* indices, uint8_t value);

void arix_tensor_fill_f32(ArixTensor* tensor, float value);
void arix_tensor_fill_f64(ArixTensor* tensor, double value);

ArixTensor* arix_tensor_empty(const size_t* shape, size_t ndim, ArixDtype dtype);
ArixTensor* arix_tensor_zeros(const size_t* shape, size_t ndim, ArixDtype dtype);
ArixTensor* arix_tensor_ones(const size_t* shape, size_t ndim, ArixDtype dtype);
ArixTensor* arix_tensor_full(const size_t* shape, size_t ndim, ArixDtype dtype, const void* value);
ArixTensor* arix_tensor_arange(float start, float stop, float step, ArixDtype dtype);
ArixTensor* arix_tensor_linspace(float start, float stop, size_t steps, ArixDtype dtype);
ArixTensor* arix_tensor_eye(size_t n, ArixDtype dtype);
ArixTensor* arix_tensor_randn(const size_t* shape, size_t ndim, ArixDtype dtype);

ArixTensor* arix_tensor_copy(const ArixTensor* src);
ArixTensor* arix_tensor_clone(const ArixTensor* src);
ArixTensor* arix_tensor_slice(const ArixTensor* src, size_t dim, size_t start, size_t end);
ArixTensor* arix_tensor_reshape(const ArixTensor* src, const size_t* new_shape, size_t new_ndim);
ArixTensor* arix_tensor_permute(const ArixTensor* src, const size_t* axes);
ArixTensor* arix_tensor_expand(const ArixTensor* src, const size_t* new_shape, size_t new_ndim);
ArixTensor* arix_tensor_squeeze(const ArixTensor* src, size_t dim);
ArixTensor* arix_tensor_unsqueeze(const ArixTensor* src, size_t dim);
ArixTensor* arix_tensor_contiguous(const ArixTensor* src);
ArixTensor* arix_tensor_as_strided(const ArixTensor* src, size_t offset, const size_t* shape, size_t ndim, const size_t* strides);
ArixTensor* arix_tensor_narrow(const ArixTensor* src, size_t dim, size_t start, size_t size);

ArixTensor* arix_tensor_concat(const ArixTensor** tensors, size_t num_tensors, size_t dim);
ArixTensor** arix_tensor_split(const ArixTensor* src, size_t num_splits, size_t dim);
ArixTensor* arix_tensor_tile(const ArixTensor* src, const size_t* reps, size_t reps_ndim);
ArixTensor* arix_tensor_repeat(const ArixTensor* src, size_t repeats, size_t dim);
ArixTensor* arix_tensor_gather(const ArixTensor* src, size_t dim, const ArixTensor* indices);
ArixTensor* arix_tensor_scatter(ArixTensor* dest, size_t dim, const ArixTensor* indices, const ArixTensor* src);
ArixTensor* arix_tensor_masked_select(const ArixTensor* src, const ArixTensor* mask);
ArixTensor* arix_tensor_masked_fill(ArixTensor* src, const ArixTensor* mask, const void* value);
ArixTensor* arix_tensor_where(const ArixTensor* condition, const ArixTensor* x, const ArixTensor* y);

ArixTensor* arix_tensor_cast(const ArixTensor* src, ArixDtype dtype);
ArixTensor* arix_tensor_to_device(const ArixTensor* src, ArixDevice device);
ArixTensor* arix_tensor_to_layout(const ArixTensor* src, ArixLayout layout);
int arix_tensor_save(const ArixTensor* src, const char* path);
ArixTensor* arix_tensor_load(const char* path);

ArixTensor* arix_tensor_eq(const ArixTensor* a, const ArixTensor* b);
ArixTensor* arix_tensor_ne(const ArixTensor* a, const ArixTensor* b);
ArixTensor* arix_tensor_lt(const ArixTensor* a, const ArixTensor* b);
ArixTensor* arix_tensor_le(const ArixTensor* a, const ArixTensor* b);
ArixTensor* arix_tensor_gt(const ArixTensor* a, const ArixTensor* b);
ArixTensor* arix_tensor_ge(const ArixTensor* a, const ArixTensor* b);

ArixTensor* arix_tensor_add(const ArixTensor* a, const ArixTensor* b);
ArixTensor* arix_tensor_sub(const ArixTensor* a, const ArixTensor* b);
ArixTensor* arix_tensor_mul(const ArixTensor* a, const ArixTensor* b);
ArixTensor* arix_tensor_div(const ArixTensor* a, const ArixTensor* b);
ArixTensor* arix_tensor_minimum(const ArixTensor* a, const ArixTensor* b);
ArixTensor* arix_tensor_maximum(const ArixTensor* a, const ArixTensor* b);
ArixTensor* arix_tensor_pow(const ArixTensor* a, const ArixTensor* b);

ArixTensor* arix_tensor_neg(const ArixTensor* src);
ArixTensor* arix_tensor_abs(const ArixTensor* src);
ArixTensor* arix_tensor_sign(const ArixTensor* src);
ArixTensor* arix_tensor_floor(const ArixTensor* src);
ArixTensor* arix_tensor_ceil(const ArixTensor* src);
ArixTensor* arix_tensor_round(const ArixTensor* src);
ArixTensor* arix_tensor_trunc(const ArixTensor* src);
ArixTensor* arix_tensor_exp(const ArixTensor* src);
ArixTensor* arix_tensor_log(const ArixTensor* src);
ArixTensor* arix_tensor_sqrt(const ArixTensor* src);
ArixTensor* arix_tensor_sin(const ArixTensor* src);
ArixTensor* arix_tensor_cos(const ArixTensor* src);
ArixTensor* arix_tensor_tan(const ArixTensor* src);
ArixTensor* arix_tensor_asin(const ArixTensor* src);
ArixTensor* arix_tensor_acos(const ArixTensor* src);
ArixTensor* arix_tensor_atan(const ArixTensor* src);
ArixTensor* arix_tensor_sinh(const ArixTensor* src);
ArixTensor* arix_tensor_cosh(const ArixTensor* src);
ArixTensor* arix_tensor_tanh(const ArixTensor* src);

ArixTensor* arix_tensor_sum(const ArixTensor* src, size_t dim);
ArixTensor* arix_tensor_mean(const ArixTensor* src, size_t dim);
ArixTensor* arix_tensor_std(const ArixTensor* src, size_t dim);
ArixTensor* arix_tensor_var(const ArixTensor* src, size_t dim);
float arix_tensor_min(const ArixTensor* src);
float arix_tensor_max(const ArixTensor* src);
size_t arix_tensor_argmin(const ArixTensor* src);
size_t arix_tensor_argmax(const ArixTensor* src);
ArixTensor* arix_tensor_cumsum(const ArixTensor* src, size_t dim);
ArixTensor* arix_tensor_cumprod(const ArixTensor* src, size_t dim);

float arix_tensor_dot(const ArixTensor* a, const ArixTensor* b);
ArixTensor* arix_tensor_matmul(const ArixTensor* a, const ArixTensor* b);
ArixTensor* arix_tensor_transpose(const ArixTensor* src, size_t dim1, size_t dim2);
ArixTensor* arix_tensor_inverse(const ArixTensor* src);
float arix_tensor_det(const ArixTensor* src);

ArixTensor* arix_tensor_conv1d(const ArixTensor* input, const ArixTensor* kernel, size_t stride, size_t padding);
ArixTensor* arix_tensor_conv2d(const ArixTensor* input, const ArixTensor* kernel, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w);
ArixTensor* arix_tensor_pool1d(const ArixTensor* src, size_t kernel_size, size_t stride);
ArixTensor* arix_tensor_pool2d(const ArixTensor* src, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w);

ArixTensor* arix_tensor_softmax(const ArixTensor* src, size_t dim);
ArixTensor* arix_tensor_log_softmax(const ArixTensor* src, size_t dim);
ArixTensor* arix_tensor_relu(const ArixTensor* src);
ArixTensor* arix_tensor_gelu(const ArixTensor* src);
ArixTensor* arix_tensor_silu(const ArixTensor* src);
ArixTensor* arix_tensor_sigmoid(const ArixTensor* src);
ArixTensor* arix_tensor_dropout(const ArixTensor* src, float rate, unsigned int seed);
ArixTensor* arix_tensor_layer_norm(const ArixTensor* src, const ArixTensor* gamma, const ArixTensor* beta, float eps);
ArixTensor* arix_tensor_batch_norm(const ArixTensor* src, const ArixTensor* gamma, const ArixTensor* beta, const ArixTensor* running_mean, const ArixTensor* running_var, float eps);
ArixTensor* arix_tensor_group_norm(const ArixTensor* src, const ArixTensor* gamma, const ArixTensor* beta, size_t num_groups, float eps);
ArixTensor* arix_tensor_instance_norm(const ArixTensor* src, const ArixTensor* gamma, const ArixTensor* beta, float eps);
ArixTensor* arix_tensor_embedding(const ArixTensor* weight, const ArixTensor* indices);

ArixTensor* arix_tensor_cross_entropy(const ArixTensor* pred, const ArixTensor* target);
ArixTensor* arix_tensor_mse_loss(const ArixTensor* pred, const ArixTensor* target);
ArixTensor* arix_tensor_mae_loss(const ArixTensor* pred, const ArixTensor* target);
ArixTensor* arix_tensor_nll_loss(const ArixTensor* pred, const ArixTensor* target);
ArixTensor* arix_tensor_kl_div(const ArixTensor* pred, const ArixTensor* target);
ArixTensor* arix_tensor_binary_cross_entropy(const ArixTensor* pred, const ArixTensor* target);

void arix_tensor_print(const ArixTensor* tensor);
size_t arix_tensor_dtype_size(ArixDtype dtype);
size_t arix_tensor_numel(const ArixTensor* tensor);
int arix_tensor_is_contiguous(const ArixTensor* tensor);
const char* arix_tensor_dtype_name(ArixDtype dtype);

#endif
