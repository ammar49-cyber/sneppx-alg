#include "test_common.h"
#include "multidimensional_tensor_engine.h"

static void test_create_0d(void) {
    size_t shape[] = {1};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT32);
    ASSERT_NOT_NULL(t, "0d tensor");
    ASSERT_EQ(t->size, 1, "size == 1");
    ASSERT_EQ(t->ndim, 1, "ndim == 1");
    SNEPPX_tensor_destroy(t);
}

static void test_create_zero_dim(void) {
    SNEPPXTensor* t = SNEPPX_tensor_create(NULL, 0, SNEPPX_FLOAT32);
    if (t) { ASSERT_EQ(t->ndim, 0, "ndim == 0"); ASSERT_EQ(t->size, 1, "size == 1"); SNEPPX_tensor_destroy(t); }
}

static void test_create_large_ndim(void) {
    size_t shape[16];
    for (size_t i = 0; i < 16; i++) shape[i] = 1;
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 16, SNEPPX_FLOAT32);
    ASSERT_NOT_NULL(t, "16-dim tensor");
    ASSERT_EQ(t->ndim, 16, "ndim == 16");
    SNEPPX_tensor_destroy(t);
}

static void test_destroy_null(void) {
    SNEPPX_tensor_destroy(NULL);
}

static void test_add_null(void) {
    size_t sh[] = {2, 2};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_add(a, NULL);
    ASSERT_NULL(r, "add NULL b -> NULL");
    r = SNEPPX_tensor_add(NULL, a);
    ASSERT_NULL(r, "add NULL a -> NULL");
    r = SNEPPX_tensor_add(NULL, NULL);
    ASSERT_NULL(r, "add NULL NULL -> NULL");
    SNEPPX_tensor_destroy(a);
}

static void test_sub_null(void) {
    size_t sh[] = {2, 2};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_sub(a, NULL);
    ASSERT_NULL(r, "sub NULL -> NULL");
    SNEPPX_tensor_destroy(a);
}

static void test_mul_null(void) {
    size_t sh[] = {2, 2};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_mul(a, NULL);
    ASSERT_NULL(r, "mul NULL -> NULL");
    SNEPPX_tensor_destroy(a);
}

static void test_div_null(void) {
    size_t sh[] = {2, 2};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_div(a, NULL);
    ASSERT_NULL(r, "div NULL -> NULL");
    SNEPPX_tensor_destroy(a);
}

static void test_matmul_null(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_matmul(a, NULL);
    ASSERT_NULL(r, "matmul NULL -> NULL");
    r = SNEPPX_tensor_matmul(NULL, a);
    ASSERT_NULL(r, "matmul NULL a -> NULL");
    SNEPPX_tensor_destroy(a);
}

static void test_matmul_mismatched_inner(void) {
    size_t sha[] = {2, 3};
    size_t shb[] = {4, 5};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sha, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* b = SNEPPX_tensor_ones(shb, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_matmul(a, b);
    ASSERT_NULL(r, "matmul mismatched inner -> NULL");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b);
}

static void test_matmul_1d_inputs(void) {
    size_t sh[] = {3};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* b = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_matmul(a, b);
    ASSERT_NULL(r, "matmul 1d -> NULL");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b);
}

static void test_slice_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_slice(NULL, 0, 0, 1);
    ASSERT_NULL(r, "slice NULL -> NULL");
}

static void test_slice_out_of_bounds_dim(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_slice(t, 5, 0, 1);
    ASSERT_NULL(r, "slice dim>=ndim -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_slice_invalid_start_end(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_slice(t, 0, 2, 1);
    ASSERT_NULL(r, "slice start>=end -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_slice_end_exceeds_shape(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_slice(t, 0, 0, 5);
    ASSERT_NULL(r, "slice end>shape -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_reshape_null(void) {
    size_t ns[] = {6};
    SNEPPXTensor* r = SNEPPX_tensor_reshape(NULL, ns, 1);
    ASSERT_NULL(r, "reshape NULL -> NULL");
}

static void test_reshape_invalid_total(void) {
    size_t sh[] = {2, 3};
    size_t ns[] = {5};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_reshape(t, ns, 1);
    ASSERT_NOT_NULL(r, "reshape mismatched still returns data");
    if (r) {
        ASSERT_EQ(r->size, 5, "size is 5 despite mismatch");
        SNEPPX_tensor_destroy(r);
    }
    SNEPPX_tensor_destroy(t);
}

static void test_permute_null(void) {
    size_t axes[] = {1, 0};
    SNEPPXTensor* r = SNEPPX_tensor_permute(NULL, axes);
    ASSERT_NULL(r, "permute NULL -> NULL");
}

static void test_transpose_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_transpose(NULL, 0, 1);
    ASSERT_NULL(r, "transpose NULL -> NULL");
}

static void test_transpose_invalid_dim(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_transpose(t, 0, 5);
    ASSERT_NULL(r, "transpose dim>=ndim -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_expand_null(void) {
    size_t ns[] = {2, 3};
    SNEPPXTensor* r = SNEPPX_tensor_expand(NULL, ns, 2);
    ASSERT_NULL(r, "expand NULL -> NULL");
}

static void test_expand_reduce_ndim(void) {
    size_t sh[] = {2, 3};
    size_t ns[] = {6};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_expand(t, ns, 1);
    ASSERT_NULL(r, "expand fewer dims -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_unsqueeze_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_unsqueeze(NULL, 0);
    ASSERT_NULL(r, "unsqueeze NULL -> NULL");
}

static void test_unsqueeze_invalid_dim(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_unsqueeze(t, 5);
    ASSERT_NULL(r, "unsqueeze dim>ndim -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_squeeze_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_squeeze(NULL, 0);
    ASSERT_NULL(r, "squeeze NULL -> NULL");
}

static void test_squeeze_invalid_dim(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_squeeze(t, 5);
    ASSERT_NOT_NULL(r, "squeeze dim>=ndim returns copy");
    SNEPPX_tensor_destroy(t);
    if (r) SNEPPX_tensor_destroy(r);
}

static void test_concat_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_concat(NULL, 0, 0);
    ASSERT_NULL(r, "concat NULL -> NULL");
}

static void test_concat_single(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* arr[] = {t};
    SNEPPXTensor* r = SNEPPX_tensor_concat(arr, 1, 0);
    ASSERT_NOT_NULL(r, "concat single tensor");
    ASSERT_EQ(r->shape[0], 2, "dim0 unchanged");
    ASSERT_EQ(r->shape[1], 3, "dim1 unchanged");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(r);
}

static void test_concat_mismatched_ndim(void) {
    size_t sh1[] = {2, 3};
    size_t sh2[] = {6};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sh1, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* b = SNEPPX_tensor_ones(sh2, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* arr[] = {a, b};
    SNEPPXTensor* r = SNEPPX_tensor_concat(arr, 2, 0);
    ASSERT_NULL(r, "concat mismatched ndim -> NULL");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b);
}

static void test_concat_mismatched_shape(void) {
    size_t sh1[] = {2, 3};
    size_t sh2[] = {2, 4};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sh1, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* b = SNEPPX_tensor_ones(sh2, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* arr[] = {a, b};
    SNEPPXTensor* r = SNEPPX_tensor_concat(arr, 2, 0);
    ASSERT_NULL(r, "concat mismatched non-concat dim -> NULL");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b);
}

static void test_sum_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_sum(NULL, 0);
    ASSERT_NULL(r, "sum NULL -> NULL");
}

static void test_sum_invalid_dim(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_sum(t, 5);
    ASSERT_NULL(r, "sum dim>=ndim -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_mean_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_mean(NULL, 0);
    ASSERT_NULL(r, "mean NULL -> NULL");
}

static void test_mean_invalid_dim(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_mean(t, 5);
    ASSERT_NULL(r, "mean dim>=ndim -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_min_max_empty(void) {
    size_t sh[] = {0};
    SNEPPXTensor* t = SNEPPX_tensor_create(sh, 1, SNEPPX_FLOAT32);
    if (t) {
        float v = SNEPPX_tensor_min(t);
        ASSERT_EQ(v, 0.0f, "min of empty -> 0");
        v = SNEPPX_tensor_max(t);
        ASSERT_EQ(v, 0.0f, "max of empty -> 0");
        size_t idx = SNEPPX_tensor_argmin(t);
        ASSERT_EQ(idx, 0, "argmin of empty -> 0");
        idx = SNEPPX_tensor_argmax(t);
        ASSERT_EQ(idx, 0, "argmax of empty -> 0");
        SNEPPX_tensor_destroy(t);
    }
}

static void test_min_max_null(void) {
    float v = SNEPPX_tensor_min(NULL);
    ASSERT_EQ(v, 0.0f, "min NULL -> 0");
    v = SNEPPX_tensor_max(NULL);
    ASSERT_EQ(v, 0.0f, "max NULL -> 0");
    size_t idx = SNEPPX_tensor_argmin(NULL);
    ASSERT_EQ(idx, 0, "argmin NULL -> 0");
    idx = SNEPPX_tensor_argmax(NULL);
    ASSERT_EQ(idx, 0, "argmax NULL -> 0");
}

static void test_dot_null(void) {
    size_t sh[] = {3};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    float v = SNEPPX_tensor_dot(a, NULL);
    ASSERT_EQ(v, 0.0f, "dot with NULL -> 0");
    v = SNEPPX_tensor_dot(NULL, a);
    ASSERT_EQ(v, 0.0f, "dot NULL -> 0");
    SNEPPX_tensor_destroy(a);
}

static void test_dot_mismatched(void) {
    size_t sh1[] = {3};
    size_t sh2[] = {4};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sh1, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* b = SNEPPX_tensor_ones(sh2, 1, SNEPPX_FLOAT32);
    float v = SNEPPX_tensor_dot(a, b);
    ASSERT_EQ(v, 0.0f, "dot mismatched -> 0");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b);
}

static void test_cast_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_cast(NULL, SNEPPX_FLOAT64);
    ASSERT_NULL(r, "cast NULL -> NULL");
}

static void test_copy_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_copy(NULL);
    ASSERT_NULL(r, "copy NULL -> NULL");
}

static void test_clone_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_clone(NULL);
    ASSERT_NULL(r, "clone NULL -> NULL");
}

static void test_split_null(void) {
    SNEPPXTensor** r = SNEPPX_tensor_split(NULL, 2, 0);
    ASSERT_NULL(r, "split NULL -> NULL");
}

static void test_split_zero_splits(void) {
    size_t sh[] = {4, 4};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor** r = SNEPPX_tensor_split(t, 0, 0);
    ASSERT_NULL(r, "split 0 -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_split_uneven(void) {
    size_t sh[] = {4, 4};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor** r = SNEPPX_tensor_split(t, 3, 0);
    ASSERT_NULL(r, "split uneven -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_relu_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_relu(NULL);
    ASSERT_NULL(r, "relu NULL -> NULL");
}

static void test_sigmoid_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_sigmoid(NULL);
    ASSERT_NULL(r, "sigmoid NULL -> NULL");
}

static void test_softmax_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_softmax(NULL, 0);
    ASSERT_NULL(r, "softmax NULL -> NULL");
}

static void test_layer_norm_null(void) {
    size_t sh[] = {4};
    SNEPPXTensor* g = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* b = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_layer_norm(NULL, g, b, 1e-5f);
    ASSERT_NULL(r, "layer_norm NULL input -> NULL");
    SNEPPX_tensor_destroy(g); SNEPPX_tensor_destroy(b);
}

static void test_where_null(void) {
    size_t sh[] = {3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_where(NULL, t, t);
    ASSERT_NULL(r, "where NULL condition -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_masked_select_null(void) {
    size_t sh[] = {3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_masked_select(NULL, t);
    ASSERT_NULL(r, "masked_select NULL src -> NULL");
    r = SNEPPX_tensor_masked_select(t, NULL);
    ASSERT_NULL(r, "masked_select NULL mask -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_gather_null(void) {
    size_t sh[] = {3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    size_t ish[] = {2};
    SNEPPXTensor* idx = SNEPPX_tensor_zeros(ish, 1, SNEPPX_INT32);
    SNEPPXTensor* r = SNEPPX_tensor_gather(NULL, 0, idx);
    ASSERT_NULL(r, "gather NULL src -> NULL");
    r = SNEPPX_tensor_gather(t, 0, NULL);
    ASSERT_NULL(r, "gather NULL idx -> NULL");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(idx);
}

static void test_tile_null(void) {
    size_t reps[] = {2};
    SNEPPXTensor* r = SNEPPX_tensor_tile(NULL, reps, 1);
    ASSERT_NULL(r, "tile NULL -> NULL");
}

static void test_repeat_null(void) {
    size_t sh[] = {3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_repeat(NULL, 2, 0);
    ASSERT_NULL(r, "repeat NULL -> NULL");
    r = SNEPPX_tensor_repeat(t, 2, 5);
    ASSERT_NULL(r, "repeat dim>=ndim -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_nan_inf_ops(void) {
    size_t sh[] = {4};
    SNEPPXTensor* t = SNEPPX_tensor_create(sh, 1, SNEPPX_FLOAT32);
    ASSERT_NOT_NULL(t, "created for nan/inf");
    float* d = (float*)t->data;
    d[0] = NAN; d[1] = INFINITY;
    d[2] = -INFINITY; d[3] = 3.14f;
    SNEPPXTensor* r = SNEPPX_tensor_add(t, t);
    ASSERT_NOT_NULL(r, "add with nan/inf");
    float* rd = (float*)r->data;
    ASSERT(isnan(rd[0]), "nan + nan = nan");
    ASSERT(isinf(rd[1]) && rd[1] > 0, "inf + inf = inf");
    ASSERT(isinf(rd[2]) && rd[2] < 0, "-inf + -inf = -inf");
    ASSERT_NEAR(rd[3], 6.28f, 1e-4f, "normal value preserved");
    SNEPPX_tensor_destroy(r);
    r = SNEPPX_tensor_mul(t, t);
    ASSERT_NOT_NULL(r, "mul with nan/inf");
    rd = (float*)r->data;
    ASSERT(isnan(rd[0]), "nan * nan = nan");
    ASSERT(isinf(rd[1]) && rd[1] > 0, "inf * inf = inf");
    SNEPPX_tensor_destroy(r);
    SNEPPX_tensor_destroy(t);
}

static void test_unary_op_null(void) {
    ASSERT_NULL(SNEPPX_tensor_neg(NULL), "neg NULL");
    ASSERT_NULL(SNEPPX_tensor_abs(NULL), "abs NULL");
    ASSERT_NULL(SNEPPX_tensor_sign(NULL), "sign NULL");
    ASSERT_NULL(SNEPPX_tensor_floor(NULL), "floor NULL");
    ASSERT_NULL(SNEPPX_tensor_ceil(NULL), "ceil NULL");
    ASSERT_NULL(SNEPPX_tensor_round(NULL), "round NULL");
    ASSERT_NULL(SNEPPX_tensor_trunc(NULL), "trunc NULL");
    ASSERT_NULL(SNEPPX_tensor_exp(NULL), "exp NULL");
    ASSERT_NULL(SNEPPX_tensor_log(NULL), "log NULL");
    ASSERT_NULL(SNEPPX_tensor_sqrt(NULL), "sqrt NULL");
    ASSERT_NULL(SNEPPX_tensor_sin(NULL), "sin NULL");
    ASSERT_NULL(SNEPPX_tensor_cos(NULL), "cos NULL");
    ASSERT_NULL(SNEPPX_tensor_tan(NULL), "tan NULL");
    ASSERT_NULL(SNEPPX_tensor_asin(NULL), "asin NULL");
    ASSERT_NULL(SNEPPX_tensor_acos(NULL), "acos NULL");
    ASSERT_NULL(SNEPPX_tensor_atan(NULL), "atan NULL");
    ASSERT_NULL(SNEPPX_tensor_sinh(NULL), "sinh NULL");
    ASSERT_NULL(SNEPPX_tensor_cosh(NULL), "cosh NULL");
    ASSERT_NULL(SNEPPX_tensor_tanh(NULL), "tanh NULL");
}

static void test_compare_op_null(void) {
    size_t sh[] = {3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    ASSERT_NULL(SNEPPX_tensor_eq(NULL, t), "eq NULL");
    ASSERT_NULL(SNEPPX_tensor_eq(t, NULL), "eq NULL b");
    ASSERT_NULL(SNEPPX_tensor_ne(NULL, t), "ne NULL");
    ASSERT_NULL(SNEPPX_tensor_lt(NULL, t), "lt NULL");
    ASSERT_NULL(SNEPPX_tensor_le(NULL, t), "le NULL");
    ASSERT_NULL(SNEPPX_tensor_gt(NULL, t), "gt NULL");
    ASSERT_NULL(SNEPPX_tensor_ge(NULL, t), "ge NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_cumsum_cumprod_null(void) {
    ASSERT_NULL(SNEPPX_tensor_cumsum(NULL, 0), "cumsum NULL");
    ASSERT_NULL(SNEPPX_tensor_cumprod(NULL, 0), "cumprod NULL");
}

static void test_cumsum_invalid_dim(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    ASSERT_NULL(SNEPPX_tensor_cumsum(t, 5), "cumsum bad dim");
    ASSERT_NULL(SNEPPX_tensor_cumprod(t, 5), "cumprod bad dim");
    SNEPPX_tensor_destroy(t);
}

int main(void) {
    run_test("create 0d", test_create_0d);
    run_test("create zero dim", test_create_zero_dim);
    run_test("create large ndim", test_create_large_ndim);
    run_test("destroy null", test_destroy_null);
    run_test("add null", test_add_null);
    run_test("sub null", test_sub_null);
    run_test("mul null", test_mul_null);
    run_test("div null", test_div_null);
    run_test("matmul null", test_matmul_null);
    run_test("matmul mismatched inner", test_matmul_mismatched_inner);
    run_test("matmul 1d inputs", test_matmul_1d_inputs);
    run_test("slice null", test_slice_null);
    run_test("slice out-of-bounds dim", test_slice_out_of_bounds_dim);
    run_test("slice invalid start/end", test_slice_invalid_start_end);
    run_test("slice end exceeds shape", test_slice_end_exceeds_shape);
    run_test("reshape null", test_reshape_null);
    run_test("permute null", test_permute_null);
    run_test("transpose null", test_transpose_null);
    run_test("transpose invalid dim", test_transpose_invalid_dim);
    run_test("expand null", test_expand_null);
    run_test("expand reduce ndim", test_expand_reduce_ndim);
    run_test("unsqueeze null", test_unsqueeze_null);
    run_test("unsqueeze invalid dim", test_unsqueeze_invalid_dim);
    run_test("squeeze null", test_squeeze_null);
    run_test("squeeze invalid dim", test_squeeze_invalid_dim);
    run_test("concat null", test_concat_null);
    run_test("concat single", test_concat_single);
    run_test("concat mismatched ndim", test_concat_mismatched_ndim);
    run_test("concat mismatched shape", test_concat_mismatched_shape);
    run_test("sum null", test_sum_null);
    run_test("sum invalid dim", test_sum_invalid_dim);
    run_test("mean null", test_mean_null);
    run_test("mean invalid dim", test_mean_invalid_dim);
    run_test("min/max empty", test_min_max_empty);
    run_test("min/max null", test_min_max_null);
    run_test("dot null", test_dot_null);
    run_test("dot mismatched", test_dot_mismatched);
    run_test("cast null", test_cast_null);
    run_test("copy null", test_copy_null);
    run_test("clone null", test_clone_null);
    run_test("split null", test_split_null);
    run_test("split zero", test_split_zero_splits);
    run_test("split uneven", test_split_uneven);
    run_test("relu null", test_relu_null);
    run_test("sigmoid null", test_sigmoid_null);
    run_test("softmax null", test_softmax_null);
    run_test("layer_norm null", test_layer_norm_null);
    run_test("where null", test_where_null);
    run_test("masked_select null", test_masked_select_null);
    run_test("gather null", test_gather_null);
    run_test("tile null", test_tile_null);
    run_test("repeat null/bad dim", test_repeat_null);
    run_test("nan/inf ops", test_nan_inf_ops);
    run_test("unary op null", test_unary_op_null);
    run_test("compare op null", test_compare_op_null);
    run_test("cumsum/cumprod null", test_cumsum_cumprod_null);
    run_test("cumsum/cumprod bad dim", test_cumsum_invalid_dim);
    printf("\nResults: %d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
