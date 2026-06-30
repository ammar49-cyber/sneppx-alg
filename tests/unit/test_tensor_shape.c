#include "test_common.h"
#include "multidimensional_tensor_engine.h"
#include "polymorphic_memory_allocator.h"

static void test_copy_independent(void) {
    ArixTensor* a = arix_tensor_arange(0.0f, 6.0f, 1.0f, ARIX_FLOAT32);
    ASSERT_NOT_NULL(a, "copy src not null");
    ArixTensor* b = arix_tensor_copy(a);
    ASSERT_NOT_NULL(b, "copy dst not null");
    ((float*)a->data)[0] = 999.0f;
    ASSERT_NEAR(((float*)b->data)[0], 0.0f, 1e-6f, "copy independent");
    arix_tensor_destroy(a); arix_tensor_destroy(b);
}

static void test_copy_null(void) {
    ASSERT_NULL(arix_tensor_copy(NULL), "copy NULL -> NULL");
}

static void test_clone_null(void) {
    ASSERT_NULL(arix_tensor_clone(NULL), "clone NULL -> NULL");
}

static void test_reshape_2d_to_1d(void) {
    ArixTensor* t = arix_tensor_arange(0.0f, 6.0f, 1.0f, ARIX_FLOAT32);
    ASSERT_NOT_NULL(t, "reshape src not null");
    size_t ns[] = {6};
    ArixTensor* r = arix_tensor_reshape(t, ns, 1);
    ASSERT_NOT_NULL(r, "reshape not null");
    ASSERT_EQ(r->ndim, 1, "reshape ndim == 1");
    ASSERT_EQ(r->size, 6, "reshape size == 6");
    float* d = (float*)r->data;
    for (size_t i = 0; i < 6; i++) ASSERT_NEAR(d[i], (float)i, 1e-6f, "reshape value");
    arix_tensor_destroy(t); arix_tensor_destroy(r);
}

static void test_reshape_auto_dim(void) {
    ArixTensor* t = arix_tensor_arange(0.0f, 24.0f, 1.0f, ARIX_FLOAT32);
    ASSERT_NOT_NULL(t, "reshape auto src not null");
    size_t ns[] = {2, (size_t)-1};
    ArixTensor* r = arix_tensor_reshape(t, ns, 2);
    ASSERT_NOT_NULL(r, "reshape auto not null");
    ASSERT_EQ(r->shape[1], 12, "reshape auto dim == 12");
    arix_tensor_destroy(t); arix_tensor_destroy(r);
}

static void test_reshape_null(void) {
    size_t ns[] = {6};
    ASSERT_NULL(arix_tensor_reshape(NULL, ns, 1), "reshape NULL -> NULL");
}

static void test_permute_2d(void) {
    size_t sh[] = {2, 3};
    ArixTensor* t = arix_tensor_ones(sh, 2, ARIX_FLOAT32);
    ASSERT_NOT_NULL(t, "permute src not null");
    size_t axes[] = {1, 0};
    ArixTensor* r = arix_tensor_permute(t, axes);
    ASSERT_NOT_NULL(r, "permute not null");
    ASSERT_EQ(r->shape[0], 3, "permute shape[0] == 3");
    ASSERT_EQ(r->shape[1], 2, "permute shape[1] == 2");
    arix_tensor_destroy(t); arix_tensor_destroy(r);
}

static void test_permute_null(void) {
    size_t axes[] = {1, 0};
    ASSERT_NULL(arix_tensor_permute(NULL, axes), "permute NULL -> NULL");
}

static void test_expand_broadcast(void) {
    ArixTensor* t = arix_tensor_arange(0.0f, 3.0f, 1.0f, ARIX_FLOAT32);
    ASSERT_NOT_NULL(t, "expand src not null");
    size_t ns[] = {2, 3};
    ArixTensor* r = arix_tensor_expand(t, ns, 2);
    ASSERT_NOT_NULL(r, "expand not null");
    ASSERT_EQ(r->shape[0], 2, "expand shape[0] == 2");
    ASSERT_EQ(r->shape[1], 3, "expand shape[1] == 3");
    arix_tensor_destroy(t); arix_tensor_destroy(r);
}

static void test_expand_null(void) {
    size_t ns[] = {2, 3};
    ASSERT_NULL(arix_tensor_expand(NULL, ns, 2), "expand NULL -> NULL");
}

static void test_expand_fewer_dims(void) {
    size_t shape[] = {2, 3};
    ArixTensor* t = arix_tensor_ones(shape, 2, ARIX_FLOAT32);
    size_t ns[] = {6};
    ASSERT_NULL(arix_tensor_expand(t, ns, 1), "expand fewer dims -> NULL");
    arix_tensor_destroy(t);
}

static void test_squeeze_specific(void) {
    size_t sh_[] = {2, 1, 3};
    ArixTensor* t = arix_tensor_ones(sh_, 3, ARIX_FLOAT32);
    ASSERT_NOT_NULL(t, "squeeze src not null");
    ArixTensor* r = arix_tensor_squeeze(t, 1);
    ASSERT_NOT_NULL(r, "squeeze not null");
    ASSERT_EQ(r->ndim, 2, "squeeze ndim == 2");
    ASSERT_EQ(r->shape[0], 2, "squeeze shape[0] == 2");
    ASSERT_EQ(r->shape[1], 3, "squeeze shape[1] == 3");
    arix_tensor_destroy(t); arix_tensor_destroy(r);
}

static void test_squeeze_not_one(void) {
    size_t shape[] = {2, 3};
    ArixTensor* t = arix_tensor_ones(shape, 2, ARIX_FLOAT32);
    ArixTensor* r = arix_tensor_squeeze(t, 0);
    ASSERT_NOT_NULL(r, "squeeze non-1 dim returns copy");
    ASSERT_EQ(r->ndim, 2, "squeeze non-1 ndim unchanged");
    arix_tensor_destroy(t); if (r) arix_tensor_destroy(r);
}

static void test_squeeze_invalid_dim(void) {
    size_t shape[] = {2, 3};
    ArixTensor* t = arix_tensor_ones(shape, 2, ARIX_FLOAT32);
    ArixTensor* r = arix_tensor_squeeze(t, 5);
    ASSERT_NOT_NULL(r, "squeeze invalid dim returns copy");
    arix_tensor_destroy(t); if (r) arix_tensor_destroy(r);
}

static void test_unsqueeze_front(void) {
    ArixTensor* t = arix_tensor_arange(0.0f, 3.0f, 1.0f, ARIX_FLOAT32);
    ASSERT_NOT_NULL(t, "unsqueeze src not null");
    ArixTensor* r = arix_tensor_unsqueeze(t, 0);
    ASSERT_NOT_NULL(r, "unsqueeze not null");
    ASSERT_EQ(r->ndim, 2, "unsqueeze ndim == 2");
    ASSERT_EQ(r->shape[0], 1, "unsqueeze shape[0] == 1");
    ASSERT_EQ(r->shape[1], 3, "unsqueeze shape[1] == 3");
    arix_tensor_destroy(t); arix_tensor_destroy(r);
}

static void test_unsqueeze_invalid_dim(void) {
    size_t shape[] = {3};
    ArixTensor* t = arix_tensor_ones(shape, 1, ARIX_FLOAT32);
    ASSERT_NULL(arix_tensor_unsqueeze(t, 5), "unsqueeze dim>ndim -> NULL");
    arix_tensor_destroy(t);
}

static void test_concat_dim0(void) {
    size_t sh[] = {2, 3};
    ArixTensor* a = arix_tensor_ones(sh, 2, ARIX_FLOAT32);
    ArixTensor* b = arix_tensor_ones(sh, 2, ARIX_FLOAT32);
    ArixTensor* arr[] = {a, b};
    ArixTensor* r = arix_tensor_concat(arr, 2, 0);
    ASSERT_NOT_NULL(r, "concat dim0 not null");
    ASSERT_EQ(r->shape[0], 4, "concat dim0 shape[0] == 4");
    ASSERT_EQ(r->shape[1], 3, "concat dim0 shape[1] == 3");
    arix_tensor_destroy(a); arix_tensor_destroy(b); arix_tensor_destroy(r);
}

static void test_concat_dim1(void) {
    size_t sh[] = {2, 3};
    ArixTensor* a = arix_tensor_ones(sh, 2, ARIX_FLOAT32);
    ArixTensor* b = arix_tensor_ones(sh, 2, ARIX_FLOAT32);
    ArixTensor* arr[] = {a, b};
    ArixTensor* r = arix_tensor_concat(arr, 2, 1);
    ASSERT_NOT_NULL(r, "concat dim1 not null");
    ASSERT_EQ(r->shape[0], 2, "concat dim1 shape[0] == 2");
    ASSERT_EQ(r->shape[1], 6, "concat dim1 shape[1] == 6");
    arix_tensor_destroy(a); arix_tensor_destroy(b); arix_tensor_destroy(r);
}

static void test_concat_null(void) {
    ASSERT_NULL(arix_tensor_concat(NULL, 0, 0), "concat NULL -> NULL");
}

static void test_split_equal(void) {
    size_t sh[] = {4, 2};
    ArixTensor* t = arix_tensor_ones(sh, 2, ARIX_FLOAT32);
    ArixTensor** parts = arix_tensor_split(t, 2, 0);
    ASSERT_NOT_NULL(parts, "split not null");
    ASSERT_EQ(parts[0]->shape[0], 2, "split part0 shape[0] == 2");
    ASSERT_EQ(parts[1]->shape[0], 2, "split part1 shape[0] == 2");
    arix_tensor_destroy(t);
    arix_tensor_destroy(parts[0]); arix_tensor_destroy(parts[1]);
    arix_free(parts, 2 * sizeof(ArixTensor*));
}

static void test_split_null(void) {
    ASSERT_NULL(arix_tensor_split(NULL, 2, 0), "split NULL -> NULL");
}

static void test_slice_basic(void) {
    size_t sh[] = {4, 3};
    ArixTensor* t = arix_tensor_ones(sh, 2, ARIX_FLOAT32);
    ASSERT_NOT_NULL(t, "slice src not null");
    ArixTensor* r = arix_tensor_slice(t, 0, 1, 3);
    ASSERT_NOT_NULL(r, "slice not null");
    ASSERT_EQ(r->shape[0], 2, "slice shape[0] == 2");
    ASSERT_EQ(r->shape[1], 3, "slice shape[1] == 3");
    arix_tensor_destroy(t); arix_tensor_destroy(r);
}

static void test_slice_null(void) {
    ASSERT_NULL(arix_tensor_slice(NULL, 0, 0, 1), "slice NULL -> NULL");
}

static void test_masked_select_basic(void) {
    size_t sh_[] = {5};
    ArixTensor* t = arix_tensor_arange(0.0f, 5.0f, 1.0f, ARIX_FLOAT32);
    ASSERT_NOT_NULL(t, "masked select src not null");
    uint8_t mask_data[] = {1, 0, 1, 0, 1};
    ArixTensor* mask = arix_tensor_empty(sh_, 1, ARIX_BOOL);
    ASSERT_NOT_NULL(mask, "mask not null");
    memcpy(mask->data, mask_data, 5);
    ArixTensor* r = arix_tensor_masked_select(t, mask);
    ASSERT_NOT_NULL(r, "masked select result not null");
    ASSERT_EQ(r->size, 3, "masked select size == 3");
    float* d = (float*)r->data;
    ASSERT_NEAR(d[0], 0.0f, 1e-6f, "masked select [0]");
    ASSERT_NEAR(d[1], 2.0f, 1e-6f, "masked select [1]");
    ASSERT_NEAR(d[2], 4.0f, 1e-6f, "masked select [2]");
    arix_tensor_destroy(t); arix_tensor_destroy(mask); arix_tensor_destroy(r);
}

static void test_masked_fill_basic(void) {
    size_t sh_[] = {5};
    ArixTensor* t = arix_tensor_arange(0.0f, 5.0f, 1.0f, ARIX_FLOAT32);
    ASSERT_NOT_NULL(t, "masked fill src not null");
    uint8_t mask_data[] = {1, 0, 0, 0, 1};
    ArixTensor* mask = arix_tensor_empty(sh_, 1, ARIX_BOOL);
    ASSERT_NOT_NULL(mask, "mask not null");
    memcpy(mask->data, mask_data, 5);
    float fill_val = -1.0f;
    ArixTensor* r = arix_tensor_masked_fill(t, mask, &fill_val);
    ASSERT_NOT_NULL(r, "masked fill result not null");
    float* d = (float*)r->data;
    ASSERT_NEAR(d[0], -1.0f, 1e-6f, "masked fill [0]");
    ASSERT_NEAR(d[1], 1.0f, 1e-6f, "masked fill [1]");
    ASSERT_NEAR(d[4], -1.0f, 1e-6f, "masked fill [4]");
    ASSERT_EQ(r, t, "masked fill returns src");
    arix_tensor_destroy(t); arix_tensor_destroy(mask);
}

static void test_where_basic(void) {
    size_t sh_[] = {4};
    ArixTensor* cond = arix_tensor_empty(sh_, 1, ARIX_BOOL);
    uint8_t cond_data[] = {1, 0, 1, 0};
    memcpy(cond->data, cond_data, 4);
    float xv[] = {10, 20, 30, 40};
    float yv[] = {-1, -2, -3, -4};
    ArixTensor* x = arix_tensor_empty(sh_, 1, ARIX_FLOAT32);
    ArixTensor* y = arix_tensor_empty(sh_, 1, ARIX_FLOAT32);
    memcpy(x->data, xv, 4 * sizeof(float));
    memcpy(y->data, yv, 4 * sizeof(float));
    ArixTensor* r = arix_tensor_where(cond, x, y);
    ASSERT_NOT_NULL(r, "where not null");
    float* d = (float*)r->data;
    ASSERT_NEAR(d[0], 10.0f, 1e-6f, "where [0]");
    ASSERT_NEAR(d[1], -2.0f, 1e-6f, "where [1]");
    ASSERT_NEAR(d[2], 30.0f, 1e-6f, "where [2]");
    ASSERT_NEAR(d[3], -4.0f, 1e-6f, "where [3]");
    arix_tensor_destroy(cond); arix_tensor_destroy(x); arix_tensor_destroy(y); arix_tensor_destroy(r);
}

static void test_gather_basic(void) {
    size_t sh_[] = {3, 3};
    ArixTensor* t = arix_tensor_ones(sh_, 2, ARIX_FLOAT32);
    ASSERT_NOT_NULL(t, "gather src not null");
    size_t idx_sh[] = {2, 3};
    int32_t idx_data[] = {0, 1, 2, 2, 1, 0};
    ArixTensor* idx = arix_tensor_empty(idx_sh, 2, ARIX_INT32);
    memcpy(idx->data, idx_data, 6 * sizeof(int32_t));
    ArixTensor* r = arix_tensor_gather(t, 0, idx);
    ASSERT_NOT_NULL(r, "gather not null");
    ASSERT_EQ(r->shape[0], 2, "gather shape[0] == 2");
    arix_tensor_destroy(t); arix_tensor_destroy(idx); arix_tensor_destroy(r);
}

static void test_tile_basic(void) {
    size_t sh_[] = {1, 2};
    ArixTensor* t = arix_tensor_ones(sh_, 2, ARIX_FLOAT32);
    ASSERT_NOT_NULL(t, "tile src not null");
    size_t reps[] = {2, 2};
    ArixTensor* r = arix_tensor_tile(t, reps, 2);
    ASSERT_NOT_NULL(r, "tile not null");
    ASSERT_EQ(r->shape[0], 2, "tile shape[0] == 2");
    ASSERT_EQ(r->shape[1], 4, "tile shape[1] == 4");
    arix_tensor_destroy(t); arix_tensor_destroy(r);
}

static void test_repeat_basic(void) {
    ArixTensor* t = arix_tensor_arange(0.0f, 3.0f, 1.0f, ARIX_FLOAT32);
    ASSERT_NOT_NULL(t, "repeat src not null");
    ArixTensor* r = arix_tensor_repeat(t, 2, 0);
    ASSERT_NOT_NULL(r, "repeat not null");
    ASSERT_EQ(r->shape[0], 6, "repeat shape[0] == 6");
    arix_tensor_destroy(t); arix_tensor_destroy(r);
}

int main(void) {
    run_test("copy independent", test_copy_independent);
    run_test("copy NULL", test_copy_null);
    run_test("clone NULL", test_clone_null);
    run_test("reshape 2d->1d", test_reshape_2d_to_1d);
    run_test("reshape auto dim", test_reshape_auto_dim);
    run_test("reshape NULL", test_reshape_null);
    run_test("permute 2d", test_permute_2d);
    run_test("permute NULL", test_permute_null);
    run_test("expand broadcast", test_expand_broadcast);
    run_test("expand NULL", test_expand_null);
    run_test("expand fewer dims", test_expand_fewer_dims);
    run_test("squeeze specific", test_squeeze_specific);
    run_test("squeeze not one", test_squeeze_not_one);
    run_test("squeeze invalid dim", test_squeeze_invalid_dim);
    run_test("unsqueeze front", test_unsqueeze_front);
    run_test("unsqueeze invalid dim", test_unsqueeze_invalid_dim);
    run_test("concat dim0", test_concat_dim0);
    run_test("concat dim1", test_concat_dim1);
    run_test("concat NULL", test_concat_null);
    run_test("split equal", test_split_equal);
    run_test("split NULL", test_split_null);
    run_test("slice basic", test_slice_basic);
    run_test("slice NULL", test_slice_null);
    run_test("masked select basic", test_masked_select_basic);
    run_test("masked fill basic", test_masked_fill_basic);
    run_test("where basic", test_where_basic);
    run_test("gather basic", test_gather_basic);
    run_test("tile basic", test_tile_basic);
    run_test("repeat basic", test_repeat_basic);
    RUN_ALL_TESTS();
}
