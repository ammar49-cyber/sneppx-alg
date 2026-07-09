#include "test_common.h"
#include "multidimensional_tensor_engine.h"

static void test_empty_2d(void) {
    size_t shape[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_empty(shape, 2, SNEPPX_FLOAT32);
    ASSERT_NOT_NULL(t, "empty 2d not null");
    ASSERT_EQ(t->ndim, 2, "ndim == 2");
    ASSERT_EQ(t->shape[0], 2, "shape[0] == 2");
    ASSERT_EQ(t->shape[1], 3, "shape[1] == 3");
    ASSERT_EQ(t->size, 6, "size == 6");
    ASSERT_EQ(t->dtype, SNEPPX_FLOAT32, "dtype == float32");
    ASSERT_NOT_NULL(t->data, "data not null");
    SNEPPX_tensor_destroy(t);
}

static void test_empty_3d(void) {
    size_t shape[] = {2, 3, 4};
    SNEPPXTensor* t = SNEPPX_tensor_empty(shape, 3, SNEPPX_FLOAT64);
    ASSERT_NOT_NULL(t, "empty 3d not null");
    ASSERT_EQ(t->ndim, 3, "ndim == 3");
    ASSERT_EQ(t->size, 24, "size == 24");
    ASSERT_EQ(t->dtype, SNEPPX_FLOAT64, "dtype == float64");
    SNEPPX_tensor_destroy(t);
}

static void test_empty_null_shape(void) {
    SNEPPXTensor* t = SNEPPX_tensor_empty(NULL, 2, SNEPPX_FLOAT32);
    ASSERT_NULL(t, "empty NULL shape -> NULL");
}

static void test_empty_zero_ndim(void) {
    SNEPPXTensor* t = SNEPPX_tensor_empty(NULL, 0, SNEPPX_FLOAT32);
    if (t) { ASSERT_EQ(t->ndim, 0, "ndim == 0"); ASSERT_EQ(t->size, 1, "size == 1"); SNEPPX_tensor_destroy(t); }
}

static void test_full_f32(void) {
    size_t shape[] = {2, 3};
    float val = 3.14f;
    SNEPPXTensor* t = SNEPPX_tensor_full(shape, 2, SNEPPX_FLOAT32, &val);
    ASSERT_NOT_NULL(t, "full f32 not null");
    float* d = (float*)t->data;
    for (size_t i = 0; i < 6; i++) ASSERT_NEAR(d[i], 3.14f, 1e-4f, "full f32 value");
    SNEPPX_tensor_destroy(t);
}

static void test_full_f64(void) {
    size_t shape[] = {3};
    double val = 2.71;
    SNEPPXTensor* t = SNEPPX_tensor_full(shape, 1, SNEPPX_FLOAT64, &val);
    ASSERT_NOT_NULL(t, "full f64 not null");
    double* d = (double*)t->data;
    for (size_t i = 0; i < 3; i++) ASSERT_NEAR(d[i], 2.71, 1e-4, "full f64 value");
    SNEPPX_tensor_destroy(t);
}

static void test_full_null_value(void) {
    size_t shape[] = {2, 2};
    SNEPPXTensor* t = SNEPPX_tensor_full(shape, 2, SNEPPX_FLOAT32, NULL);
    ASSERT_NOT_NULL(t, "full NULL value returns tensor");
    SNEPPX_tensor_destroy(t);
}

static void test_full_int32(void) {
    size_t shape[] = {4};
    int32_t val = 42;
    SNEPPXTensor* t = SNEPPX_tensor_full(shape, 1, SNEPPX_INT32, &val);
    ASSERT_NOT_NULL(t, "full int32 not null");
    int32_t* d = (int32_t*)t->data;
    for (size_t i = 0; i < 4; i++) ASSERT_EQ(d[i], 42, "full int32 value");
    SNEPPX_tensor_destroy(t);
}

static void test_arange_f32(void) {
    SNEPPXTensor* t = SNEPPX_tensor_arange(0.0f, 5.0f, 1.0f, SNEPPX_FLOAT32);
    ASSERT_NOT_NULL(t, "arange f32 not null");
    ASSERT_EQ(t->size, 5, "arange size == 5");
    float* d = (float*)t->data;
    for (size_t i = 0; i < 5; i++) ASSERT_NEAR(d[i], (float)i, 1e-4f, "arange f32 value");
    SNEPPX_tensor_destroy(t);
}

static void test_arange_f64(void) {
    SNEPPXTensor* t = SNEPPX_tensor_arange(1.0f, 10.0f, 3.0f, SNEPPX_FLOAT64);
    ASSERT_NOT_NULL(t, "arange f64 not null");
    double* d = (double*)t->data;
    ASSERT_NEAR(d[0], 1.0, 1e-4, "arange f64 [0]");
    ASSERT_NEAR(d[1], 4.0, 1e-4, "arange f64 [1]");
    ASSERT_NEAR(d[2], 7.0, 1e-4, "arange f64 [2]");
    ASSERT_EQ(t->size, 3, "arange f64 size == 3");
    SNEPPX_tensor_destroy(t);
}

static void test_arange_int32(void) {
    SNEPPXTensor* t = SNEPPX_tensor_arange(0.0f, 5.0f, 1.0f, SNEPPX_INT32);
    ASSERT_NOT_NULL(t, "arange int32 not null");
    ASSERT_EQ(t->size, 5, "arange int32 size == 5");
    int32_t* d = (int32_t*)t->data;
    for (size_t i = 0; i < 5; i++) ASSERT_EQ(d[i], (int32_t)i, "arange int32 value");
    SNEPPX_tensor_destroy(t);
}

static void test_arange_zero_step(void) {
    SNEPPXTensor* t = SNEPPX_tensor_arange(0.0f, 5.0f, 0.0f, SNEPPX_FLOAT32);
    ASSERT_NULL(t, "arange zero step -> NULL");
}

static void test_arange_negative_step(void) {
    SNEPPXTensor* t = SNEPPX_tensor_arange(5.0f, 0.0f, -1.0f, SNEPPX_FLOAT32);
    ASSERT_NOT_NULL(t, "arange negative step not null");
    ASSERT_EQ(t->size, 5, "arange negative step size == 5");
    float* d = (float*)t->data;
    ASSERT_NEAR(d[0], 5.0f, 1e-4f, "arange negative [0]");
    ASSERT_NEAR(d[4], 1.0f, 1e-4f, "arange negative [4]");
    SNEPPX_tensor_destroy(t);
}

static void test_linspace_f32(void) {
    SNEPPXTensor* t = SNEPPX_tensor_linspace(0.0f, 1.0f, 5, SNEPPX_FLOAT32);
    ASSERT_NOT_NULL(t, "linspace f32 not null");
    ASSERT_EQ(t->size, 5, "linspace size == 5");
    float* d = (float*)t->data;
    ASSERT_NEAR(d[0], 0.0f, 1e-4f, "linspace f32 start");
    ASSERT_NEAR(d[4], 1.0f, 1e-4f, "linspace f32 end");
    ASSERT_NEAR(d[2], 0.5f, 1e-4f, "linspace f32 mid");
    SNEPPX_tensor_destroy(t);
}

static void test_linspace_f64(void) {
    SNEPPXTensor* t = SNEPPX_tensor_linspace(1.0f, 3.0f, 3, SNEPPX_FLOAT64);
    ASSERT_NOT_NULL(t, "linspace f64 not null");
    double* d = (double*)t->data;
    ASSERT_NEAR(d[0], 1.0, 1e-4, "linspace f64 [0]");
    ASSERT_NEAR(d[1], 2.0, 1e-4, "linspace f64 [1]");
    ASSERT_NEAR(d[2], 3.0, 1e-4, "linspace f64 [2]");
    SNEPPX_tensor_destroy(t);
}

static void test_linspace_zero_steps(void) {
    SNEPPXTensor* t = SNEPPX_tensor_linspace(0.0f, 1.0f, 0, SNEPPX_FLOAT32);
    ASSERT_NULL(t, "linspace zero steps -> NULL");
}

static void test_eye_2d(void) {
    SNEPPXTensor* t = SNEPPX_tensor_eye(3, SNEPPX_FLOAT32);
    ASSERT_NOT_NULL(t, "eye 3 not null");
    ASSERT_EQ(t->ndim, 2, "eye ndim == 2");
    ASSERT_EQ(t->shape[0], 3, "eye shape[0] == 3");
    ASSERT_EQ(t->shape[1], 3, "eye shape[1] == 3");
    float* d = (float*)t->data;
    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < 3; j++)
            ASSERT_NEAR(d[i * 3 + j], (i == j) ? 1.0f : 0.0f, 1e-6f, "eye value");
    SNEPPX_tensor_destroy(t);
}

static void test_eye_1(void) {
    SNEPPXTensor* t = SNEPPX_tensor_eye(1, SNEPPX_FLOAT32);
    ASSERT_NOT_NULL(t, "eye 1 not null");
    float* d = (float*)t->data;
    ASSERT_NEAR(d[0], 1.0f, 1e-6f, "eye 1x1 = 1");
    SNEPPX_tensor_destroy(t);
}

static void test_eye_f64(void) {
    SNEPPXTensor* t = SNEPPX_tensor_eye(2, SNEPPX_FLOAT64);
    ASSERT_NOT_NULL(t, "eye f64 not null");
    double* d = (double*)t->data;
    ASSERT_NEAR(d[0], 1.0, 1e-6, "eye f64 [0,0]");
    ASSERT_NEAR(d[1], 0.0, 1e-6, "eye f64 [0,1]");
    ASSERT_NEAR(d[2], 0.0, 1e-6, "eye f64 [1,0]");
    ASSERT_NEAR(d[3], 1.0, 1e-6, "eye f64 [1,1]");
    SNEPPX_tensor_destroy(t);
}

int main(void) {
    run_test("empty 2d", test_empty_2d);
    run_test("empty 3d", test_empty_3d);
    run_test("empty NULL shape", test_empty_null_shape);
    run_test("empty zero ndim", test_empty_zero_ndim);
    run_test("full f32", test_full_f32);
    run_test("full f64", test_full_f64);
    run_test("full NULL value", test_full_null_value);
    run_test("full int32", test_full_int32);
    run_test("arange f32", test_arange_f32);
    run_test("arange f64", test_arange_f64);
    run_test("arange int32", test_arange_int32);
    run_test("arange zero step", test_arange_zero_step);
    run_test("arange negative step", test_arange_negative_step);
    run_test("linspace f32", test_linspace_f32);
    run_test("linspace f64", test_linspace_f64);
    run_test("linspace zero steps", test_linspace_zero_steps);
    run_test("eye 3x3", test_eye_2d);
    run_test("eye 1x1", test_eye_1);
    run_test("eye f64", test_eye_f64);
    RUN_ALL_TESTS();
}
