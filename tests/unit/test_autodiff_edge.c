#include "test_common.h"
#include "automatic_differentiation_framework.h"

static void test_variable_null_data(void) {
    SNEPPXVariable* v = SNEPPX_variable_create(NULL, 1);
    ASSERT_NOT_NULL(v, "var with NULL data is allowed");
    ASSERT_NULL(v->data, "data is NULL");
    ASSERT_EQ(v->requires_grad, 1, "requires_grad set");
    v->data = NULL;
    SNEPPX_variable_destroy(v);
}

static void test_variable_no_grad(void) {
    size_t sh[] = {3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* v = SNEPPX_variable_create(t, 0);
    ASSERT_NOT_NULL(v, "var requires_grad=0");
    ASSERT_EQ(v->requires_grad, 0, "requires_grad == 0");
    ASSERT_NULL(v->grad, "grad is NULL");
    SNEPPX_variable_destroy(v);
}

static void test_tape_destroy_null(void) {
    SNEPPX_tape_destroy(NULL);
}

static void test_variable_destroy_null(void) {
    SNEPPX_variable_destroy(NULL);
}

static void test_add_null_vars(void) {
    size_t sh[] = {3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_add(tape, a, NULL);
    ASSERT_NULL(c, "add with NULL b");
    c = SNEPPX_add(tape, NULL, a);
    ASSERT_NULL(c, "add with NULL a");
    c = SNEPPX_add(tape, NULL, NULL);
    ASSERT_NULL(c, "add NULL NULL");
    SNEPPX_tape_destroy(tape);
    a->data = NULL;
    SNEPPX_variable_destroy(a);
}

static void test_sub_null_vars(void) {
    size_t sh[] = {3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_sub(tape, a, NULL);
    ASSERT_NULL(c, "sub with NULL");
    SNEPPX_tape_destroy(tape);
    a->data = NULL;
    SNEPPX_variable_destroy(a);
}

static void test_mul_null_vars(void) {
    size_t sh[] = {3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_mul(tape, a, NULL);
    ASSERT_NULL(c, "mul with NULL");
    SNEPPX_tape_destroy(tape);
    a->data = NULL;
    SNEPPX_variable_destroy(a);
}

static void test_matmul_null_vars(void) {
    size_t sh[] = {2, 3};
    size_t sh2[] = {3, 4};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_ones(sh2, 2, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_matmul(tape, a, NULL);
    ASSERT_NULL(c, "matmul NULL b");
    c = SNEPPX_matmul(tape, NULL, b);
    ASSERT_NULL(c, "matmul NULL a");
    SNEPPX_tape_destroy(tape);
    a->data = NULL; b->data = NULL;
    SNEPPX_variable_destroy(a); SNEPPX_variable_destroy(b);
}

static void test_matmul_mismatched_dims(void) {
    size_t sh1[] = {2, 3};
    size_t sh2[] = {5, 4};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh1, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_ones(sh2, 2, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_matmul(tape, a, b);
    ASSERT_NULL(c, "matmul mismatched inner");
    SNEPPX_tape_destroy(tape);
    a->data = NULL; b->data = NULL;
    SNEPPX_variable_destroy(a); SNEPPX_variable_destroy(b);
}

static void test_mse_loss_null(void) {
    size_t sh[] = {3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_mse_loss(tape, a, NULL);
    ASSERT_NULL(c, "mse_loss NULL target");
    c = SNEPPX_mse_loss(tape, NULL, a);
    ASSERT_NULL(c, "mse_loss NULL pred");
    SNEPPX_tape_destroy(tape);
    a->data = NULL;
    SNEPPX_variable_destroy(a);
}

static void test_mse_loss_mismatched(void) {
    size_t sh1[] = {3};
    size_t sh2[] = {4};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh1, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_ones(sh2, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_mse_loss(tape, a, b);
    ASSERT_NULL(c, "mse_loss mismatched sizes");
    SNEPPX_tape_destroy(tape);
    a->data = NULL; b->data = NULL;
    SNEPPX_variable_destroy(a); SNEPPX_variable_destroy(b);
}

static void test_relu_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_relu(tape, NULL);
    ASSERT_NULL(c, "relu NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_gelu_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_gelu(tape, NULL);
    ASSERT_NULL(c, "gelu NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_sigmoid_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_sigmoid(tape, NULL);
    ASSERT_NULL(c, "sigmoid NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_silu_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_silu(tape, NULL);
    ASSERT_NULL(c, "silu NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_div_null(void) {
    size_t sh[] = {3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_div(tape, a, NULL);
    ASSERT_NULL(c, "div NULL b");
    c = SNEPPX_div(tape, NULL, a);
    ASSERT_NULL(c, "div NULL a");
    SNEPPX_tape_destroy(tape);
    a->data = NULL;
    SNEPPX_variable_destroy(a);
}

static void test_pow_null(void) {
    size_t sh[] = {3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_pow(tape, a, NULL);
    ASSERT_NULL(c, "pow NULL b");
    c = SNEPPX_pow(tape, NULL, a);
    ASSERT_NULL(c, "pow NULL a");
    SNEPPX_tape_destroy(tape);
    a->data = NULL;
    SNEPPX_variable_destroy(a);
}

static void test_neg_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_neg(tape, NULL);
    ASSERT_NULL(c, "neg NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_tanh_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_tanh(tape, NULL);
    ASSERT_NULL(c, "tanh NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_softmax_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_softmax(tape, NULL, 0);
    ASSERT_NULL(c, "softmax NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_exp_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_exp(tape, NULL);
    ASSERT_NULL(c, "exp NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_log_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_log(tape, NULL);
    ASSERT_NULL(c, "log NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_sum_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_sum(tape, NULL, 0);
    ASSERT_NULL(c, "sum NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_mean_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_mean(tape, NULL, 0);
    ASSERT_NULL(c, "mean NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_transpose_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_transpose(tape, NULL, 0, 1);
    ASSERT_NULL(c, "transpose NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_dropout_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_dropout(tape, NULL, 0.5f, 42);
    ASSERT_NULL(c, "dropout NULL");
    SNEPPX_tape_destroy(tape);
}

static void test_layer_norm_null(void) {
    size_t sh[] = {4};
    SNEPPXTensor* tg = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* g = SNEPPX_variable_create(tg, 0);
    SNEPPXVariable* b = SNEPPX_variable_create(tb, 0);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_layer_norm(tape, NULL, g, b, 1e-5f);
    ASSERT_NULL(c, "layer_norm NULL input");
    SNEPPX_tape_destroy(tape);
    g->data = NULL; b->data = NULL;
    SNEPPX_variable_destroy(g); SNEPPX_variable_destroy(b);
}

static void test_concat_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* c = SNEPPX_concat(tape, NULL, 0, 0);
    ASSERT_NULL(c, "concat NULL array");
    size_t sh[] = {3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* arr[] = {a, NULL};
    c = SNEPPX_concat(tape, arr, 2, 0);
    ASSERT_NULL(c, "concat with NULL var");
    SNEPPX_tape_destroy(tape);
    a->data = NULL;
    SNEPPX_variable_destroy(a);
}

static void test_tape_backward_constant(void) {
    size_t sh[] = {3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 0);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPX_tape_backward(tape, a);
    ASSERT_NOT_NULL(a->grad, "backward sets grad (grad=ones)");
    if (a->grad) {
        float* gd = (float*)a->grad->data;
        ASSERT_EQ(gd[0], 1.0f, "grad == 1");
    }
    SNEPPX_tape_destroy(tape);
    a->data = NULL; a->grad = NULL;
    SNEPPX_variable_destroy(a);
}

static void test_tape_record_null(void) {
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPX_tape_record(tape, NULL);
    SNEPPX_tape_destroy(tape);
}

static void test_no_grad_scoping(void) {
    size_t sh[] = {3};
    SNEPPXTensor* ta = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* tb = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXVariable* a = SNEPPX_variable_create(ta, 1);
    SNEPPXVariable* b = SNEPPX_variable_create(tb, 1);
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPX_no_grad_enter();
    SNEPPXVariable* c = SNEPPX_add(tape, a, b);
    ASSERT_NOT_NULL(c, "add in no_grad mode");
    ASSERT(c->requires_grad == 0 || c->requires_grad == 0, "result requires_grad=0");
    ASSERT_EQ(SNEPPX_no_grad_is_active(), 1, "no_grad active");
    SNEPPX_no_grad_exit();
    ASSERT_EQ(SNEPPX_no_grad_is_active(), 0, "no_grad inactive after exit");
    SNEPPX_tape_destroy(tape);
    a->data = NULL; b->data = NULL;
    if (c) { c->data = NULL; c->grad = NULL; SNEPPX_variable_destroy(c); }
    SNEPPX_variable_destroy(a); SNEPPX_variable_destroy(b);
}

int main(void) {
    run_test("variable null data", test_variable_null_data);
    run_test("variable no grad", test_variable_no_grad);
    run_test("tape destroy null", test_tape_destroy_null);
    run_test("variable destroy null", test_variable_destroy_null);
    run_test("add null vars", test_add_null_vars);
    run_test("sub null vars", test_sub_null_vars);
    run_test("mul null vars", test_mul_null_vars);
    run_test("matmul null vars", test_matmul_null_vars);
    run_test("matmul mismatched dims", test_matmul_mismatched_dims);
    run_test("mse loss null", test_mse_loss_null);
    run_test("mse loss mismatched", test_mse_loss_mismatched);
    run_test("relu null", test_relu_null);
    run_test("gelu null", test_gelu_null);
    run_test("sigmoid null", test_sigmoid_null);
    run_test("tanh null", test_tanh_null);
    run_test("softmax null", test_softmax_null);
    run_test("exp null", test_exp_null);
    run_test("log null", test_log_null);
    run_test("sum null", test_sum_null);
    run_test("mean null", test_mean_null);
    run_test("transpose null", test_transpose_null);
    run_test("dropout null", test_dropout_null);
    run_test("layer_norm null", test_layer_norm_null);
    run_test("concat null", test_concat_null);
    run_test("silu null", test_silu_null);
    run_test("div null", test_div_null);
    run_test("pow null", test_pow_null);
    run_test("neg null", test_neg_null);
    run_test("tape backward constant", test_tape_backward_constant);
    run_test("tape record null", test_tape_record_null);
    run_test("no_grad scoping", test_no_grad_scoping);
    printf("\nResults: %d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
