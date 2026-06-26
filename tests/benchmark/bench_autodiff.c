#include "bench_common.h"
#include "arix_autodiff.h"

static void bench_variable_ops(void) {
    BENCH_INIT(bs);
    size_t sh[] = {256, 256};
    ArixTensor* t = arix_tensor_ones(sh, 2, ARIX_FLOAT32);
    printf("  Variable ops [256,256]:\n");

    BENCH_START(bs, 10000, 500, {
        ArixVariable* v = arix_variable_create(t, 1);
        v->data = NULL;
        arix_variable_destroy(v);
    });
    bench_print("variable_create/destroy", &bs);

    arix_tensor_destroy(t);
}

static void bench_add_backward(void) {
    BENCH_INIT(bs);
    size_t sh[] = {256, 256};
    ArixTensor* ta = arix_tensor_ones(sh, 2, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_ones(sh, 2, ARIX_FLOAT32);
    printf("  Add backward [256,256]:\n");

    BENCH_START(bs, 500, 50, {
        ArixVariable* a = arix_variable_create(ta, 1);
        ArixVariable* b = arix_variable_create(tb, 1);
        ArixTape* tape = arix_tape_create();
        ArixVariable* c = arix_add(tape, a, b);
        arix_tape_backward(tape, c);
        a->data = NULL; a->grad = NULL;
        b->data = NULL; b->grad = NULL;
        c->data = NULL; c->grad = NULL;
        arix_tape_destroy(tape);
    });
    bench_print("add + backward", &bs);

    arix_tensor_destroy(ta); arix_tensor_destroy(tb);
}

static void bench_mul_backward(void) {
    BENCH_INIT(bs);
    size_t sh[] = {256, 256};
    ArixTensor* ta = arix_tensor_ones(sh, 2, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_ones(sh, 2, ARIX_FLOAT32);
    printf("  Mul backward [256,256]:\n");

    BENCH_START(bs, 500, 50, {
        ArixVariable* a = arix_variable_create(ta, 1);
        ArixVariable* b = arix_variable_create(tb, 1);
        ArixTape* tape = arix_tape_create();
        ArixVariable* c = arix_mul(tape, a, b);
        arix_tape_backward(tape, c);
        a->data = NULL; a->grad = NULL;
        b->data = NULL; b->grad = NULL;
        c->data = NULL; c->grad = NULL;
        arix_tape_destroy(tape);
    });
    bench_print("mul + backward", &bs);

    arix_tensor_destroy(ta); arix_tensor_destroy(tb);
}

static void bench_matmul_backward(void) {
    BENCH_INIT(bs);
    size_t sh1[] = {64, 128};
    size_t sh2[] = {128, 64};
    ArixTensor* ta = arix_tensor_ones(sh1, 2, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_ones(sh2, 2, ARIX_FLOAT32);
    printf("  Matmul backward [64,128]x[128,64]:\n");

    BENCH_START(bs, 200, 20, {
        ArixVariable* a = arix_variable_create(ta, 1);
        ArixVariable* b = arix_variable_create(tb, 1);
        ArixTape* tape = arix_tape_create();
        ArixVariable* c = arix_matmul(tape, a, b);
        arix_tape_backward(tape, c);
        a->data = NULL; a->grad = NULL;
        b->data = NULL; b->grad = NULL;
        c->data = NULL; c->grad = NULL;
        arix_tape_destroy(tape);
    });
    bench_print("matmul + backward", &bs);

    arix_tensor_destroy(ta); arix_tensor_destroy(tb);
}

static void bench_chain_rule(void) {
    BENCH_INIT(bs);
    size_t sh[] = {64, 64};
    ArixTensor* ta = arix_tensor_ones(sh, 2, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_ones(sh, 2, ARIX_FLOAT32);
    printf("  Chain rule [64,64]:\n");

    BENCH_START(bs, 200, 20, {
        ArixVariable* a = arix_variable_create(ta, 1);
        ArixVariable* b = arix_variable_create(tb, 1);
        ArixTape* tape = arix_tape_create();
        ArixVariable* c = arix_add(tape, a, b);
        ArixVariable* d = arix_mul(tape, c, a);
        ArixVariable* e = arix_relu(tape, d);
        arix_tape_backward(tape, e);
        a->data = NULL; a->grad = NULL;
        b->data = NULL; b->grad = NULL;
        c->data = NULL; c->grad = NULL;
        d->data = NULL; d->grad = NULL;
        e->data = NULL; e->grad = NULL;
        arix_tape_destroy(tape);
    });
    bench_print("add->mul->relu + backward", &bs);

    arix_tensor_destroy(ta); arix_tensor_destroy(tb);
}

static void bench_relu_sigmoid_backward(void) {
    BENCH_INIT(bs);
    size_t sh[] = {256, 256};
    ArixTensor* ta = arix_tensor_ones(sh, 2, ARIX_FLOAT32);
    printf("  Activation backward [256,256]:\n");

    BENCH_START(bs, 500, 50, {
        ArixVariable* a = arix_variable_create(ta, 1);
        ArixTape* tape = arix_tape_create();
        ArixVariable* c = arix_relu(tape, a);
        arix_tape_backward(tape, c);
        a->data = NULL; a->grad = NULL;
        c->data = NULL; c->grad = NULL;
        arix_tape_destroy(tape);
    });
    bench_print("relu + backward", &bs);

    BENCH_START(bs, 500, 50, {
        ArixVariable* a = arix_variable_create(ta, 1);
        ArixTape* tape = arix_tape_create();
        ArixVariable* c = arix_sigmoid(tape, a);
        arix_tape_backward(tape, c);
        a->data = NULL; a->grad = NULL;
        c->data = NULL; c->grad = NULL;
        arix_tape_destroy(tape);
    });
    bench_print("sigmoid + backward", &bs);

    arix_tensor_destroy(ta);
}

static void bench_mse_loss_backward(void) {
    BENCH_INIT(bs);
    size_t sh[] = {64, 64};
    ArixTensor* ta = arix_tensor_ones(sh, 2, ARIX_FLOAT32);
    ArixTensor* tb = arix_tensor_zeros(sh, 2, ARIX_FLOAT32);
    printf("  MSE loss backward [64,64]:\n");

    BENCH_START(bs, 500, 50, {
        ArixVariable* a = arix_variable_create(ta, 1);
        ArixVariable* b = arix_variable_create(tb, 0);
        ArixTape* tape = arix_tape_create();
        ArixVariable* c = arix_mse_loss(tape, a, b);
        arix_tape_backward(tape, c);
        a->data = NULL; a->grad = NULL;
        c->data = NULL; c->grad = NULL;
        arix_tape_destroy(tape);
    });
    bench_print("mse_loss + backward", &bs);

    arix_tensor_destroy(ta); arix_tensor_destroy(tb);
}

int main(void) {
    printf("=== Autodiff Benchmarks ===\n");
    BENCH_RUN("Variable ops", bench_variable_ops);
    BENCH_RUN("Add backward", bench_add_backward);
    BENCH_RUN("Mul backward", bench_mul_backward);
    BENCH_RUN("Matmul backward", bench_matmul_backward);
    BENCH_RUN("Chain rule", bench_chain_rule);
    BENCH_RUN("Activation backward", bench_relu_sigmoid_backward);
    BENCH_RUN("MSE loss backward", bench_mse_loss_backward);
    BENCH_MAIN();
}
