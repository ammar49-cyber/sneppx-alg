#include "arix_hss.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s (%s)\n", msg, #cond); \
        tests_failed++; \
        return; \
    } \
} while(0)

static void run_test(const char* name, void (*test_fn)(void)) {
    printf("Running %s... ", name);
    fflush(stdout);
    test_fn();
    printf("PASS\n");
    tests_passed++;
}

static void test_layer_create(void) {
    ArixHSSConfig cfg = arix_hss_config_default();
    cfg.state_dim = 4;
    cfg.input_dim = 8;
    cfg.output_dim = 8;
    ArixHSSLayer* layer = arix_hss_layer_create(&cfg, 42);
    ASSERT(layer != NULL, "layer not null");
    ASSERT(layer->A != NULL, "A not null");
    ASSERT(layer->B != NULL, "B not null");
    ASSERT(layer->C != NULL, "C not null");
    ASSERT(layer->D != NULL, "D not null");
    ASSERT(layer->dt != NULL, "dt not null");
    ASSERT(layer->h != NULL, "h not null");
    ASSERT(layer->x_proj != NULL, "x_proj not null");
    ASSERT(layer->x_proj_bias != NULL, "x_proj_bias not null");
    ASSERT(layer->A->shape[0] == 4, "A rows == 4");
    ASSERT(layer->A->shape[1] == 4, "A cols == 4");
    ASSERT(layer->B->shape[0] == 4, "B rows == 4");
    ASSERT(layer->B->shape[1] == 8, "B cols == 8");
    ASSERT(layer->C->shape[0] == 8, "C rows == 8");
    ASSERT(layer->C->shape[1] == 4, "C cols == 4");
    ASSERT(layer->dt->shape[0] == 4, "dt size == 4");
    arix_hss_layer_destroy(layer);
}

static void test_discretize(void) {
    ArixHSSConfig cfg = arix_hss_config_default();
    cfg.state_dim = 4;
    cfg.input_dim = 8;
    cfg.output_dim = 8;
    ArixHSSLayer* layer = arix_hss_layer_create(&cfg, 42);
    ASSERT(layer != NULL, "layer not null");
    arix_hss_discretize(layer);
    ASSERT(layer->A_bar != NULL, "A_bar not null");
    ASSERT(layer->B_bar != NULL, "B_bar not null");
    ASSERT(layer->A_bar->shape[0] == 4, "A_bar rows == 4");
    ASSERT(layer->A_bar->shape[1] == 4, "A_bar cols == 4");
    ASSERT(layer->B_bar->shape[0] == 4, "B_bar rows == 4");
    ASSERT(layer->B_bar->shape[1] == 8, "B_bar cols == 8");
    arix_hss_layer_destroy(layer);
}

static void test_step(void) {
    ArixHSSConfig cfg = arix_hss_config_default();
    cfg.state_dim = 4;
    cfg.input_dim = 8;
    cfg.output_dim = 8;
    ArixHSSLayer* layer = arix_hss_layer_create(&cfg, 42);
    ASSERT(layer != NULL, "layer not null");
    arix_hss_discretize(layer);

    size_t shape_x[] = {8};
    size_t shape_h[] = {4};
    ArixTensor* x = arix_tensor_zeros(shape_x, 1, ARIX_FLOAT32);
    ArixTensor* h_next = arix_tensor_zeros(shape_h, 1, ARIX_FLOAT32);
    ASSERT(x != NULL && h_next != NULL, "tensors not null");
    ((float*)x->data)[0] = 1.0f;

    arix_hss_step(layer, x, h_next);
    float* hd = (float*)h_next->data;
    int finite = 1;
    for (size_t i = 0; i < 4; i++) {
        if (!isfinite(hd[i])) finite = 0;
    }
    ASSERT(finite, "h_next values are finite");
    arix_tensor_destroy(x);
    arix_tensor_destroy(h_next);
    arix_hss_layer_destroy(layer);
}

static void test_scan(void) {
    ArixHSSConfig cfg = arix_hss_config_default();
    cfg.state_dim = 4;
    cfg.input_dim = 8;
    cfg.output_dim = 8;
    cfg.seq_len = 10;
    ArixHSSLayer* layer = arix_hss_layer_create(&cfg, 42);
    ASSERT(layer != NULL, "layer not null");
    arix_hss_discretize(layer);

    size_t shape_xs[] = {10, 8};
    size_t shape_hs[] = {10, 4};
    size_t shape_ys[] = {10, 8};
    ArixTensor* x_seq = arix_tensor_randn(shape_xs, 2, ARIX_FLOAT32);
    ArixTensor* h_seq = arix_tensor_zeros(shape_hs, 2, ARIX_FLOAT32);
    ArixTensor* y_seq = arix_tensor_zeros(shape_ys, 2, ARIX_FLOAT32);
    ASSERT(x_seq && h_seq && y_seq, "scan tensors not null");

    arix_hss_scan(layer, x_seq, h_seq, y_seq);

    ASSERT(h_seq->shape[0] == 10, "h_seq rows == 10");
    ASSERT(h_seq->shape[1] == 4, "h_seq cols == 4");
    ASSERT(y_seq->shape[0] == 10, "y_seq rows == 10");
    ASSERT(y_seq->shape[1] == 8, "y_seq cols == 8");

    arix_tensor_destroy(x_seq);
    arix_tensor_destroy(h_seq);
    arix_tensor_destroy(y_seq);
    arix_hss_layer_destroy(layer);
}

int main(void) {
    run_test("test_layer_create", test_layer_create);
    run_test("test_discretize", test_discretize);
    run_test("test_step", test_step);
    run_test("test_scan", test_scan);
    printf("\nLayer tests: %d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
