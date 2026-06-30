#include "bench_common.h"
#include "sparse_expert_routing.h"

static void bench_ser_create(void) {
    BENCH_INIT(bs);
    printf("  SER create:\n");

    BENCH_START(bs, 10000, 100, {
        ArixSERConfig cfg = arix_ser_config_default();
        cfg.num_experts = 8; cfg.input_dim = 16; cfg.expert_dim = 32;
        ArixSERLayer* l = arix_ser_layer_create(&cfg, 42);
        arix_ser_layer_destroy(l);
    });
    bench_print("layer_create(8 experts, dim=16)", &bs);

    BENCH_START(bs, 5000, 50, {
        ArixSERConfig cfg = arix_ser_config_default();
        cfg.num_experts = 32; cfg.input_dim = 64; cfg.expert_dim = 128;
        ArixSERLayer* l = arix_ser_layer_create(&cfg, 42);
        arix_ser_layer_destroy(l);
    });
    bench_print("layer_create(32 experts, dim=64)", &bs);
}

static void bench_ser_forward(void) {
    BENCH_INIT(bs);
    ArixSERConfig cfg = arix_ser_config_default();
    cfg.num_experts = 8; cfg.num_active = 2;
    cfg.input_dim = 32; cfg.expert_dim = 64; cfg.output_dim = 32;
    ArixSERLayer* layer = arix_ser_layer_create(&cfg, 42);

    size_t sh[] = {64, 32};
    ArixTensor* input = arix_tensor_ones(sh, 2, ARIX_FLOAT32);
    ArixTensor* output = NULL;

    printf("  SER forward:\n");
    BENCH_START(bs, 500, 50, {
        arix_ser_forward(layer, input, &output);
        if (output) arix_tensor_destroy(output);
    });
    bench_print("forward [64,32] 8 experts", &bs);

    arix_tensor_destroy(input);
    arix_ser_layer_destroy(layer);
}

static void bench_ser_forward_large(void) {
    BENCH_INIT(bs);
    ArixSERConfig cfg = arix_ser_config_default();
    cfg.num_experts = 32; cfg.num_active = 4;
    cfg.input_dim = 128; cfg.expert_dim = 256; cfg.output_dim = 128;
    ArixSERLayer* layer = arix_ser_layer_create(&cfg, 42);

    size_t sh[] = {128, 128};
    ArixTensor* input = arix_tensor_ones(sh, 2, ARIX_FLOAT32);
    ArixTensor* output = NULL;

    printf("  SER forward large:\n");
    BENCH_START(bs, 100, 10, {
        arix_ser_forward(layer, input, &output);
        if (output) arix_tensor_destroy(output);
    });
    bench_print("forward [128,128] 32 experts", &bs);

    arix_tensor_destroy(input);
    arix_ser_layer_destroy(layer);
}

int main(void) {
    printf("=== SER Benchmarks ===\n");
    BENCH_RUN("Create", bench_ser_create);
    BENCH_RUN("Forward", bench_ser_forward);
    BENCH_RUN("Forward large", bench_ser_forward_large);
    BENCH_MAIN();
}
