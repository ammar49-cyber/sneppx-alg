#include "bench_common.h"
#include "hierarchical_state_space.h"

static void bench_hss_create(void) {
    BENCH_INIT(bs);
    printf("  HSS create:\n");

    BENCH_START(bs, 10000, 100, {
        ArixHSSConfig cfg = arix_hss_config_default();
        cfg.state_dim = 16; cfg.num_layers = 2;
        ArixHSSModel* m = arix_hss_model_create(&cfg, 42);
        arix_hss_model_destroy(m);
    });
    bench_print("model_create(state=16,layers=2)", &bs);

    BENCH_START(bs, 5000, 50, {
        ArixHSSConfig cfg = arix_hss_config_default();
        cfg.state_dim = 128; cfg.num_layers = 4;
        ArixHSSModel* m = arix_hss_model_create(&cfg, 42);
        arix_hss_model_destroy(m);
    });
    bench_print("model_create(state=128,layers=4)", &bs);
}

static void bench_hss_forward(void) {
    BENCH_INIT(bs);
    ArixHSSConfig cfg = arix_hss_config_default();
    cfg.state_dim = 64; cfg.input_dim = 32; cfg.output_dim = 16;
    cfg.num_layers = 4; cfg.seq_len = 128;
    ArixHSSModel* model = arix_hss_model_create(&cfg, 42);

    size_t sh[] = {128, 32};
    ArixTensor* input = arix_tensor_ones(sh, 2, ARIX_FLOAT32);
    ArixTensor* output = NULL;

    printf("  HSS forward:\n");
    BENCH_START(bs, 200, 20, {
        arix_hss_forward(model, input, &output);
        if (output) arix_tensor_destroy(output);
    });
    bench_print("forward seq=128,dim=32,state=64,layers=4", &bs);

    arix_tensor_destroy(input);
    arix_hss_model_destroy(model);
}

static void bench_hss_forward_large(void) {
    BENCH_INIT(bs);
    ArixHSSConfig cfg = arix_hss_config_default();
    cfg.state_dim = 256; cfg.input_dim = 128; cfg.output_dim = 64;
    cfg.num_layers = 6; cfg.seq_len = 512;
    ArixHSSModel* model = arix_hss_model_create(&cfg, 42);

    size_t sh[] = {512, 128};
    ArixTensor* input = arix_tensor_ones(sh, 2, ARIX_FLOAT32);
    ArixTensor* output = NULL;

    printf("  HSS forward large:\n");
    BENCH_START(bs, 20, 5, {
        arix_hss_forward(model, input, &output);
        if (output) arix_tensor_destroy(output);
    });
    bench_print("forward seq=512,dim=128,state=256,layers=6", &bs);

    arix_tensor_destroy(input);
    arix_hss_model_destroy(model);
}

int main(void) {
    printf("=== HSS Benchmarks ===\n");
    BENCH_RUN("Create", bench_hss_create);
    BENCH_RUN("Forward", bench_hss_forward);
    BENCH_RUN("Forward large", bench_hss_forward_large);
    BENCH_MAIN();
}
