#include "arix_hss.h"
#include <stdio.h>
#include <math.h>

int main(void) {
    ArixHSSConfig cfg = arix_hss_config_default();
    cfg.state_dim = 4;
    cfg.input_dim = 8;
    cfg.output_dim = 8;
    cfg.num_layers = 1;
    cfg.seq_len = 16;
    cfg.use_hierarchical = 0;

    ArixHSSModel* model = arix_hss_model_create(&cfg, 42);
    if (!model) {
        printf("ERROR: failed to create model\n");
        return 1;
    }

    size_t shape_in[] = {16, 8};
    ArixTensor* input = arix_tensor_randn(shape_in, 2, ARIX_FLOAT32);
    if (!input) {
        printf("ERROR: failed to create input\n");
        arix_hss_model_destroy(model);
        return 1;
    }

    ArixTensor* output = NULL;
    int ret = arix_hss_forward(model, input, &output);
    if (ret != 0 || !output) {
        printf("ERROR: forward pass failed (ret=%d)\n", ret);
        arix_tensor_destroy(input);
        arix_hss_model_destroy(model);
        return 1;
    }

    printf("Output shape: [%zu, %zu]\n", output->shape[0], output->shape[1]);

    float* data = (float*)output->data;
    printf("First 8 values: ");
    for (size_t i = 0; i < 8 && i < output->size; i++) {
        printf("%f ", data[i]);
    }
    printf("\n");

    int has_nan = 0, has_inf = 0;
    for (size_t i = 0; i < output->size; i++) {
        if (isnan(data[i])) has_nan = 1;
        if (isinf(data[i])) has_inf = 1;
    }
    printf("NaN: %s, Inf: %s\n", has_nan ? "YES" : "NO", has_inf ? "YES" : "NO");

    arix_tensor_destroy(input);
    arix_tensor_destroy(output);
    arix_hss_model_destroy(model);

    printf("Demo complete.\n");
    return (has_nan || has_inf) ? 1 : 0;
}
