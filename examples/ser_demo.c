#include "sparse_expert_routing.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main(void) {
    ArixSERConfig cfg = arix_ser_config_default();
    cfg.num_experts = 4; cfg.num_active = 2; cfg.input_dim = 16;
    cfg.expert_dim = 32; cfg.output_dim = 16;

    ArixSERModel* model = arix_ser_model_create(&cfg, 42, 1);
    if (!model) { printf("ERROR: create model\n"); return 1; }

    size_t shape_in[] = {8, 16};
    ArixTensor* input = arix_tensor_randn(shape_in, 2, ARIX_FLOAT32);
    if (!input) { printf("ERROR: create input\n"); return 1; }

    ArixTensor* output = NULL;
    arix_ser_forward(model->layers[0], input, &output);
    if (!output) { printf("ERROR: forward\n"); return 1; }

    printf("Output shape: [%zu, %zu]\n", output->shape[0], output->shape[1]);

    ArixTensor* gw = NULL;
    int* ei = NULL;
    arix_ser_route(model->layers[0], input, &gw, &ei);
    if (gw && ei) {
        int counts[4] = {0};
        for (size_t t = 0; t < 8; t++) {
            for (size_t k = 0; k < 2; k++) {
                counts[ei[t * 2 + k]]++;
            }
        }
        printf("Routing distribution (tokens per expert): ");
        for (int i = 0; i < 4; i++) printf("%d ", counts[i]);
        printf("\n");

        float loss = arix_ser_load_balance_loss(gw, ei, 8);
        printf("Load balance loss: %f\n", loss);

        arix_tensor_destroy(gw);
        free(ei);
    }

    float* od = (float*)output->data;
    int has_nan = 0, has_inf = 0;
    for (size_t i = 0; i < output->size; i++) {
        if (isnan(od[i])) has_nan = 1;
        if (isinf(od[i])) has_inf = 1;
    }
    printf("NaN: %s, Inf: %s\n", has_nan ? "YES" : "NO", has_inf ? "YES" : "NO");

    arix_tensor_destroy(input);
    arix_tensor_destroy(output);
    arix_ser_model_destroy(model);
    printf("Demo complete.\n");
    return (has_nan || has_inf) ? 1 : 0;
}
