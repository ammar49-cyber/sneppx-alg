#include "arix_ser.h"
#include "arix_autodiff.h"
#include "arix_memory.h"
#include <string.h>
#include <stdlib.h>

int arix_ser_build_train_graph(ArixSERModel* model, ArixTape* tape,
                               ArixVariable* input_var,
                               ArixVariable** weight_vars, size_t num_weights,
                               ArixVariable** output_var) {
    if (!model || !tape || !input_var || !output_var) return -1;

    size_t B = input_var->data->shape[0];
    size_t D = model->config.input_dim;
    size_t E = model->config.expert_dim;
    size_t O = model->config.output_dim;
    size_t N = model->config.num_experts;
    size_t wp_layer = 2 + 4 * N;
    size_t total_layers = model->num_layers;

    if (num_weights < total_layers * wp_layer) return -1;

    size_t wi = 0;
    ArixVariable* cur_input = input_var;
    size_t cur_dim = D;

    for (size_t l = 0; l < total_layers; l++) {
        ArixVariable* router = weight_vars[wi++];
        ArixVariable* router_bias = weight_vars[wi++];

        ArixVariable** w1 = (ArixVariable**)arix_malloc(N * sizeof(ArixVariable*), 64);
        ArixVariable** b1 = (ArixVariable**)arix_malloc(N * sizeof(ArixVariable*), 64);
        ArixVariable** w2 = (ArixVariable**)arix_malloc(N * sizeof(ArixVariable*), 64);
        ArixVariable** b2 = (ArixVariable**)arix_malloc(N * sizeof(ArixVariable*), 64);
        if (!w1 || !b1 || !w2 || !b2) {
            arix_free(w1, N * sizeof(ArixVariable*));
            arix_free(b1, N * sizeof(ArixVariable*));
            arix_free(w2, N * sizeof(ArixVariable*));
            arix_free(b2, N * sizeof(ArixVariable*));
            return -1;
        }

        for (size_t e = 0; e < N; e++) {
            w1[e] = weight_vars[wi++];
            b1[e] = weight_vars[wi++];
            w2[e] = weight_vars[wi++];
            b2[e] = weight_vars[wi++];
        }

        ArixVariable* router_T = arix_transpose(tape, router, 0, 1);
        ArixVariable* logits = arix_matmul(tape, cur_input, router_T);
        ArixVariable* gate = arix_softmax(tape, logits, 1);

        size_t ones_shape[] = {1, O};
        ArixTensor* ones_t = arix_tensor_ones(ones_shape, 2, ARIX_FLOAT32);
        ArixVariable* ones_v = arix_variable_create(ones_t, 0);
        arix_tape_record(tape, ones_v);

        size_t acc_shape[] = {B, O};
        ArixTensor* acc_t = arix_tensor_zeros(acc_shape, 2, ARIX_FLOAT32);
        ArixVariable* acc = arix_variable_create(acc_t, 0);
        arix_tape_record(tape, acc);

        for (size_t e = 0; e < N; e++) {
            float* hot_data = (float*)arix_malloc(N * sizeof(float), 64);
            if (!hot_data) { arix_free(w1, N * sizeof(ArixVariable*)); arix_free(b1, N * sizeof(ArixVariable*)); arix_free(w2, N * sizeof(ArixVariable*)); arix_free(b2, N * sizeof(ArixVariable*)); return -1; }
            memset(hot_data, 0, N * sizeof(float));
            hot_data[e] = 1.0f;
            size_t hot_shape[] = {1, N};
            ArixTensor* hot_t = arix_tensor_create(hot_shape, 2, ARIX_FLOAT32);
            if (!hot_t) { arix_free(hot_data, N * sizeof(float)); return -1; }
            memcpy((float*)hot_t->data, hot_data, N * sizeof(float));
            arix_free(hot_data, N * sizeof(float));

            ArixVariable* hot_v = arix_variable_create(hot_t, 0);
            arix_tape_record(tape, hot_v);
            ArixVariable* hot_T = arix_transpose(tape, hot_v, 0, 1);
            ArixVariable* gw = arix_matmul(tape, gate, hot_T);
            ArixVariable* gw_exp = arix_matmul(tape, gw, ones_v);

            ArixVariable* w1_T = arix_transpose(tape, w1[e], 0, 1);
            ArixVariable* hidden = arix_matmul(tape, cur_input, w1_T);
            hidden = arix_add(tape, hidden, b1[e]);
            hidden = arix_relu(tape, hidden);

            ArixVariable* w2_T = arix_transpose(tape, w2[e], 0, 1);
            ArixVariable* exp_out = arix_matmul(tape, hidden, w2_T);
            exp_out = arix_add(tape, exp_out, b2[e]);

            ArixVariable* weighted = arix_mul(tape, exp_out, gw_exp);
            acc = arix_add(tape, acc, weighted);
        }

        arix_free(w1, N * sizeof(ArixVariable*));
        arix_free(b1, N * sizeof(ArixVariable*));
        arix_free(w2, N * sizeof(ArixVariable*));
        arix_free(b2, N * sizeof(ArixVariable*));

        if (l == total_layers - 1) {
            *output_var = arix_sum(tape, acc, 0);
        } else {
            cur_input = acc;
            cur_dim = O;
        }
    }

    return 0;
}
