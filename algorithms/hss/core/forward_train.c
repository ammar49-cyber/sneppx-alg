#include "hierarchical_state_space.h"
#include "automatic_differentiation_framework.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

int arix_hss_build_train_graph(ArixHSSModel* model, ArixTape* tape,
                               ArixVariable* input_var,
                               ArixVariable** weight_vars, size_t num_weights,
                               ArixVariable** output_var) {
    if (!model || !tape || !input_var || !output_var) return -1;
    size_t seq_len = input_var->data->shape[0];
    size_t i_dim = input_var->data->shape[1];
    size_t o_dim = model->config.output_dim;
    size_t s_dim = model->config.state_dim;
    size_t wi = 0;

    ArixVariable* layer_input = input_var;
    size_t cur_dim = i_dim;

    for (size_t l = 0; l < model->config.num_layers; l++) {
        if (wi + 9 > num_weights) return -1;
        ArixVariable* A   = weight_vars[wi++];
        ArixVariable* B   = weight_vars[wi++];
        ArixVariable* C   = weight_vars[wi++];
        ArixVariable* D   = weight_vars[wi++];
        ArixVariable* dt  = weight_vars[wi++];
        ArixVariable* xp  = weight_vars[wi++];
        ArixVariable* xpb = weight_vars[wi++];
        ArixVariable* ng  = weight_vars[wi++];
        ArixVariable* nb  = weight_vars[wi++];

        size_t s1[] = {s_dim, 1};
        ArixTensor* dt2_t = arix_tensor_zeros(s1, 2, ARIX_FLOAT32);
        if (!dt2_t) return -1;
        memcpy((float*)dt2_t->data, (float*)dt->data->data, s_dim * sizeof(float));
        ArixVariable* dt2 = arix_variable_create(dt2_t, 1);
        arix_tape_record(tape, dt2);

        size_t or1[] = {1, s_dim};
        ArixTensor* or1_t = arix_tensor_ones(or1, 2, ARIX_FLOAT32);
        if (!or1_t) return -1;
        ArixVariable* or1_v = arix_variable_create(or1_t, 0);
        arix_tape_record(tape, or1_v);

        ArixVariable* dt_mat = arix_matmul(tape, dt2, or1_v);

        ArixTensor* eye_t = arix_tensor_eye(s_dim, ARIX_FLOAT32);
        if (!eye_t) return -1;
        ArixVariable* eye_v = arix_variable_create(eye_t, 0);
        arix_tape_record(tape, eye_v);

        ArixVariable* A_dt = arix_mul(tape, A, dt_mat);
        ArixVariable* A_bar = arix_add(tape, eye_v, A_dt);

        size_t ori[] = {1, cur_dim};
        ArixTensor* ori_t = arix_tensor_ones(ori, 2, ARIX_FLOAT32);
        if (!ori_t) return -1;
        ArixVariable* ori_v = arix_variable_create(ori_t, 0);
        arix_tape_record(tape, ori_v);

        ArixVariable* dt_mat2 = arix_matmul(tape, dt2, ori_v);
        ArixVariable* B_bar = arix_mul(tape, B, dt_mat2);

        size_t hs[] = {s_dim, 1};
        ArixTensor* h0 = arix_tensor_zeros(hs, 2, ARIX_FLOAT32);
        if (!h0) return -1;
        ArixVariable* h = arix_variable_create(h0, 0);
        arix_tape_record(tape, h);

        ArixVariable** yts = (ArixVariable**)arix_malloc(seq_len * sizeof(ArixVariable*), 64);
        if (!yts) return -1;
        memset(yts, 0, seq_len * sizeof(ArixVariable*));

        for (size_t t = 0; t < seq_len; t++) {
            ArixTensor* xs = arix_tensor_slice(layer_input->data, 0, (int)t, (int)(t + 1));
            if (!xs) { arix_free(yts, seq_len * sizeof(ArixVariable*)); return -1; }
            ArixVariable* xt_row = arix_variable_create(xs, 0);
            arix_tape_record(tape, xt_row);

            ArixVariable* xt_col = arix_transpose(tape, xt_row, 0, 1);

            ArixVariable* xp_out = arix_matmul(tape, xp, xt_col);
            ArixVariable* xp_b = arix_add(tape, xp_out, xpb);
            ArixVariable* xp_b_row = arix_transpose(tape, xp_b, 0, 1);
            ArixVariable* xp_norm = arix_layer_norm(tape, xp_b_row, ng, nb, 1e-5f);
            ArixVariable* xp_norm_col = arix_transpose(tape, xp_norm, 0, 1);

            size_t iz[] = {cur_dim, 1};
            ArixTensor* zt = arix_tensor_zeros(iz, 2, ARIX_FLOAT32);
            if (!zt) { arix_free(yts, seq_len * sizeof(ArixVariable*)); return -1; }
            ArixVariable* zv = arix_variable_create(zt, 0);
            arix_tape_record(tape, zv);
            ArixVariable* x_col2 = arix_add(tape, zv, xp_norm_col);

            ArixVariable* h_new = arix_add(tape,
                arix_matmul(tape, A_bar, h),
                arix_matmul(tape, B_bar, x_col2));

            ArixVariable* yt = arix_add(tape,
                arix_matmul(tape, C, h_new),
                arix_matmul(tape, D, x_col2));

            h = h_new;
            yts[t] = yt;
        }

        {
            ArixVariable** yt_rows = (ArixVariable**)arix_malloc(seq_len * sizeof(ArixVariable*), 64);
            if (!yt_rows) { arix_free(yts, seq_len * sizeof(ArixVariable*)); return -1; }
            for (size_t t = 0; t < seq_len; t++) {
                yt_rows[t] = arix_transpose(tape, yts[t], 0, 1);
            }
            ArixVariable* layer_out = arix_concat(tape, yt_rows, seq_len, 0);
            arix_free(yt_rows, seq_len * sizeof(ArixVariable*));
            arix_free(yts, seq_len * sizeof(ArixVariable*));
            if (!layer_out) return -1;
            if (l == model->config.num_layers - 1) {
                *output_var = layer_out;
            } else {
                layer_input = layer_out;
                cur_dim = o_dim;
            }
        }
    }
    return 0;
}
