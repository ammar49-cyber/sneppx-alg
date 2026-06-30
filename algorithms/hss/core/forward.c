#include "hierarchical_state_space.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

static void layer_norm(float* data, size_t n, const float* gamma, const float* beta) {
    float mean = 0.0f;
    for (size_t i = 0; i < n; i++) mean += data[i];
    mean /= (float)n;
    float var = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = data[i] - mean;
        var += d * d;
    }
    var /= (float)n;
    float std = sqrtf(var + 1e-5f);
    for (size_t i = 0; i < n; i++) {
        data[i] = gamma[i] * ((data[i] - mean) / std) + beta[i];
    }
}

static int process_sequence(ArixHSSModel* model, const float* input_seq,
                            float* output_seq, size_t seq_len, size_t input_dim) {
    size_t s_dim = model->config.state_dim;
    size_t o_dim = model->config.output_dim;

    size_t cur_dim = input_dim;
    float* buf = (float*)malloc(seq_len * o_dim * sizeof(float));
    if (!buf) return -1;
    float* cur_buf = (float*)malloc(seq_len * (cur_dim > o_dim ? cur_dim : o_dim) * sizeof(float));
    if (!cur_buf) { free(buf); return -1; }

    memcpy(cur_buf, input_seq, seq_len * input_dim * sizeof(float));

    for (size_t l = 0; l < model->config.num_layers; l++) {
        ArixHSSLayer* layer = model->layers[l];
        arix_hss_discretize(layer);

        float* x_proj_data = (float*)layer->x_proj->data;
        float* x_proj_bias_data = (float*)layer->x_proj_bias->data;
        float* gamma_data = (float*)model->norm_gamma[l]->data;
        float* beta_data = (float*)model->norm_beta[l]->data;

        for (size_t t = 0; t < seq_len; t++) {
            float* xt = cur_buf + t * cur_dim;
            float projected[4096];
            for (size_t i = 0; i < cur_dim; i++) {
                float sum = 0.0f;
                for (size_t j = 0; j < cur_dim; j++) {
                    sum += x_proj_data[i * cur_dim + j] * xt[j];
                }
                projected[i] = sum + x_proj_bias_data[i];
            }
            memcpy(xt, projected, cur_dim * sizeof(float));
            layer_norm(xt, cur_dim, gamma_data, beta_data);
        }

        size_t shape_xs[] = {seq_len, cur_dim};
        ArixTensor x_seq;
        x_seq.data = cur_buf;
        x_seq.shape = shape_xs;
        x_seq.ndim = 2;
        x_seq.size = seq_len * cur_dim;
        x_seq.item_size = sizeof(float);
        x_seq.dtype = ARIX_FLOAT32;
        x_seq.strides = NULL;

        size_t shape_hs[] = {seq_len, s_dim};
        size_t shape_ys[] = {seq_len, o_dim};
        ArixTensor h_seq;
        float* h_seq_data = (float*)malloc(seq_len * s_dim * sizeof(float));
        ArixTensor y_seq;
        float* y_seq_data = (l == model->config.num_layers - 1) ? output_seq : buf;
        if (!h_seq_data) { free(buf); free(cur_buf); return -1; }
        h_seq.data = h_seq_data;
        h_seq.shape = shape_hs;
        h_seq.ndim = 2;
        h_seq.size = seq_len * s_dim;
        h_seq.item_size = sizeof(float);
        h_seq.dtype = ARIX_FLOAT32;
        h_seq.strides = NULL;
        y_seq.data = y_seq_data;
        y_seq.shape = shape_ys;
        y_seq.ndim = 2;
        y_seq.size = seq_len * o_dim;
        y_seq.item_size = sizeof(float);
        y_seq.dtype = ARIX_FLOAT32;
        y_seq.strides = NULL;

        if (model->config.use_hierarchical) {
            arix_hss_hierarchical_scan(layer, &x_seq, &y_seq);
        } else {
            arix_hss_scan(layer, &x_seq, &h_seq, &y_seq);
        }

        free(h_seq_data);

        if (l < model->config.num_layers - 1) {
            float* tmp = cur_buf;
            cur_buf = (float*)realloc(cur_buf, seq_len * o_dim * sizeof(float));
            if (!cur_buf) { free(buf); free(tmp); return -1; }
            memcpy(cur_buf, buf, seq_len * o_dim * sizeof(float));
            cur_dim = o_dim;
        }
    }

    free(buf);
    free(cur_buf);
    return 0;
}

int arix_hss_forward(ArixHSSModel* model, const ArixTensor* input, ArixTensor** output) {
    if (!model || !input || !output) return -1;

    size_t batch, seq_len, input_dim;
    if (input->ndim == 3) {
        batch = input->shape[0];
        seq_len = input->shape[1];
        input_dim = input->shape[2];
    } else if (input->ndim == 2) {
        batch = 1;
        seq_len = input->shape[0];
        input_dim = input->shape[1];
    } else {
        return -1;
    }

    size_t o_dim = model->config.output_dim;
    size_t shape_out[] = {batch, seq_len, o_dim};
    *output = arix_tensor_create(shape_out, 3, ARIX_FLOAT32);
    if (!*output) return -1;

    float* input_data = (float*)input->data;
    float* output_data = (float*)(*output)->data;

    for (size_t b = 0; b < batch; b++) {
        const float* in_b = input_data + b * seq_len * input_dim;
        float* out_b = output_data + b * seq_len * o_dim;

        int ret = process_sequence(model, in_b, out_b, seq_len, input_dim);
        if (ret != 0) return ret;
    }

    return 0;
}
