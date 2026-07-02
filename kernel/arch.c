#include "system_architecture_definitions.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

ArixArchConfig arix_arch_config_default(void) {
    ArixArchConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.hss_config = arix_hss_config_default();
    cfg.ser_config = arix_ser_config_default();
    cfg.arc_config = arix_arc_config_default();
    cfg.npe_config = arix_npe_config_default();
    cfg.fm_config = arix_fm_config_default();
    cfg.attention_config = arix_attn_config_default();
    cfg.input_dim = 16;
    cfg.output_dim = 16;
    cfg.vocab_size = 256;
    cfg.seed = 42;
    cfg.enable_attention = 1;
    cfg.enable_hss = 1;
    cfg.enable_ser = 0;
    cfg.enable_arc = 0;
    cfg.enable_npe = 0;
    cfg.enable_fm = 0;
    return cfg;
}

ArixModel* arix_model_create(const ArixArchConfig* config) {
    if (!config) return NULL;
    ArixModel* model = (ArixModel*)arix_malloc(sizeof(ArixModel), 64);
    if (!model) return NULL;
    memset(model, 0, sizeof(ArixModel));

    model->config = *config;
    unsigned int seed = config->seed ? config->seed : 42;
    size_t in_dim = config->input_dim ? config->input_dim : 16;
    size_t vocab = config->vocab_size ? config->vocab_size : 256;

    if (config->enable_hss) {
        ArixHSSConfig hss_cfg = config->hss_config;
        if (hss_cfg.input_dim == 0) { hss_cfg = arix_hss_config_default(); hss_cfg.input_dim = in_dim; }
        if (hss_cfg.output_dim == 0) hss_cfg.output_dim = in_dim;
        model->hss_model = arix_hss_model_create(&hss_cfg, seed);
        if (!model->hss_model) { arix_model_destroy(model); return NULL; }
    }

    if (config->enable_ser) {
        ArixSERConfig ser_cfg = config->ser_config;
        if (ser_cfg.num_experts == 0) { ser_cfg = arix_ser_config_default(); ser_cfg.input_dim = in_dim; }
        if (ser_cfg.output_dim == 0) ser_cfg.output_dim = in_dim;
        model->ser_model = arix_ser_model_create(&ser_cfg, seed + 1, 1);
        if (!model->ser_model) { arix_model_destroy(model); return NULL; }
    }

    if (config->enable_arc) {
        ArixARCConfig arc_cfg = config->arc_config;
        if (arc_cfg.input_guard_strength == 0.0f) arc_cfg = arix_arc_config_default();
        model->arc_layer = arix_arc_layer_create(&arc_cfg, in_dim, in_dim, seed + 2);
        if (!model->arc_layer) { arix_model_destroy(model); return NULL; }
    }

    if (config->enable_npe) {
        ArixNPEConfig npe_cfg = config->npe_config;
        if (npe_cfg.max_program_length == 0) npe_cfg = arix_npe_config_default();
        model->npe_program = arix_npe_compile_mlp(in_dim, in_dim * 2);
        if (!model->npe_program) { arix_model_destroy(model); return NULL; }
        unsigned long s = seed;
        size_t total_w = in_dim * (in_dim * 2) + (in_dim * 2) + (in_dim * 2) * in_dim + in_dim;
        size_t limit = total_w < model->npe_program->memory->size ? total_w : model->npe_program->memory->size;
        for (size_t i = 0; i < limit; i++) {
            s = s * 1103515245UL + 12345UL;
            ((float*)model->npe_program->memory->data)[i] = ((float)((s >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 0.1f;
        }
        model->npe_vm = arix_npe_vm_create(&npe_cfg);
        if (!model->npe_vm) { arix_model_destroy(model); return NULL; }
        arix_npe_vm_load(model->npe_vm, model->npe_program);
    }

    if (config->enable_fm) {
        ArixFMConfig fm_cfg = config->fm_config;
        if (fm_cfg.num_nodes == 0) fm_cfg = arix_fm_config_default();
        model->fm_controller = arix_fm_controller_create(&fm_cfg);
        if (!model->fm_controller) { arix_model_destroy(model); return NULL; }
    }

    if (config->enable_attention) {
        ArixAttentionConfig attn_cfg = config->attention_config;
        if (attn_cfg.d_model == 0) { attn_cfg = arix_attn_config_default(); }
        attn_cfg.d_model = in_dim;
        if (attn_cfg.head_dim == 0) attn_cfg.head_dim = in_dim / (attn_cfg.num_heads > 0 ? attn_cfg.num_heads : 1);
        model->attention = arix_attn_weights_create(attn_cfg, seed + 3);
        if (!model->attention) { arix_model_destroy(model); return NULL; }

        size_t esh[] = {vocab, in_dim};
        model->embed_weight = arix_tensor_create(esh, 2, ARIX_FLOAT32);
        if (model->embed_weight) {
            unsigned long s = seed + 4;
            for (size_t i = 0; i < vocab * in_dim; i++) {
                s = s * 1103515245UL + 12345UL;
                ((float*)model->embed_weight->data)[i] = ((float)((s >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 0.1f;
            }
        }
        size_t ush[] = {in_dim, vocab};
        model->unembed_weight = arix_tensor_create(ush, 2, ARIX_FLOAT32);
        if (model->unembed_weight) {
            unsigned long s = seed + 5;
            for (size_t i = 0; i < in_dim * vocab; i++) {
                s = s * 1103515245UL + 12345UL;
                ((float*)model->unembed_weight->data)[i] = ((float)((s >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 0.1f;
            }
        }
    }

    return model;
}

void arix_model_destroy(ArixModel* model) {
    if (!model) return;
    if (model->hss_model) arix_hss_model_destroy(model->hss_model);
    if (model->ser_model) arix_ser_model_destroy(model->ser_model);
    if (model->arc_layer) arix_arc_layer_destroy(model->arc_layer);
    if (model->npe_vm) arix_npe_vm_destroy(model->npe_vm);
    if (model->npe_program) arix_npe_program_destroy(model->npe_program);
    if (model->fm_controller) arix_fm_controller_destroy(model->fm_controller);
    if (model->attention) arix_attn_weights_destroy(model->attention);
    if (model->embed_weight) arix_tensor_destroy(model->embed_weight);
    if (model->unembed_weight) arix_tensor_destroy(model->unembed_weight);
    arix_free(model, sizeof(ArixModel));
}

int arix_model_forward(ArixModel* model, const ArixTensor* input, ArixTensor** output) {
    if (!model || !input || !output) return 1;

    ArixTensor* current = NULL;

    /* For attention models with embedding, do embedding lookup on token indices */
    if (model->attention && model->embed_weight && input->ndim == 2) {
        /* Input: [B, S] token IDs stored as floats */
        size_t B = input->shape[0], S = input->shape[1];
        size_t D = model->attention->config.d_model;
        (void)D;
        ArixTensor* idx_t = arix_tensor_create(input->shape, input->ndim, ARIX_INT32);
        if (!idx_t) return 1;
        size_t n = B * S;
        int* id = (int*)idx_t->data;
        float* fd = (float*)input->data;
        for (size_t i = 0; i < n; i++) id[i] = (int)fd[i];
        current = arix_tensor_embedding(model->embed_weight, idx_t);
        arix_tensor_destroy(idx_t);
        if (current) {
            size_t sh3[] = {B, S, D};
            ArixTensor* r = arix_tensor_reshape(current, sh3, 3);
            if (r) { arix_tensor_destroy(current); current = r; }
        }
    } else if (input->ndim == 3) {
        size_t b = input->shape[0], s = input->shape[1], d = input->shape[2];
        size_t sh[] = {b * s, d};
        current = arix_tensor_create(sh, 2, ARIX_FLOAT32);
        if (current) memcpy(current->data, input->data, b * s * d * sizeof(float));
    } else if (input->ndim == 2) {
        current = arix_tensor_create(input->shape, input->ndim, ARIX_FLOAT32);
        if (current) memcpy(current->data, input->data, input->size * sizeof(float));
    }
    if (!current) return 1;

    if (model->attention && model->embed_weight) {
        size_t S = current->ndim >= 3 ? current->shape[1] : current->shape[0];
        ArixTensor* cos_full = arix_rope_precompute(S, model->attention->config.head_dim, model->attention->config.rope_base);
        if (cos_full) {
            ArixTensor* attn_out = arix_attn_forward(model->attention, current, cos_full, cos_full);
            arix_tensor_destroy(cos_full);
            if (attn_out) {
                arix_tensor_destroy(current);
                current = attn_out;
            }
        }
    }

    if (model->hss_model) {
        ArixTensor* hss_out = NULL;
        if (arix_hss_forward(model->hss_model, current, &hss_out) == 0 && hss_out) {
            arix_tensor_destroy(current);
            current = hss_out;
        }
    }

    if (model->ser_model) {
        size_t seq = current->ndim == 2 ? current->shape[0] : current->shape[1];
        size_t dim = current->shape[current->ndim - 1];
        ArixTensor flat;
        size_t fsh[] = {seq, dim};
        flat.data = current->data; flat.shape = fsh; flat.ndim = 2;
        flat.size = seq * dim; flat.item_size = sizeof(float);
        flat.dtype = ARIX_FLOAT32; flat.strides = NULL;
        ArixTensor* ser_out = NULL;
        arix_ser_forward(model->ser_model->layers[0], &flat, &ser_out);
        if (ser_out) {
            arix_tensor_destroy(current);
            current = ser_out;
        }
    }

    if (model->arc_layer) {
        float metrics[4];
        ArixTensor* arc_out = NULL;
        arix_arc_forward(model->arc_layer, current, &arc_out, metrics);
        if (arc_out) {
            arix_tensor_destroy(current);
            current = arc_out;
        }
    }

    if (model->npe_vm && model->npe_program) {
        ArixTensor* npe_out = NULL;
        arix_npe_vm_run(model->npe_vm, current, &npe_out);
        if (npe_out) {
            arix_tensor_destroy(current);
            current = npe_out;
        }
    }

    if (model->fm_controller) {
        ArixTensor* fm_out = NULL;
        int ret = arix_fm_forward(model->fm_controller, 0, current, &fm_out);
        if (ret == 0 && fm_out) {
            arix_tensor_destroy(current);
            current = fm_out;
        }
    }

    *output = current;
    return current ? 0 : 1;
}

size_t arix_model_get_params(const ArixModel* model, ArixTensor** out, size_t max_out) {
    if (!model) return 0;
    size_t total = 0;

    if (model->attention) {
        total += 8; /* 8 weight tensors: w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o */
    }
    if (model->hss_model) {
        total += arix_hss_get_params(model->hss_model, NULL, 0);
    }
    if (model->ser_model) {
        total += arix_ser_get_params(model->ser_model, NULL, 0);
    }
    if (model->arc_layer) {
        total += arix_arc_get_params(model->arc_layer, NULL, 0);
    }
    if (model->npe_program) {
        total += arix_npe_get_params(model->npe_program, NULL, 0);
    }
    if (model->fm_controller) {
        total += arix_fm_get_params(model->fm_controller, NULL, 0);
    }
    if (model->embed_weight) total++;
    if (model->unembed_weight) total++;

    size_t idx = 0;
    if (out) {
        if (model->attention) {
            ArixTensor* attn_params[8];
            int n = arix_attn_get_params(model->attention, attn_params, 8);
            for (int i = 0; i < n && idx < max_out; i++, idx++) out[idx] = attn_params[i];
        }
        if (model->hss_model) {
            size_t n = arix_hss_get_params(model->hss_model, out + idx, max_out > idx ? max_out - idx : 0);
            idx += n;
        }
        if (model->ser_model) {
            size_t n = arix_ser_get_params(model->ser_model, out + idx, max_out > idx ? max_out - idx : 0);
            idx += n;
        }
        if (model->arc_layer) {
            size_t n = arix_arc_get_params(model->arc_layer, out + idx, max_out > idx ? max_out - idx : 0);
            idx += n;
        }
        if (model->npe_program) {
            size_t n = arix_npe_get_params(model->npe_program, out + idx, max_out > idx ? max_out - idx : 0);
            idx += n;
        }
        if (model->fm_controller) {
            size_t n = arix_fm_get_params(model->fm_controller, out + idx, max_out > idx ? max_out - idx : 0);
            idx += n;
        }
        if (model->embed_weight && idx < max_out) out[idx++] = model->embed_weight;
        if (model->unembed_weight && idx < max_out) out[idx++] = model->unembed_weight;
    }
    return total;
}

int arix_model_build_train_graph(ArixModel* model, ArixTape* tape,
                                  ArixVariable* input_var,
                                  ArixVariable** weight_vars, size_t num_weights,
                                  ArixVariable** output_var) {
    if (!model || !tape || !input_var || !output_var) return -1;
    (void)num_weights;
    ArixVariable* current = input_var;
    size_t woff = 0;

    if (model->attention) {
        size_t S = current->data->ndim >= 2 ? current->data->shape[1] : current->data->shape[0];
        ArixTensor* cos_t = arix_rope_precompute(S, model->attention->config.head_dim, model->attention->config.rope_base);
        ArixTensor* sin_t = cos_t ? arix_tensor_copy(cos_t) : NULL;
        ArixVariable* attn_out = NULL;
        int ret = arix_attn_build_train_graph(model->attention, tape, current,
                                               weight_vars + woff, 8, &attn_out,
                                               cos_t, sin_t);
        if (cos_t) arix_tensor_destroy(cos_t);
        if (sin_t) arix_tensor_destroy(sin_t);
        if (ret != 0 || !attn_out) return -1;
        current = attn_out;
        woff += 8;
    }

    /* Flatten 3D -> 2D for HSS/SER/ARC */
    if ((model->hss_model || model->ser_model || model->arc_layer) && current->data->ndim == 3) {
        size_t flat_sh[] = {current->data->shape[0] * current->data->shape[1], current->data->shape[2]};
        current = arix_reshape(tape, current, flat_sh, 2);
    }

    if (model->hss_model) {
        size_t nhss = arix_hss_get_params(model->hss_model, NULL, 0);
        ArixVariable* hss_out = NULL;
        int ret = arix_hss_build_train_graph(model->hss_model, tape, current,
                                              weight_vars + woff, nhss, &hss_out);
        if (ret != 0 || !hss_out) return -1;
        current = hss_out;
        woff += nhss;
    }

    if (model->ser_model) {
        size_t nser = arix_ser_get_params(model->ser_model, NULL, 0);
        ArixVariable* ser_out = NULL;
        int ret = arix_ser_build_train_graph(model->ser_model, tape, current,
                                              weight_vars + woff, nser, &ser_out);
        if (ret != 0 || !ser_out) return -1;
        current = ser_out;
        woff += nser;
    }

    if (model->arc_layer) {
        size_t narc = arix_arc_get_params(model->arc_layer, NULL, 0);
        ArixVariable* arc_out = NULL;
        int ret = arix_arc_build_train_graph(model->arc_layer, tape, current,
                                              weight_vars + woff, narc, &arc_out);
        if (ret != 0 || !arc_out) return -1;
        current = arc_out;
        woff += narc;
    }

    if (model->npe_program) {
        if (current->data->ndim == 3) {
            size_t flat_sh[] = {current->data->shape[0] * current->data->shape[1], current->data->shape[2]};
            current = arix_reshape(tape, current, flat_sh, 2);
        }
        size_t nnpe = arix_npe_get_params(model->npe_program, NULL, 0);
        ArixVariable* npe_out = NULL;
        int ret = arix_npe_build_train_graph(model->npe_program, tape, current,
                                              weight_vars + woff, nnpe, &npe_out);
        if (ret != 0 || !npe_out) return -1;
        current = npe_out;
        woff += nnpe;
    }

    if (model->fm_controller) {
        size_t nfm = arix_fm_get_params(model->fm_controller, NULL, 0);
        ArixVariable* fm_out = NULL;
        int ret = arix_fm_build_train_graph(model->fm_controller, tape, current,
                                             weight_vars + woff, nfm, &fm_out);
        if (ret != 0 || !fm_out) return -1;
        current = fm_out;
        woff += nfm;
    }

    *output_var = current;
    return 0;
}
