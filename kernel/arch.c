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
        if (attn_cfg.d_model == 0) { attn_cfg = arix_attn_config_default(); attn_cfg.d_model = in_dim; }
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
    if (input->ndim == 3) {
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
        ArixTensor cos_t;
        /* Use RoPE cos/sin table for fixed length */
        size_t S = input->ndim == 3 ? input->shape[1] : input->shape[0];
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
        total += arix_attn_num_params(model->attention);
    }
    if (model->hss_model) {
        total += arix_hss_get_params(model->hss_model, NULL, 0);
    }
    if (model->ser_model) {
        total += arix_ser_get_params(model->ser_model, NULL, 0);
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
        if (model->embed_weight && idx < max_out) out[idx++] = model->embed_weight;
        if (model->unembed_weight && idx < max_out) out[idx++] = model->unembed_weight;
    }
    return total;
}
