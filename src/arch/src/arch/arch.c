#include "arix_arch.h"
#include "arix_memory.h"
#include <string.h>
#include <stdlib.h>

ArixArchConfig arix_arch_config_default(void) {
    ArixArchConfig cfg;
    cfg.hss_config = arix_hss_config_default();
    cfg.ser_config = arix_ser_config_default();
    cfg.arc_config = arix_arc_config_default();
    cfg.npe_config = arix_npe_config_default();
    cfg.fm_config = arix_fm_config_default();
    cfg.input_dim = 16;
    cfg.output_dim = 16;
    cfg.seed = 42;
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
    size_t out_dim = config->output_dim ? config->output_dim : 16;

    ArixHSSConfig hss_cfg = config->hss_config;
    if (hss_cfg.input_dim == 0 || hss_cfg.num_layers == 0) {
        hss_cfg = arix_hss_config_default();
        hss_cfg.input_dim = in_dim;
        hss_cfg.output_dim = out_dim;
    }
    ArixSERConfig ser_cfg = config->ser_config;
    if (ser_cfg.num_experts == 0 || ser_cfg.expert_dim == 0) {
        ser_cfg = arix_ser_config_default();
        ser_cfg.input_dim = in_dim;
        ser_cfg.output_dim = out_dim;
    }
    ArixARCConfig arc_cfg = config->arc_config;
    if (arc_cfg.input_guard_strength == 0.0f) {
        arc_cfg = arix_arc_config_default();
    }
    ArixNPEConfig npe_cfg = config->npe_config;
    if (npe_cfg.max_program_length == 0) {
        npe_cfg = arix_npe_config_default();
    }
    ArixFMConfig fm_cfg = config->fm_config;
    if (fm_cfg.num_nodes == 0) {
        fm_cfg = arix_fm_config_default();
    }

    model->hss_model = arix_hss_model_create(&hss_cfg, seed);
    if (!model->hss_model) { arix_model_destroy(model); return NULL; }

    model->ser_model = arix_ser_model_create(&ser_cfg, seed + 1, 1);
    if (!model->ser_model) { arix_model_destroy(model); return NULL; }

    model->arc_layer = arix_arc_layer_create(&arc_cfg, config->input_dim, config->output_dim, seed + 2);
    if (!model->arc_layer) { arix_model_destroy(model); return NULL; }

    model->npe_program = arix_npe_compile_mlp(config->input_dim, config->input_dim * 2);
    if (!model->npe_program) { arix_model_destroy(model); return NULL; }

    unsigned long s = seed;
    size_t total_weights = in_dim * (in_dim * 2) + (in_dim * 2) + (in_dim * 2) * out_dim + out_dim;
    for (size_t i = 0; i < total_weights && i < model->npe_program->memory->size; i++) {
        s = s * 1103515245UL + 12345UL;
        ((float*)model->npe_program->memory->data)[i] = ((float)((s >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 0.1f;
    }

    model->npe_vm = arix_npe_vm_create(&npe_cfg);
    if (!model->npe_vm) { arix_model_destroy(model); return NULL; }
    arix_npe_vm_load(model->npe_vm, model->npe_program);

    model->fm_controller = arix_fm_controller_create(&fm_cfg);
    if (!model->fm_controller) { arix_model_destroy(model); return NULL; }

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
    arix_free(model, sizeof(ArixModel));
}

int arix_model_forward(ArixModel* model, const ArixTensor* input, ArixTensor** output) {
    if (!model || !input || !output) return 1;

    ArixTensor* hss_out = NULL;
    int ret = arix_hss_forward(model->hss_model, input, &hss_out);
    if (ret != 0 || !hss_out) return 1;

    size_t seq = hss_out->shape[1];
    size_t dim = hss_out->shape[2];

    ArixTensor flat;
    size_t flat_shape[] = {seq, dim};
    flat.data = hss_out->data; flat.shape = flat_shape; flat.ndim = 2;
    flat.size = seq * dim; flat.item_size = sizeof(float);
    flat.dtype = ARIX_FLOAT32; flat.strides = NULL;

    ArixTensor* ser_out = NULL;
    arix_ser_forward(model->ser_model->layers[0], &flat, &ser_out);
    if (!ser_out) { arix_tensor_destroy(hss_out); return 1; }

    ArixTensor* arc_out = NULL;
    float metrics[4];
    arix_arc_forward(model->arc_layer, ser_out, &arc_out, metrics);
    if (!arc_out) { arix_tensor_destroy(hss_out); arix_tensor_destroy(ser_out); return 1; }

    ArixTensor* npe_out = NULL;
    arix_npe_vm_run(model->npe_vm, arc_out, &npe_out);
    if (!npe_out) { arix_tensor_destroy(hss_out); arix_tensor_destroy(ser_out); arix_tensor_destroy(arc_out); return 1; }

    ret = arix_fm_forward(model->fm_controller, 0, npe_out, output);
    if (ret != 0 || !(*output)) {
        arix_tensor_destroy(npe_out);
        *output = npe_out;
    }

    arix_tensor_destroy(hss_out);
    arix_tensor_destroy(ser_out);
    arix_tensor_destroy(arc_out);
    if (*output != npe_out) arix_tensor_destroy(npe_out);

    return 0;
}

size_t arix_model_get_params(const ArixModel* model, ArixTensor** out, size_t max_out) {
    if (!model) return 0;
    size_t hss_n = 0, ser_n = 0;
    if (model->hss_model) {
        hss_n = arix_hss_get_params(model->hss_model, NULL, 0);
    }
    if (model->ser_model) {
        ser_n = arix_ser_get_params(model->ser_model, NULL, 0);
    }
    size_t total = hss_n + ser_n;
    size_t idx = 0;
    if (out) {
        if (model->hss_model) {
            size_t n = hss_n;
            if (idx + n > max_out) n = max_out - idx;
            arix_hss_get_params(model->hss_model, out + idx, n);
            idx += n;
        }
        if (model->ser_model) {
            size_t n = ser_n;
            if (idx + n > max_out) n = max_out - idx;
            arix_ser_get_params(model->ser_model, out + idx, n);
            idx += n;
        }
    }
    return total;
}
