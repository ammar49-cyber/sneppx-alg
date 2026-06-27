#include "arix_train.h"
#include "arix_autodiff.h"
#include "arix_memory.h"
#include "arix_arch.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

ArixTrainConfig arix_train_config_default(void) {
    ArixTrainConfig cfg;
    cfg.num_epochs = 10;
    cfg.batch_size = 32;
    cfg.learning_rate = 0.01f;
    cfg.log_interval = 10;
    cfg.save_interval = 100;
    cfg.device = ARIX_DEVICE_CPU;
    return cfg;
}

ArixTrainer* arix_trainer_create(ArixModel* model, const ArixTrainConfig* config) {
    if (!model || !config) return NULL;
    ArixTrainer* trainer = (ArixTrainer*)arix_malloc(sizeof(ArixTrainer), 64);
    if (!trainer) return NULL;
    memset(trainer, 0, sizeof(ArixTrainer));

    trainer->model = model;
    trainer->config = *config;

    ArixOptimizerConfig opt_cfg = arix_optimizer_config_default();
    opt_cfg.learning_rate = config->learning_rate;
    trainer->optimizer = arix_optimizer_create(&opt_cfg);
    if (!trainer->optimizer) { arix_free(trainer, sizeof(ArixTrainer)); return NULL; }

    size_t hist_shape[] = {1024};
    trainer->loss_history = arix_tensor_zeros(hist_shape, 1, ARIX_FLOAT32);
    if (!trainer->loss_history) {
        arix_optimizer_destroy(trainer->optimizer);
        arix_free(trainer, sizeof(ArixTrainer));
        return NULL;
    }

    trainer->step_count = 0;
    return trainer;
}

void arix_trainer_destroy(ArixTrainer* trainer) {
    if (!trainer) return;
    if (trainer->optimizer) arix_optimizer_destroy(trainer->optimizer);
    if (trainer->loss_history) arix_tensor_destroy(trainer->loss_history);
    arix_free(trainer, sizeof(ArixTrainer));
}

float arix_trainer_train_step(ArixTrainer* trainer, const ArixTensor* batch_input, const ArixTensor* batch_target) {
    if (!trainer || !batch_input || !batch_target) return -1.0f;

    size_t nw = arix_model_get_params(trainer->model, NULL, 0);

    ArixTensor** params = (ArixTensor**)arix_malloc(nw * sizeof(ArixTensor*), 64);
    if (!params) return -1.0f;
    arix_model_get_params(trainer->model, params, nw);

    ArixTape* tape = arix_tape_create();

    ArixVariable** wv = (ArixVariable**)arix_malloc(nw * sizeof(ArixVariable*), 64);
    if (!wv) { arix_free(params, nw * sizeof(ArixTensor*)); arix_tape_destroy(tape); return -1.0f; }
    for (size_t i = 0; i < nw; i++) {
        wv[i] = arix_variable_create(params[i], 1);
    }

    size_t in_shape2[2];
    size_t in_ndim;
    size_t in_size;
    const size_t* in_shape;
    if (batch_input->ndim == 3) {
        in_shape2[0] = batch_input->shape[0] * batch_input->shape[1];
        in_shape2[1] = batch_input->shape[2];
        in_shape = in_shape2;
        in_ndim = 2;
        in_size = in_shape2[0] * in_shape2[1];
    } else {
        in_shape = batch_input->shape;
        in_ndim = batch_input->ndim;
        in_size = batch_input->size;
    }
    ArixTensor* inv_t = arix_tensor_create(in_shape, in_ndim, ARIX_FLOAT32);
    memcpy(inv_t->data, ((ArixTensor*)batch_input)->data, in_size * sizeof(float));
    ArixVariable* inv = arix_variable_create(inv_t, 0);

    ArixVariable* outv = NULL;
    int ret = -1;
    if (trainer->model->hss_model && trainer->model->hss_model->config.num_layers > 0) {
        ret = arix_hss_build_train_graph(trainer->model->hss_model, tape, inv, wv, nw, &outv);
    } else if (trainer->model->ser_model && trainer->model->ser_model->num_layers > 0) {
        ret = arix_ser_build_train_graph(trainer->model->ser_model, tape, inv, wv, nw, &outv);
    }

    float loss_val = -1.0f;
    if (ret == 0 && outv) {
        size_t out_sz = outv->data->size;
        size_t tgt_sz = batch_target->size;
        size_t copy_sz = out_sz < tgt_sz ? out_sz : tgt_sz;
        ArixTensor* tgt_flat = arix_tensor_zeros(&out_sz, 1, ARIX_FLOAT32);
        if (tgt_flat) {
            memcpy((float*)tgt_flat->data, (float*)batch_target->data, copy_sz * sizeof(float));
        }
        ArixVariable* tgv = arix_variable_create(tgt_flat, 0);
        ArixVariable* loss = arix_mse_loss(tape, outv, tgv);

        if (loss) {
            arix_tape_backward(tape, loss);

            ArixTensor** grads = (ArixTensor**)arix_malloc(nw * sizeof(ArixTensor*), 64);
            if (grads) {
                for (size_t i = 0; i < nw; i++) {
                    grads[i] = wv[i]->grad;
                }
                arix_optimizer_step(trainer->optimizer, params, grads, nw);
                arix_free(grads, nw * sizeof(ArixTensor*));
            }

            loss_val = ((float*)loss->data)[0];
        }

        tgv->data = NULL; arix_variable_destroy(tgv);
    }

    size_t idx = trainer->step_count;
    if (idx < trainer->loss_history->size) {
        ((float*)trainer->loss_history->data)[idx] = loss_val;
    }

    trainer->step_count++;

    for (size_t i = 0; i < nw; i++) { wv[i]->data = NULL; wv[i]->grad = NULL; arix_variable_destroy(wv[i]); }
    arix_variable_destroy(inv);
    arix_free(wv, nw * sizeof(ArixVariable*));
    arix_free(params, nw * sizeof(ArixTensor*));
    arix_tape_destroy(tape);

    if (trainer->config.log_interval > 0 && trainer->step_count % trainer->config.log_interval == 0) {
        printf("[Step %zu] loss = %.6f\n", trainer->step_count, (double)loss_val);
    }

    return loss_val;
}

float arix_trainer_evaluate(ArixTrainer* trainer, const ArixTensor* val_input, const ArixTensor* val_target) {
    if (!trainer || !val_input || !val_target) return -1.0f;

    ArixTensor* output = NULL;
    int ret = arix_model_forward(trainer->model, val_input, &output);
    if (ret != 0 || !output) return -1.0f;

    float* od = (float*)output->data;
    float* td = (float*)val_target->data;
    size_t sz = output->size < val_target->size ? output->size : val_target->size;

    float sum_sq = 0.0f;
    for (size_t i = 0; i < sz; i++) {
        float d = od[i] - td[i];
        sum_sq += d * d;
    }
    float loss = sum_sq / (float)(sz > 0 ? sz : 1);

    arix_tensor_destroy(output);
    return loss;
}

int arix_trainer_save_checkpoint(ArixTrainer* trainer, const char* path) {
    if (!trainer || !path) return 1;
    FILE* f = fopen(path, "wb");
    if (!f) return 1;

    unsigned int magic = 0x41524958;
    unsigned int version = 2;
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&version, sizeof(version), 1, f);
    fwrite(&trainer->step_count, sizeof(trainer->step_count), 1, f);

    size_t nw = arix_model_get_params(trainer->model, NULL, 0);
    fwrite(&nw, sizeof(nw), 1, f);

    if (nw > 0) {
        ArixTensor** params = (ArixTensor**)arix_malloc(nw * sizeof(ArixTensor*), 64);
        if (params) {
            arix_model_get_params(trainer->model, params, nw);
            for (size_t i = 0; i < nw; i++) {
                if (!params[i]) continue;
                unsigned int ndim = (unsigned int)params[i]->ndim;
                fwrite(&ndim, sizeof(ndim), 1, f);
                fwrite(params[i]->shape, sizeof(size_t), ndim, f);
                fwrite(&params[i]->size, sizeof(params[i]->size), 1, f);
                fwrite(params[i]->data, sizeof(float), params[i]->size, f);
            }
            arix_free(params, nw * sizeof(ArixTensor*));
        }
    }

    fclose(f);
    return 0;
}

int arix_trainer_load_checkpoint(ArixTrainer* trainer, const char* path) {
    if (!trainer || !path) return 1;
    FILE* f = fopen(path, "rb");
    if (!f) return 1;

    unsigned int magic, version;
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != 0x41524958) {
        fclose(f); return 1;
    }
    if (fread(&version, sizeof(version), 1, f) != 1 || version < 1 || version > 2) {
        fclose(f); return 1;
    }
    if (fread(&trainer->step_count, sizeof(trainer->step_count), 1, f) != 1) {
        fclose(f); return 1;
    }

    if (version >= 2) {
        size_t nw = 0;
        if (fread(&nw, sizeof(nw), 1, f) != 1) { fclose(f); return 1; }
        if (nw > 0) {
            ArixTensor** params = (ArixTensor**)arix_malloc(nw * sizeof(ArixTensor*), 64);
            if (params) {
                size_t actual = arix_model_get_params(trainer->model, params, nw);
                for (size_t i = 0; i < actual && i < nw; i++) {
                    if (!params[i]) continue;
                    unsigned int ndim;
                    size_t stored_shape[8];
                    size_t stored_size;
                    if (fread(&ndim, sizeof(ndim), 1, f) != 1) break;
                    if (ndim > 8) { for (size_t j = 0; j < ndim; j++) { size_t tmp; fread(&tmp, sizeof(tmp), 1, f); } break; }
                    if (fread(stored_shape, sizeof(size_t), ndim, f) != ndim) break;
                    if (fread(&stored_size, sizeof(stored_size), 1, f) != 1) break;
                    size_t copy_sz = stored_size < params[i]->size ? stored_size : params[i]->size;
                    if (fread(params[i]->data, sizeof(float), copy_sz, f) != copy_sz) break;
                    if (stored_size > copy_sz) { float tmp; for (size_t j = copy_sz; j < stored_size; j++) fread(&tmp, sizeof(float), 1, f); }
                }
                arix_free(params, nw * sizeof(ArixTensor*));
            }
        }
    }

    fclose(f);
    return 0;
}
