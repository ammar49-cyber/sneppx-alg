#ifndef SNEPPX_ACTIVATIONS_H
#define SNEPPX_ACTIVATIONS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SNEPPX_ACT_RELU,
    SNEPPX_ACT_LEAKY_RELU,
    SNEPPX_ACT_PRELU,
    SNEPPX_ACT_ELU,
    SNEPPX_ACT_SELU,
    SNEPPX_ACT_GELU_TANH,
    SNEPPX_ACT_GELU_ERF,
    SNEPPX_ACT_SILU,
    SNEPPX_ACT_SIGMOID,
    SNEPPX_ACT_TANH,
    SNEPPX_ACT_SOFTMAX,
    SNEPPX_ACT_LOG_SOFTMAX,
} SNEPPXActivationType;

int SNEPPX_act_forward(const float* x, float* output, SNEPPXActivationType act, size_t n);
int SNEPPX_act_backward(const float* x, const float* grad_out, float* grad_in, SNEPPXActivationType act, size_t n);

int SNEPPX_act_softmax_stable(const float* x, float* output, size_t n);
int SNEPPX_act_log_softmax(const float* x, float* output, size_t n);

int SNEPPX_act_leaky_relu(const float* x, float* output, float alpha, size_t n);
int SNEPPX_act_prelu(const float* x, float* output, const float* alpha, size_t n);
int SNEPPX_act_elu(const float* x, float* output, float alpha, size_t n);
int SNEPPX_act_selu(const float* x, float* output, size_t n);
int SNEPPX_act_gelu_tanh(const float* x, float* output, size_t n);
int SNEPPX_act_gelu_erf(const float* x, float* output, size_t n);

typedef enum {
    SNEPPX_GATED_SWIGLU,
    SNEPPX_GATED_GEGLU,
    SNEPPX_GATED_REGLU,
} SNEPPXGatedActType;

int SNEPPX_gated_activation_forward(const float* x, const float* gate, float* output, SNEPPXGatedActType act, size_t n);
int SNEPPX_gated_activation_backward(const float* x, const float* gate, const float* grad_out, float* grad_x, float* grad_gate, SNEPPXGatedActType act, size_t n);
int SNEPPX_gated_ffn_forward(const float* x, const float* w1, const float* w2, const float* w3, float* output, SNEPPXGatedActType act, size_t dim, size_t hidden_dim);

#ifdef __cplusplus
}
#endif

#endif
