#ifndef SNEPPX_QUANTIZATION_H
#define SNEPPX_QUANTIZATION_H

#include <stddef.h>
#include <stdint.h>
#include "multidimensional_tensor_engine.h"

typedef enum {
    SNEPPX_QUANT_NONE = 0,
    SNEPPX_QUANT_INT8_SYM,
    SNEPPX_QUANT_INT8_ASYM,
    SNEPPX_QUANT_INT4_SYM,
    SNEPPX_QUANT_FP8_E4M3,
    SNEPPX_QUANT_FP8_E5M2,
    SNEPPX_QUANT_AWQ,
    SNEPPX_QUANT_GPTQ,
} SNEPPXQuantMode;

typedef enum {
    SNEPPX_QUANT_PER_TENSOR = 0,
    SNEPPX_QUANT_PER_CHANNEL,
    SNEPPX_QUANT_PER_GROUP,
    SNEPPX_QUANT_PER_TOKEN,
} SNEPPXQuantGranularity;

typedef struct {
    float scale;
    int32_t zero_point;
    float qmin;
    float qmax;
    float scale_max;
} SNEPPXQuantParams;

typedef struct {
    float* scales;
    int32_t* zero_points;
    size_t num_params;
    SNEPPXQuantGranularity granularity;
} SNEPPXQuantState;

typedef struct {
    float* weights;        /* original high-precision weights */
    float* hessian;        /* Hessian matrix (GPTQ) */
    size_t rows;
    size_t cols;
    int group_size;
} SNEPPXGPTQState;

#ifdef __cplusplus
extern "C" {
#endif

/* INT8 symmetric quantization */
int SNEPPX_quantize_int8_sym(const float* input, int8_t* output,
                              size_t n, float* scale_out);
int SNEPPX_dequantize_int8_sym(const int8_t* input, float* output,
                                size_t n, float scale);

/* INT8 asymmetric quantization */
int SNEPPX_quantize_int8_asym(const float* input, int8_t* output,
                               size_t n, float* scale_out, int32_t* zp_out);
int SNEPPX_dequantize_int8_asym(const int8_t* input, float* output,
                                 size_t n, float scale, int32_t zp);

/* Per-channel INT8 symmetric */
int SNEPPX_quantize_int8_channel(const float* input, int8_t* output,
                                  size_t rows, size_t cols,
                                  float* scales_out);

/* INT4 symmetric (packed) */
int SNEPPX_quantize_int4_sym(const float* input, uint8_t* output,
                              size_t n, float* scale_out);
int SNEPPX_dequantize_int4_sym(const uint8_t* input, float* output,
                                size_t n, float scale);

/* FP8 E4M3 encode/decode (software fallback) */
uint8_t SNEPPX_float_to_fp8_e4m3(float value);
float SNEPPX_fp8_e4m3_to_float(uint8_t fp8);
int SNEPPX_quantize_fp8_e4m3(const float* input, uint8_t* output, size_t n);
int SNEPPX_dequantize_fp8_e4m3(const uint8_t* input, float* output, size_t n);

/* FP8 E5M2 encode/decode (software fallback) */
uint8_t SNEPPX_float_to_fp8_e5m2(float value);
float SNEPPX_fp8_e5m2_to_float(uint8_t fp8);
int SNEPPX_quantize_fp8_e5m2(const float* input, uint8_t* output, size_t n);
int SNEPPX_dequantize_fp8_e5m2(const uint8_t* input, float* output, size_t n);

/* AWQ: Activation-aware Weight Quantization */
int SNEPPX_awq_scale_weights(float* weights, size_t rows, size_t cols,
                              const float* act_scales, float* scale_out);
int SNEPPX_awq_quantize(const float* weights, int8_t* qweight,
                         const float* act_scales, float* scales,
                         size_t rows, size_t cols, int group_size);

/* GPTQ: Post-training quantization */
int SNEPPX_gptq_quantize(float* weights, size_t rows, size_t cols,
                          int group_size, int bits,
                          SNEPPXDtype* qweight, float* scales,
                          int32_t* zeros, int sym);
int SNEPPX_gptq_compute_hessian(float* hessian, const float* activation,
                                 size_t n, size_t dim);
int SNEPPX_gptq_quantize_block(float* w, float* h_inv,
                                float* qw, float* scale, int32_t* zero,
                                size_t block_size, int bits, int sym);

/* Utility */
int SNEPPX_quant_params_create(SNEPPXQuantParams* params, SNEPPXQuantMode mode);
const char* SNEPPX_quant_mode_name(SNEPPXQuantMode mode);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_QUANTIZATION_H */
