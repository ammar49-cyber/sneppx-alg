#ifndef SNEPPX_ONNX_FORMAT_H
#define SNEPPX_ONNX_FORMAT_H
#include <stddef.h>
#ifdef __cplusplus
/*
 * SNEPPX - Onnx Format
 *
 * WHAT
 *   Onnx Format.
 *
 * CONCEPT
 *   Provides the Onnx Format.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
/**
 * @brief Load Onnx.
 *
 * @param path [in] Path value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_onnx_load(const char* path);
/**
 * @brief Destroy Onnx.
 *
 * @param model [out] Model value.
 */
void SNEPPX_onnx_destroy(void* model);
/**
 * @brief Perform Onnx Get Input Count.
 *
 * @param model [out] Model value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_onnx_get_input_count(void* model);
/**
 * @brief Perform Onnx Get Input Info.
 *
 * @param model [out] Model value.
 * @param idx [in] Idx value.
 * @param name [out] Name value.
 * @param name_max [in] Name Max value.
 * @param shape [out] Shape value.
 * @param ndim [out] Ndim value.
 * @param dtype [out] Dtype value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_onnx_get_input_info(void* model, int idx, char* name, size_t name_max, size_t** shape, size_t* ndim, int* dtype);
/**
 * @brief Perform Onnx Get Output Count.
 *
 * @param model [out] Model value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_onnx_get_output_count(void* model);
/**
 * @brief Run Onnx.
 *
 * @param model [out] Model value.
 * @param inputs [out] Inputs value.
 * @param num_inputs [in] Num Inputs value.
 * @param outputs [out] Outputs value.
 * @param num_outputs [in] Num Outputs value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_onnx_run(void* model, void** inputs, size_t num_inputs, void** outputs, size_t num_outputs);
/**
 * @brief Perform Onnx Export.
 *
 * @param path [in] Path value.
 * @param graph [out] Graph value.
 * @param input_names [in] Input Names value.
 * @param num_inputs [in] Num Inputs value.
 * @param output_names [in] Output Names value.
 * @param num_outputs [in] Num Outputs value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_onnx_export(const char* path, void* graph, const char** input_names, size_t num_inputs, const char** output_names, size_t num_outputs);
/**
 * @brief Perform Onnx Check.
 *
 * @param path [in] Path value.
 * @param error_msg [out] Error Msg value.
 * @param error_max [in] Error Max value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_onnx_check(const char* path, char* error_msg, size_t error_max);
#ifdef __cplusplus
}
#endif
#endif
