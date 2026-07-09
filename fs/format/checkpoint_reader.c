/*
 * Checkpoint Format Implementation — SKELETON
 * VERSION: v0.5
 */

#include "checkpoint_reader.h"
#include <stdlib.h>
#include <string.h>

int SNEPPX_ckpt_write_open(const char* path, SNEPPXCheckpointHeader* header, void** handle) {
    (void)path; (void)header; if (handle) *handle = NULL; return 0;
}
int SNEPPX_ckpt_write_tensor(void* handle, const void* tensor_data, const SNEPPXTensorRecord* record) {
    (void)handle; (void)tensor_data; (void)record; return 0;
}
int SNEPPX_ckpt_write_metadata(void* handle, const char* metadata_json, size_t json_len) {
    (void)handle; (void)metadata_json; (void)json_len; return 0;
}
int SNEPPX_ckpt_write_close(void* handle) { (void)handle; return 0; }

int SNEPPX_ckpt_read_open(const char* path, SNEPPXCheckpointHeader* header, void** handle) {
    (void)path; if (header) memset(header, 0, sizeof(*header)); if (handle) *handle = NULL; return 0;
}
int SNEPPX_ckpt_read_tensor(void* handle, size_t tensor_idx, void* tensor_data, SNEPPXTensorRecord* record) {
    (void)handle; (void)tensor_idx; (void)tensor_data; (void)record; return 0;
}
int SNEPPX_ckpt_read_metadata(void* handle, char** metadata_json, size_t* json_len) {
    (void)handle; if (metadata_json) *metadata_json = NULL; if (json_len) *json_len = 0; return 0;
}
int SNEPPX_ckpt_read_close(void* handle) { (void)handle; return 0; }

int SNEPPX_ckpt_validate(const char* path) { (void)path; return 0; }
int SNEPPX_ckpt_supports_version(uint32_t version) { (void)version; return 1; }
