#include "checkpoint_reader.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if defined(_WIN32) || defined(_WIN64)
    #define FSEEK64(fp, offset, origin) _fseeki64(fp, offset, origin)
    #define OFF64_T __int64
#else
    #define FSEEK64(fp, offset, origin) fseeko64(fp, offset, origin)
    #define OFF64_T off64_t
#endif

/*
 * SNEPPX - Checkpoint Reader
 *
 * WHAT
 *   Checkpoint Reader.
 *
 * CONCEPT
 *   Provides checkpointing and fault tolerance.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct {
    FILE* fp;
    SNEPPXCheckpointHeader header;
    SNEPPXTensorRecord* records;
    uint32_t num_records;
} SNEPPXCkptHandle;

/**
 * @brief Open Ckpt Write.
 *
 * @param path [in] Path value.
 * @param header [out] Header value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ckpt_write_open(const char* path, SNEPPXCheckpointHeader* header, void** handle) {
    if (!path || !header || !handle) return -1;
    FILE* fp = fopen(path, "wb");
    if (!fp) return -1;
    SNEPPXCkptHandle* h = (SNEPPXCkptHandle*)calloc(1, sizeof(SNEPPXCkptHandle));
    if (!h) { fclose(fp); return -1; }
    h->fp = fp;
    memcpy(&h->header, header, sizeof(SNEPPXCheckpointHeader));
    h->header.magic_lo = SNEPPX_CKPT_MAGIC;
    h->header.magic_hi = SNEPPX_CKPT_MAGIC_HI;
    h->header.version = SNEPPX_CKPT_VERSION;
    
    // Write header
    fwrite(&h->header, sizeof(SNEPPXCheckpointHeader), 1, fp);
    
    // Write placeholder records (will be filled in write_tensor)
    if (header->num_tensors > 0) {
        h->records = (SNEPPXTensorRecord*)calloc(header->num_tensors, sizeof(SNEPPXTensorRecord));
        if (!h->records) { fclose(fp); return -1; }
        h->num_records = header->num_tensors;
        // Write placeholder records
        fwrite(h->records, sizeof(SNEPPXTensorRecord), header->num_tensors, fp);
    }
    *handle = h;
    return 0;
}

/**
 * @brief Perform Ckpt Write Tensor.
 *
 * @param handle [out] Handle value.
 * @param tensor_data [in] Tensor Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ckpt_write_tensor(void* handle, const void* tensor_data, const SNEPPXTensorRecord* record) {
    if (!handle || !tensor_data || !record) return -1;
    SNEPPXCkptHandle* h = (SNEPPXCkptHandle*)handle;
    if (h->num_records == 0) return -1;
    uint32_t idx = h->header.num_tensors - h->num_records;
    
    // Get current file position (this is where tensor data will be written)
    FSEEK64(h->fp, 0, SEEK_END);
    OFF64_T data_offset = ftell(h->fp);
    
    // Update the record in memory with the data offset
    h->records[idx] = *record;
    h->records[idx].data_offset = (uint64_t)data_offset;
    
    // Update the record in the file (seek to record position in the records array)
    uint64_t record_pos = sizeof(SNEPPXCheckpointHeader) + idx * sizeof(SNEPPXTensorRecord);
    FSEEK64(h->fp, (OFF64_T)record_pos, SEEK_SET);
    fwrite(&h->records[idx], sizeof(SNEPPXTensorRecord), 1, h->fp);
    
    // Seek back to end and write tensor data
    FSEEK64(h->fp, 0, SEEK_END);
    fwrite(tensor_data, 1, record->data_size, h->fp);
    
    h->num_records--;
    return 0;
}

/**
 * @brief Perform Ckpt Write Metadata.
 *
 * @param handle [out] Handle value.
 * @param metadata_json [in] Metadata Json value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ckpt_write_metadata(void* handle, const char* metadata_json, size_t json_len) {
    if (!handle || !metadata_json) return -1;
    SNEPPXCkptHandle* h = (SNEPPXCkptHandle*)handle;
    FSEEK64(h->fp, 0, SEEK_END);
    h->header.metadata_offset = (uint64_t)ftell(h->fp);
    h->header.metadata_size = (uint64_t)json_len;
    fwrite(metadata_json, 1, json_len, h->fp);
    return 0;
}

/**
 * @brief Close Ckpt Write.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ckpt_write_close(void* handle) {
    if (!handle) return -1;
    SNEPPXCkptHandle* h = (SNEPPXCkptHandle*)handle;
    
    // Update header with final total_size
    FSEEK64(h->fp, 0, SEEK_END);
    long end_pos = ftell(h->fp);
    h->header.total_size = (uint64_t)end_pos;
    
    // Write final header
    FSEEK64(h->fp, 0, SEEK_SET);
    fwrite(&h->header, sizeof(SNEPPXCheckpointHeader), 1, h->fp);
    
    fclose(h->fp);
    free(h->records);
    free(h);
    return 0;
}

/**
 * @brief Open Ckpt Read.
 *
 * @param path [in] Path value.
 * @param header [out] Header value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ckpt_read_open(const char* path, SNEPPXCheckpointHeader* header, void** handle) {
    if (!path || !header || !handle) return -1;
    FILE* fp = fopen(path, "rb");
    if (!fp) return -1;
    SNEPPXCkptHandle* h = (SNEPPXCkptHandle*)calloc(1, sizeof(SNEPPXCkptHandle));
    if (!h) { fclose(fp); return -1; }
    h->fp = fp;
    if (fread(&h->header, sizeof(SNEPPXCheckpointHeader), 1, fp) != 1) {
        fclose(fp); free(h); return -1;
    }
    if (h->header.magic_lo != SNEPPX_CKPT_MAGIC || h->header.magic_hi != SNEPPX_CKPT_MAGIC_HI) {
        fclose(fp); free(h); return -1;
    }
    memcpy(header, &h->header, sizeof(SNEPPXCheckpointHeader));
    if (h->header.num_tensors > 0) {
        h->records = (SNEPPXTensorRecord*)calloc(h->header.num_tensors, sizeof(SNEPPXTensorRecord));
        if (!h->records) { fclose(fp); free(h); return -1; }
        size_t nread = fread(h->records, sizeof(SNEPPXTensorRecord), h->header.num_tensors, fp);
        if (nread != h->header.num_tensors) {
            fclose(fp); free(h->records); free(h); return -1;
        }
    }
    *handle = h;
    return 0;
}

/**
 * @brief Perform Ckpt Read Tensor.
 *
 * @param handle [out] Handle value.
 * @param tensor_idx [in] Tensor Idx value.
 * @param tensor_data [out] Tensor Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ckpt_read_tensor(void* handle, size_t tensor_idx, void* tensor_data, SNEPPXTensorRecord* record) {
    if (!handle || !tensor_data) return -1;
    SNEPPXCkptHandle* h = (SNEPPXCkptHandle*)handle;
    if (tensor_idx >= h->header.num_tensors) return -1;
    if (record) memcpy(record, &h->records[tensor_idx], sizeof(SNEPPXTensorRecord));
    FSEEK64(h->fp, (OFF64_T)h->records[tensor_idx].data_offset, SEEK_SET);
    size_t nread = fread(tensor_data, 1, h->records[tensor_idx].data_size, h->fp);
    if (nread != h->records[tensor_idx].data_size) return -1;
    return 0;
}

/**
 * @brief Perform Ckpt Read Metadata.
 *
 * @param handle [out] Handle value.
 * @param metadata_json [out] Metadata Json value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ckpt_read_metadata(void* handle, char** metadata_json, size_t* json_len) {
    if (!handle || !metadata_json || !json_len) return -1;
    SNEPPXCkptHandle* h = (SNEPPXCkptHandle*)handle;
    if (h->header.metadata_size == 0) { *metadata_json = NULL; *json_len = 0; return 0; }
    *metadata_json = (char*)malloc(h->header.metadata_size + 1);
    if (!*metadata_json) return -1;
    FSEEK64(h->fp, (OFF64_T)h->header.metadata_offset, SEEK_SET);
    size_t nread = fread(*metadata_json, 1, h->header.metadata_size, h->fp);
    if (nread != h->header.metadata_size) { free(*metadata_json); *metadata_json = NULL; return -1; }
    (*metadata_json)[h->header.metadata_size] = '\0';
    *json_len = h->header.metadata_size;
    return 0;
}

/**
 * @brief Close Ckpt Read.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ckpt_read_close(void* handle) {
    if (!handle) return -1;
    SNEPPXCkptHandle* h = (SNEPPXCkptHandle*)handle;
    fclose(h->fp);
    free(h->records);
    free(h);
    return 0;
}

/**
 * @brief Perform Ckpt Validate.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ckpt_validate(const char* path) {
    if (!path) return -1;
    FILE* fp = fopen(path, "rb");
    if (!fp) return -1;
    SNEPPXCheckpointHeader header;
    if (fread(&header, sizeof(SNEPPXCheckpointHeader), 1, fp) != 1) {
        fclose(fp); return -1;
    }
    fclose(fp);
    if (header.magic_lo != SNEPPX_CKPT_MAGIC || header.magic_hi != SNEPPX_CKPT_MAGIC_HI) return -1;
    if (header.version > SNEPPX_CKPT_VERSION) return -1;
    return 0;
}

/**
 * @brief Perform Ckpt Supports Version.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ckpt_supports_version(uint32_t version) {
    return version <= SNEPPX_CKPT_VERSION ? 1 : 0;
}
