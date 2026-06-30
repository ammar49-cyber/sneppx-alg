#ifndef ARIX_CHECKPOINT_FORMAT_H
#define ARIX_CHECKPOINT_FORMAT_H
/*
 * Checkpoint File Format — v0.5 (training persistence)
 *
 * PURPOSE: Binary format for saving/loading model weights and optimizer
 * state.  Format: magic | header | tensor records | metadata.
 *   - Magic: 8 bytes (0x41524958 0x434B5054 = "ARIXCKPT")
 *   - Header: version, num_tensors, metadata_offset
 *   - Tensor records: shape, dtype, offset, size (repeated)
 *   - Metadata: JSON-like key-value pairs (optimizer step, epoch, RNG seed)
 *
 * Endianness: little-endian.  Alignment: 8-byte boundaries.
 * Forward compat: version field allows readers to reject newer formats.
 *
 * DEPENDENCIES: multidimensional_tensor_engine.h
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ARIX_CKPT_MAGIC      0x41524958  /* "ARIX" */
#define ARIX_CKPT_MAGIC_HI   0x434B5054  /* "CKPT" */
#define ARIX_CKPT_VERSION    1

#pragma pack(push, 1)
typedef struct {
    uint32_t magic_lo;
    uint32_t magic_hi;
    uint32_t version;
    uint32_t num_tensors;
    uint64_t metadata_offset;
    uint64_t metadata_size;
    uint64_t total_size;
    uint8_t  reserved[32];
} ArixCheckpointHeader;

typedef struct {
    uint64_t shape[8];
    uint32_t ndim;
    uint32_t dtype;
    uint64_t data_offset;
    uint64_t data_size;
    uint64_t stride[8];
} ArixTensorRecord;
#pragma pack(pop)

/* ---------- Writer ---------- */
int arix_ckpt_write_open(const char* path, ArixCheckpointHeader* header, void** handle);
int arix_ckpt_write_tensor(void* handle, const void* tensor_data, const ArixTensorRecord* record);
int arix_ckpt_write_metadata(void* handle, const char* metadata_json, size_t json_len);
int arix_ckpt_write_close(void* handle);

/* ---------- Reader ---------- */
int arix_ckpt_read_open(const char* path, ArixCheckpointHeader* header, void** handle);
int arix_ckpt_read_tensor(void* handle, size_t tensor_idx, void* tensor_data, ArixTensorRecord* record);
int arix_ckpt_read_metadata(void* handle, char** metadata_json, size_t* json_len);
int arix_ckpt_read_close(void* handle);

/* ---------- Utility ---------- */
int  arix_ckpt_validate(const char* path);
int  arix_ckpt_supports_version(uint32_t version);

#ifdef __cplusplus
}
#endif

#endif /* ARIX_CHECKPOINT_FORMAT_H */
