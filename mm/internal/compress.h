#ifndef ARIX_INTERNAL_COMPRESS_H
#define ARIX_INTERNAL_COMPRESS_H
/*
 * Memory Compression — v0.5 (internal to arix_memory)
 *
 * PURPOSE: Block-floating-point (BFP) and sparse compression of tensor
 * data to reduce memory bandwidth pressure.  Compression is transparent
 * to the tensor API: a compressed buffer is tagged with codec ID and
 * decompressed on access.
 *
 * BFP: groups elements into blocks, shared exponent per block.
 * Sparse: bitmask of non-zero elements + values array.
 *
 * DEPENDENCIES: multidimensional_tensor_engine.h
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ARIX_COMPRESS_NONE,
    ARIX_COMPRESS_BFP4,      /* block floating-point, 4-bit mantissa */
    ARIX_COMPRESS_BFP8,      /* block floating-point, 8-bit mantissa */
    ARIX_COMPRESS_SPARSE,    /* sparse bitmap encoding */
} ArixCompressionCodec;

typedef struct {
    ArixCompressionCodec  codec;
    void*                 compressed_data;
    size_t                compressed_bytes;
    size_t                original_bytes;
    size_t                original_elements;
    int                   original_dtype;
    size_t                block_size;       /* for BFP */
    void*                 metadata;         /* codec-specific (shared exponents, bitmask) */
} ArixCompressedBuffer;

typedef struct {
    ArixCompressionCodec codec;
    int                  (*compress)(const void* src, size_t src_bytes, int dtype,
                                     ArixCompressedBuffer* dst);
    int                  (*decompress)(const ArixCompressedBuffer* src,
                                       void* dst, size_t dst_bytes);
    void                 (*destroy)(ArixCompressedBuffer* buf);
    const char*          name;
} ArixCompressionCodecImpl;

/* ---------- API ---------- */
int arix_compress_init(void);
int arix_compress_shutdown(void);

int arix_compress_register_codec(const ArixCompressionCodecImpl* codec);
int arix_compress_unregister_codec(ArixCompressionCodec codec);

int arix_compress_apply(const void* src, size_t src_bytes, int dtype,
                        ArixCompressionCodec codec, ArixCompressedBuffer* dst);
int arix_compress_decompress(const ArixCompressedBuffer* src, void* dst, size_t dst_bytes);
void arix_compress_buffer_destroy(ArixCompressedBuffer* buf);

/* ---------- BFP helpers ---------- */
int arix_compress_bfp_block_size(size_t element_bytes, size_t* block_size);

#ifdef __cplusplus
}
#endif

#endif /* ARIX_INTERNAL_COMPRESS_H */
