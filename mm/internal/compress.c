/*
 * Memory Compression Implementation — SKELETON
 * VERSION: v0.5
 */

#include "compress.h"
#include <stdlib.h>
#include <string.h>

int SNEPPX_compress_init(void) { return 0; }
int SNEPPX_compress_shutdown(void) { return 0; }
int SNEPPX_compress_register_codec(const SNEPPXCompressionCodecImpl* codec) {
    (void)codec; return 0;
}
int SNEPPX_compress_unregister_codec(SNEPPXCompressionCodec codec) { (void)codec; return 0; }
int SNEPPX_compress_apply(const void* src, size_t src_bytes, int dtype,
                        SNEPPXCompressionCodec codec, SNEPPXCompressedBuffer* dst) {
    (void)src; (void)src_bytes; (void)dtype; (void)codec;
    if (dst) memset(dst, 0, sizeof(*dst));
    return 0;
}
int SNEPPX_compress_decompress(const SNEPPXCompressedBuffer* src, void* dst, size_t dst_bytes) {
    (void)src; (void)dst; (void)dst_bytes; return 0;
}
void SNEPPX_compress_buffer_destroy(SNEPPXCompressedBuffer* buf) { free(buf->compressed_data); free(buf->metadata); }
int SNEPPX_compress_bfp_block_size(size_t element_bytes, size_t* block_size) {
    (void)element_bytes; if (block_size) *block_size = 128; return 0;
}
