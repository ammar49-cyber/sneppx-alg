#ifndef SNEPPX_STRUTIL_H
#define SNEPPX_STRUTIL_H
/*
 * Safe String Utilities — v0.5 (generic library)
 *
 * PURPOSE: Bounds-checked replacements for standard C string functions.
 * Used in security-critical paths where buffer overflows are unacceptable.
 *
 * All functions guarantee null-termination of the destination buffer
 * as long as dst_cap > 0.
 *
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char*  buf;
    size_t capacity;
    size_t length;
} SNEPPXStringBuf;

SNEPPXStringBuf* SNEPPX_strbuf_create(size_t capacity);
void           SNEPPX_strbuf_destroy(SNEPPXStringBuf* sb);

int  SNEPPX_strbuf_append(SNEPPXStringBuf* sb, const char* src);
int  SNEPPX_strbuf_append_n(SNEPPXStringBuf* sb, const char* src, size_t n);
int  SNEPPX_strbuf_format(SNEPPXStringBuf* sb, const char* fmt, ...);
void SNEPPX_strbuf_clear(SNEPPXStringBuf* sb);

/* ---------- Safe C string replacements ---------- */
size_t SNEPPX_strlcpy(char* dst, const char* src, size_t dst_cap);
size_t SNEPPX_strlcat(char* dst, const char* src, size_t dst_cap);
int    SNEPPX_strcmp(const char* a, const char* b);
char*  SNEPPX_strdup_s(const char* src);

/* ---------- Split / join ---------- */
size_t SNEPPX_strsplit(const char* str, char delimiter, char*** out_tokens, size_t max_tokens);
char*  SNEPPX_strjoin(const char** tokens, size_t num_tokens, char delimiter);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_STRUTIL_H */
