#ifndef ARIX_STRUTIL_H
#define ARIX_STRUTIL_H
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
} ArixStringBuf;

ArixStringBuf* arix_strbuf_create(size_t capacity);
void           arix_strbuf_destroy(ArixStringBuf* sb);

int  arix_strbuf_append(ArixStringBuf* sb, const char* src);
int  arix_strbuf_append_n(ArixStringBuf* sb, const char* src, size_t n);
int  arix_strbuf_format(ArixStringBuf* sb, const char* fmt, ...);
void arix_strbuf_clear(ArixStringBuf* sb);

/* ---------- Safe C string replacements ---------- */
size_t arix_strlcpy(char* dst, const char* src, size_t dst_cap);
size_t arix_strlcat(char* dst, const char* src, size_t dst_cap);
int    arix_strcmp(const char* a, const char* b);
char*  arix_strdup_s(const char* src);

/* ---------- Split / join ---------- */
size_t arix_strsplit(const char* str, char delimiter, char*** out_tokens, size_t max_tokens);
char*  arix_strjoin(const char** tokens, size_t num_tokens, char delimiter);

#ifdef __cplusplus
}
#endif

#endif /* ARIX_STRUTIL_H */
