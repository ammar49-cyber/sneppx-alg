#ifndef SNEPPX_S8_EXTENSIONS_H
#define SNEPPX_S8_EXTENSIONS_H
/* S8 extensions: TLA+ parser, LTL model checker, symbolic execution,
   loop invariant inference, data flow analysis, Lean 4 proof export */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_TLA_MAX_SPEC 4096
#define SNEPPX_LTL_MAX_FORMULA 256
#define SNEPPX_SYMEX_MAX_PATHS 1024

/* TLA+ specification parser (simplified) */
typedef struct {
    char spec[SNEPPX_TLA_MAX_SPEC];
    int parsed;
    int state_count;
} SNEPPXTLAParser;

int  SNEPPX_tla_parse(SNEPPXTLAParser* parser, const char* spec_text);

/* LTL property verifier */
typedef struct {
    char formula[SNEPPX_LTL_MAX_FORMULA];
    int holds;
} SNEPPXLTLVerifier;

int  SNEPPX_ltl_init(SNEPPXLTLVerifier* ltl, const char* formula);
int  SNEPPX_ltl_check(SNEPPXLTLVerifier* ltl, int* trace, int trace_len);

/* Symbolic execution engine */
typedef struct {
    uint64_t explored_paths;
    uint64_t bounded_paths;
    int depth_limit;
} SNEPPXSymExEngine;

int  SNEPPX_symex_init(SNEPPXSymExEngine* se, int depth_limit);
int  SNEPPX_symex_explore(SNEPPXSymExEngine* se, const uint8_t* bytecode, size_t bc_len);

/* Loop invariant inference */
int  SNEPPX_loop_invariant_infer(const char* loop_body, char* invariant_out, size_t inv_size);

/* Data flow analysis */
typedef struct {
    int taint_marks[256];
    int taint_count;
} SNEPPXDataFlow;

int  SNEPPX_data_flow_init(SNEPPXDataFlow* df);
int  SNEPPX_data_flow_taint(SNEPPXDataFlow* df, int var_id);
int  SNEPPX_data_flow_propagate(SNEPPXDataFlow* df);

/* Lean 4 proof export */
int  SNEPPX_lean_export_proof(const char* theorem_name, const char* proof_body, const char* output_path);

#ifdef __cplusplus
}
#endif
#endif
