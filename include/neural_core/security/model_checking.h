#ifndef SNEPPX_MODEL_CHECKING_H
#define SNEPPX_MODEL_CHECKING_H
/*
 * S8 Formal Verification — Model Checking Framework
 * Lightweight model checking for crypto primitives, data flow, and invariants.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_MODEL_MAX_STATES 1024
#define SNEPPX_MODEL_MAX_TRANSITIONS 4096
#define SNEPPX_MODEL_PROP_MAX_LEN 256

typedef struct {
    uint32_t state_id;
    uint32_t next_states[8];
    int next_count;
    int is_accepting;
    int is_error;
} SNEPPXModelState;

typedef struct {
    SNEPPXModelState states[SNEPPX_MODEL_MAX_STATES];
    int state_count;
    int initial_state;
    char property[SNEPPX_MODEL_PROP_MAX_LEN];
} SNEPPXFormalModel;

typedef struct {
    int total_states;
    int reachable_states;
    int deadlock_states;
    int error_states;
    int property_satisfied;
    char counterexample[SNEPPX_MODEL_PROP_MAX_LEN];
} SNEPPXModelCheckResult;

int  SNEPPX_model_init(SNEPPXFormalModel* model);
int  SNEPPX_model_add_state(SNEPPXFormalModel* model, uint32_t state_id, int is_accepting, int is_error);
int  SNEPPX_model_add_transition(SNEPPXFormalModel* model, uint32_t from, uint32_t to);
int  SNEPPX_model_set_property(SNEPPXFormalModel* model, const char* property);
SNEPPXModelCheckResult SNEPPX_model_check(SNEPPXFormalModel* model);
int  SNEPPX_model_verify_invariant(SNEPPXFormalModel* model, int (*invariant)(uint32_t state_id));

#ifdef __cplusplus
}
#endif
#endif
