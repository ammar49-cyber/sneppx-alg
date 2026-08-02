#ifndef SNEPPX_REINFORCEMENT_LEARNING_H
#define SNEPPX_REINFORCEMENT_LEARNING_H
#include <stddef.h>
#include <stdbool.h>
/*
 * SNEPPX - Reinforcement Learning
 *
 * WHAT
 *   Reinforcement Learning.
 *
 * CONCEPT
 *   RL configuration and training API for the SNEPPX-Algo system.
 *
 * ROLE
 *   Declares the RL API for reinforcement learning operations within the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal module).
 */


#ifdef __cplusplus
extern "C" {
#endif

typedef struct SNEPPXRLAgent SNEPPXRLAgent;

SNEPPXRLAgent* SNEPPX_rl_create(size_t state_dim, size_t action_dim, size_t hidden_dim,
    float lr, float gamma);
void SNEPPX_rl_destroy(void* agent);
int SNEPPX_rl_select_action(void* agent, const float* state, float* action);
int SNEPPX_rl_update(void* agent, const float* state, const float* action, float reward,
    const float* next_state, int done);

#ifdef __cplusplus
}
#endif
#endif
