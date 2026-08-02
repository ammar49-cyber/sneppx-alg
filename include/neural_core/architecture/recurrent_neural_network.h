#ifndef SNEPPX_RECURRENT_NEURAL_NETWORK_H
#define SNEPPX_RECURRENT_NEURAL_NETWORK_H
#include <stddef.h>
#include <stdbool.h>
/*
 * SNEPPX - Recurrent Neural Network
 *
 * WHAT
 *   Recurrent Neural Network.
 *
 * CONCEPT
 *   RNN configuration and forward API for sequential data.
 *
 * ROLE
 *   Declares the RNN API for recurrent neural network operations within the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal module).
 */


#ifdef __cplusplus
extern "C" {
#endif

typedef struct SNEPPXRNN SNEPPXRNN;

SNEPPXRNN* SNEPPX_rnn_create(size_t input_size, size_t hidden_size, size_t num_layers,
    int bidirectional, float dropout, const char* rnn_type);
void SNEPPX_rnn_destroy(void* rnn);
int SNEPPX_rnn_forward(void* rnn, const float* input, size_t seq_len, size_t batch_size,
    float* output, float* hidden);

#ifdef __cplusplus
}
#endif
#endif
