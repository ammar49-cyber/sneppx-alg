#ifndef SNEPPX_RECURRENT_NEURAL_NETWORK_H
#define SNEPPX_RECURRENT_NEURAL_NETWORK_H
#include <stddef.h>
#include <stdbool.h>
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
