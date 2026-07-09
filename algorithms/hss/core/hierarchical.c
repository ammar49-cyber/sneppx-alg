#include "hierarchical_state_space.h"

void SNEPPX_hss_hierarchical_scan(const SNEPPXHSSLayer* layer, const SNEPPXTensor* x_seq, SNEPPXTensor* y_seq) {
    /* For v0.1, hierarchical scan delegates to standard scan.
       Future multi-scale implementation:
       - Coarse scale: low-resolution state updates every N steps
       - Fine scale: high-resolution updates between coarse steps
       - Cross-scale fusion: combine coarse and fine predictions */
    size_t seq_len = x_seq->shape[0];
    size_t s_dim = layer->A_bar->shape[0];

    size_t shape_hs[] = {seq_len, s_dim};
    SNEPPXTensor* h_seq = SNEPPX_tensor_create(shape_hs, 2, SNEPPX_FLOAT32);

    if (h_seq) {
        SNEPPX_hss_scan(layer, x_seq, h_seq, y_seq);
        SNEPPX_tensor_destroy(h_seq);
    }
}
