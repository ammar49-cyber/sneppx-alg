#include "arix_hss.h"

void arix_hss_hierarchical_scan(const ArixHSSLayer* layer, const ArixTensor* x_seq, ArixTensor* y_seq) {
    /* For v0.1, hierarchical scan delegates to standard scan.
       Future multi-scale implementation:
       - Coarse scale: low-resolution state updates every N steps
       - Fine scale: high-resolution updates between coarse steps
       - Cross-scale fusion: combine coarse and fine predictions */
    size_t seq_len = x_seq->shape[0];
    size_t s_dim = layer->A_bar->shape[0];

    size_t shape_hs[] = {seq_len, s_dim};
    ArixTensor* h_seq = arix_tensor_create(shape_hs, 2, ARIX_FLOAT32);

    if (h_seq) {
        arix_hss_scan(layer, x_seq, h_seq, y_seq);
        arix_tensor_destroy(h_seq);
    }
}
