#include "position_encoding.h"
#include <math.h>

int SNEPPX_alibi_compute_slopes(float* slopes, int num_heads, float slope_base) {
    if (!slopes || num_heads <= 0) return -1;
    for (int h = 0; h < num_heads; h++)
        slopes[h] = 1.0f / powf(slope_base, (float)(h + 1));
    return 0;
}

int SNEPPX_alibi_apply(float* attn_scores, const float* slopes, int batch, int heads, int seq_q, int seq_k) {
    if (!attn_scores || !slopes || batch <= 0 || heads <= 0 || seq_q <= 0 || seq_k <= 0) return -1;
    for (int h = 0; h < heads; h++) {
        float slope = slopes[h];
        for (int b = 0; b < batch; b++) {
            for (int qi = 0; qi < seq_q; qi++) {
                for (int kj = 0; kj < seq_k; kj++) {
                    int idx = ((b * heads + h) * seq_q + qi) * seq_k + kj;
                    attn_scores[idx] += slope * (float)(kj - qi);
                }
            }
        }
    }
    return 0;
}
