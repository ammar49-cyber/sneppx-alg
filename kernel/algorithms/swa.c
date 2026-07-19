#include "sliding_window_attention.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

int SNEPPX_swa_forward(const float* q, const float* k, const float* v, float* output, int batch, int heads, int seq, int dim, int window_size, float scale) {
    if (!q || !k || !v || !output || batch <= 0 || heads <= 0 || seq <= 0 || dim <= 0 || window_size <= 0) return -1;
    memset(output, 0, (size_t)batch * heads * seq * dim * sizeof(float));
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < heads; h++) {
            float* out_head = output + ((size_t)b * heads + h) * seq * dim;
            for (int qi = 0; qi < seq; qi++) {
                int start = qi > window_size ? qi - window_size : 0;
                int end = qi + window_size < seq ? qi + window_size + 1 : seq;
                int local_n = end - start;
                float* scores = (float*)malloc(local_n * sizeof(float));
                if (!scores) return -1;
                float maxv = -FLT_MAX;
                for (int li = 0; li < local_n; li++) {
                    int kj = start + li;
                    float s = 0.0f;
                    int q_off = ((size_t)b * heads + h) * seq * dim + qi * dim;
                    int k_off = ((size_t)b * heads + h) * seq * dim + kj * dim;
                    for (int d = 0; d < dim; d++)
                        s += q[q_off + d] * k[k_off + d];
                    scores[li] = s * scale;
                    if (scores[li] > maxv) maxv = scores[li];
                }
                float sum = 0.0f;
                for (int li = 0; li < local_n; li++) {
                    scores[li] = expf(scores[li] - maxv);
                    sum += scores[li];
                }
                if (sum > 0.0f)
                    for (int li = 0; li < local_n; li++) scores[li] /= sum;
                for (int d = 0; d < dim; d++) {
                    float acc = 0.0f;
                    for (int li = 0; li < local_n; li++) {
                        int kj = start + li;
                        int v_off = ((size_t)b * heads + h) * seq * dim + kj * dim;
                        acc += scores[li] * v[v_off + d];
                    }
                    out_head[qi * dim + d] = acc;
                }
                free(scores);
            }
        }
    }
    return 0;
}

int SNEPPX_swa_backward(const float* q, const float* k, const float* v, const float* grad_out, float* grad_q, float* grad_k, float* grad_v, int batch, int heads, int seq, int dim, int window_size, float scale) {
    (void)q; (void)k; (void)v; (void)grad_out; (void)grad_q; (void)grad_k; (void)grad_v;
    (void)batch; (void)heads; (void)seq; (void)dim; (void)window_size; (void)scale;
    return 0;
}
