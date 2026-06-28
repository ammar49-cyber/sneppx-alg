#include "arix_arc.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

void arix_arc_simulate_attack(const ArixTensor* clean_input, int attack_type, float epsilon, ArixTensor** adversarial) {
    size_t batch = clean_input->shape[0];
    size_t dim = clean_input->ndim > 1 ? clean_input->shape[1] : clean_input->shape[0];
    size_t total = clean_input->size;

    size_t shape_a[] = {batch, dim};
    *adversarial = arix_tensor_create(shape_a, clean_input->ndim, ARIX_FLOAT32);
    if (!*adversarial) return;

    float* clean = (float*)clean_input->data;
    float* adv = (float*)(*adversarial)->data;

    unsigned long state = 54321;

    if (attack_type == ARIX_ATTACK_FGSM) {
        for (size_t i = 0; i < total; i++) {
            state = state * 1103515245UL + 12345UL;
            float sign = ((state >> 16) & 1) ? 1.0f : -1.0f;
            adv[i] = clean[i] + epsilon * sign;
            if (adv[i] > 1.0f) adv[i] = 1.0f;
            if (adv[i] < -1.0f) adv[i] = -1.0f;
        }
    } else if (attack_type == ARIX_ATTACK_PGD) {
        for (size_t i = 0; i < total; i++) {
            state = state * 1103515245UL + 12345UL;
            float rnd = ((float)((state >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 2.0f * epsilon;
            adv[i] = clean[i] + rnd;
            if (adv[i] > 1.0f) adv[i] = 1.0f;
            if (adv[i] < -1.0f) adv[i] = -1.0f;
        }
        int num_iters = 3;
        float step = epsilon / (float)num_iters;
        for (int iter = 0; iter < num_iters; iter++) {
            for (size_t i = 0; i < total; i++) {
                state = state * 1103515245UL + 12345UL;
                float sign = ((state >> 16) & 1) ? 1.0f : -1.0f;
                adv[i] += step * sign;
                if (adv[i] > clean[i] + epsilon) adv[i] = clean[i] + epsilon;
                if (adv[i] < clean[i] - epsilon) adv[i] = clean[i] - epsilon;
                if (adv[i] > 1.0f) adv[i] = 1.0f;
                if (adv[i] < -1.0f) adv[i] = -1.0f;
            }
        }
    } else if (attack_type == ARIX_ATTACK_CW) {
        for (size_t i = 0; i < total; i++) {
            state = state * 1103515245UL + 12345UL;
            float sign = ((state >> 16) & 1) ? 1.0f : -1.0f;
            float l2_penalty = 0.5f;
            adv[i] = clean[i] + epsilon * sign * l2_penalty;
            if (adv[i] > 1.0f) adv[i] = 1.0f;
            if (adv[i] < -1.0f) adv[i] = -1.0f;
        }
    }
}
