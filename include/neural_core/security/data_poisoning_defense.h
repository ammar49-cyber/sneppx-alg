#ifndef SNEPPX_DATA_POISONING_DEFENSE_H
#define SNEPPX_DATA_POISONING_DEFENSE_H
/*
 * S5 AI Sanitizer — Data Poisoning Defense
 * Detects poisoned training samples, outlier data points, and
 * backdoor triggers in datasets.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_POISON_MAX_FEATURES 1024

typedef struct {
    double* feature_means;
    double* feature_stds;
    int feature_count;
    double outlier_threshold;
    int trained;
} SNEPPXPoisonDetector;

int  SNEPPX_poison_detector_init(SNEPPXPoisonDetector* pd, int feature_count);
void SNEPPX_poison_detector_destroy(SNEPPXPoisonDetector* pd);
int  SNEPPX_poison_detector_train(SNEPPXPoisonDetector* pd,
                                 const double* samples, int sample_count);
int  SNEPPX_poison_detector_score(SNEPPXPoisonDetector* pd,
                                 const double* sample, int feature_count,
                                 double* outlier_score);
int  SNEPPX_poison_detector_is_outlier(SNEPPXPoisonDetector* pd,
                                      const double* sample, int feature_count);

#ifdef __cplusplus
}
#endif
#endif
