#ifndef STAGE2_INFER_H
#define STAGE2_INFER_H

#include <stdint.h>

// Runs DS-CNN forward pass over quantized [39 x 16] int8_t Mel-spectrogram
// Returns predicted class index (0 to 11)
int8_t stage2_classify(const int8_t *spectrogram_features);

// Direct high-precision Q8 fixed-point (scale 256.0f) inference pass
int8_t stage2_classify_q8(const int16_t *in_q8);

// Convenience wrapper for float spectrogram features (direct high-precision Q8 conversion)
int8_t stage2_classify_float(const float *spectrogram_features);

// Returns the logit / score of the top-1 predicted class
float  stage2_get_last_score(void);

// Returns the top-2 runner-up class index and score
int8_t stage2_get_second_class(void);
float  stage2_get_second_score(void);

#endif