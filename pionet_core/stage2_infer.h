#ifndef STAGE2_INFER_H
#define STAGE2_INFER_H

#include <stdint.h>

// Runs DS-CNN forward pass over a [39 x 16] float spectrogram
// Returns predicted class index (0 to 11)
int8_t stage2_classify(const float *spectrogram_features);

// Returns the logit / score of the predicted class
float stage2_get_last_score(void);

#endif