#ifndef STAGE2_INFER_H
#define STAGE2_INFER_H

#include <stdint.h>

// Runs DS-CNN forward pass over a [39 x 16] int8 spectrogram
// Returns class index (0 to 11)
int8_t stage2_classify(const int8_t *spectrogram_features);

#endif