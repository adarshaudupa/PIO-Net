// ==============================================================================
// PIO-Net Stage-2 Feature Extractor: 16 kHz Audio -> [39 x 16] Mel-Spectrogram
// Matches Gigi's PyTorch compute_mel_spectrogram() in stage2/train_stage2.py
// ==============================================================================

#ifndef STAGE2_FEATURES_H
#define STAGE2_FEATURES_H

#include <stdint.h>
#include <stdbool.h>

#define STAGE2_SAMPLE_RATE      16000
#define STAGE2_TOTAL_SAMPLES    16000   // 1.0 second
#define STAGE2_FRAME_LEN        512     // 32 ms window
#define STAGE2_HOP_LEN          400     // 25 ms hop
#define STAGE2_NUM_FRAMES       39
#define STAGE2_NUM_MELS         16

/**
 * @brief Precomputes Hann window, bit reversal, FFT twiddle factors, and Mel filterbank LUT.
 */
void stage2_features_init(void);

/**
 * @brief Extracts [39 x 16] float Mel-spectrogram from 1.0 second of 16 kHz PCM audio.
 * @param audio_pcm 16,000 int16_t PCM audio samples
 * @param out_spec  Output buffer for 39 x 16 = 624 float spectrogram features
 */
void stage2_extract_spectrogram(const int16_t *audio_pcm, float *out_spec);

/**
 * @brief Returns the spectral standard deviation of the last extracted spectrogram.
 */
float stage2_features_get_last_std_dev(void);

#endif // STAGE2_FEATURES_H
