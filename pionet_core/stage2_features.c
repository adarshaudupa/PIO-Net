// ==============================================================================
// PIO-Net Stage-2 Feature Extractor: 16 kHz Audio -> [39 x 16] Mel-Spectrogram
// Matches Gigi's PyTorch compute_mel_spectrogram() in stage2/train_stage2.py
// ==============================================================================

#include "stage2_features.h"
#include <math.h>
#include <string.h>

#define PI_F 3.14159265358979323846f

static float hann_window[STAGE2_FRAME_LEN];
static uint16_t bit_rev[STAGE2_FRAME_LEN];
static float twiddle_cos[STAGE2_FRAME_LEN / 2];
static float twiddle_sin[STAGE2_FRAME_LEN / 2];
static bool initialized = false;

// 18 bin edges defining 16 triangular Mel filterbank channels (from 100 Hz to 7000 Hz)
static const uint16_t bin_edges[18] = {
    2, 16, 31, 46, 60, 75, 90, 104, 119, 134, 149, 163, 178, 193, 207, 222, 237, 252
};

void stage2_features_init(void) {
    if (initialized) return;

    // 1. Precompute Hann window
    for (int i = 0; i < STAGE2_FRAME_LEN; i++) {
        hann_window[i] = 0.5f * (1.0f - cosf(2.0f * PI_F * (float)i / (float)STAGE2_FRAME_LEN));
    }

    // 2. Precompute bit reversal permutation table (for N = 512 = 2^9)
    for (uint16_t i = 0; i < STAGE2_FRAME_LEN; i++) {
        uint16_t rev = 0;
        uint16_t temp = i;
        for (int b = 0; b < 9; b++) {
            rev = (rev << 1) | (temp & 1);
            temp >>= 1;
        }
        bit_rev[i] = rev;
    }

    // 3. Precompute twiddle factors W_N^k = e^(-j * 2 * pi * k / 512)
    for (int k = 0; k < STAGE2_FRAME_LEN / 2; k++) {
        float angle = -2.0f * PI_F * (float)k / (float)STAGE2_FRAME_LEN;
        twiddle_cos[k] = cosf(angle);
        twiddle_sin[k] = sinf(angle);
    }

    initialized = true;
}

// In-place Cooley-Tukey Radix-2 Real/Complex FFT (N = 512)
static void fft512(float *real, float *imag) {
    // Bit-reversal permutation
    for (uint16_t i = 0; i < STAGE2_FRAME_LEN; i++) {
        uint16_t j = bit_rev[i];
        if (i < j) {
            float tr = real[i]; real[i] = real[j]; real[j] = tr;
            float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
        }
    }

    // 9 Butterfly stages: len = 2, 4, 8, ... 512
    for (int len = 2; len <= STAGE2_FRAME_LEN; len <<= 1) {
        int half_len = len >> 1;
        int step = STAGE2_FRAME_LEN / len;

        for (int i = 0; i < STAGE2_FRAME_LEN; i += len) {
            int tw_idx = 0;
            for (int j = 0; j < half_len; j++) {
                float c = twiddle_cos[tw_idx];
                float s = twiddle_sin[tw_idx];
                tw_idx += step;

                int u_idx = i + j;
                int v_idx = i + j + half_len;

                float vr = real[v_idx];
                float vi = imag[v_idx];

                // Complex multiplication: (vr + j*vi) * (c + j*s)
                float wr = vr * c - vi * s;
                float wi = vr * s + vi * c;

                float ur = real[u_idx];
                float ui = imag[u_idx];

                real[u_idx] = ur + wr;
                imag[u_idx] = ui + wi;
                real[v_idx] = ur - wr;
                imag[v_idx] = ui - wi;
            }
        }
    }
}

void stage2_extract_spectrogram(const int16_t *audio_pcm, int8_t *out_spec) {
    if (!initialized) {
        stage2_features_init();
    }

    static float real_buf[STAGE2_FRAME_LEN];
    static float imag_buf[STAGE2_FRAME_LEN];
    static float mag_buf[STAGE2_FRAME_LEN / 2 + 1];
    static float mel_matrix[STAGE2_NUM_FRAMES * STAGE2_NUM_MELS];

    // Process all 39 frames (hop = 400 samples, window = 512 samples)
    for (int f = 0; f < STAGE2_NUM_FRAMES; f++) {
        const int16_t *frame_in = &audio_pcm[f * STAGE2_HOP_LEN];

        // Apply Hann window and convert int16 audio to float
        for (int i = 0; i < STAGE2_FRAME_LEN; i++) {
            real_buf[i] = ((float)frame_in[i] / 32768.0f) * hann_window[i];
            imag_buf[i] = 0.0f;
        }

        // Run 512-point FFT
        fft512(real_buf, imag_buf);

        // Compute magnitude for first 257 frequency bins (0 to 8000 Hz)
        for (int k = 0; k <= STAGE2_FRAME_LEN / 2; k++) {
            float r = real_buf[k];
            float im = imag_buf[k];
            mag_buf[k] = sqrtf(r * r + im * im);
        }

        // Apply 16 triangular Mel filters
        for (int m = 0; m < STAGE2_NUM_MELS; m++) {
            uint16_t left   = bin_edges[m];
            uint16_t center = bin_edges[m + 1];
            uint16_t right  = bin_edges[m + 2];

            float energy = 0.0f;
            float left_denom  = (float)(center - left);
            float right_denom = (float)(right - center);

            for (uint16_t k = left; k < center; k++) {
                float weight = (float)(k - left) / left_denom;
                energy += weight * mag_buf[k];
            }
            for (uint16_t k = center; k <= right; k++) {
                float weight = (float)(right - k) / right_denom;
                energy += weight * mag_buf[k];
            }

            // Log dynamic range compression
            mel_matrix[f * STAGE2_NUM_MELS + m] = logf(energy + 1e-6f);
        }
    }

    // Global Mean & Standard Deviation Normalization across [39 x 16] matrix
    float sum = 0.0f;
    int total_elements = STAGE2_NUM_FRAMES * STAGE2_NUM_MELS; // 624
    for (int i = 0; i < total_elements; i++) {
        sum += mel_matrix[i];
    }
    float mean = sum / (float)total_elements;

    float sum_sq = 0.0f;
    for (int i = 0; i < total_elements; i++) {
        float diff = mel_matrix[i] - mean;
        sum_sq += diff * diff;
    }
    float std_dev = sqrtf(sum_sq / (float)total_elements) + 1e-6f;

    // Quantize to INT8: normalized * 32.0, clamped to [-128, 127]
    for (int i = 0; i < total_elements; i++) {
        float normalized = (mel_matrix[i] - mean) / std_dev;
        int32_t q = (int32_t)roundf(normalized * 32.0f);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        out_spec[i] = (int8_t)q;
    }
}
