// ==============================================================================
// PIO-Net: Acoustic Wake-Up via RP2040 PIO State Machine (SM-A)
// Authors: Adarsh & Anukul (Embedded Team) & Gigi Koneti (ML / Acoustic Lead)
// Project: SenSys 2027
// ==============================================================================

#ifndef PIONET_CORE_MAIN_H
#define PIONET_CORE_MAIN_H

#include <stdint.h>
#include <stdbool.h>
#include "pico/stdlib.h"
#include "hardware/dma.h"

#ifndef __scratch_x
#define __scratch_x(foo)
#endif
#ifndef __scratch_y
#define __scratch_y(foo)
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ------------------------------------------------------------------------------
// Hardware Pin Configuration
// ------------------------------------------------------------------------------
#define PIONET_PDM_DATA_PIN         11      // GPIO 11: PDM Microphone Data Out
#define PIONET_PDM_CLK_PIN          10      // GPIO 10: PDM Microphone Bit Clock Out
#ifndef PICO_DEFAULT_LED_PIN
#define PIONET_LED_PIN              25      // Default onboard LED on Raspberry Pi Pico
#else
#define PIONET_LED_PIN              PICO_DEFAULT_LED_PIN
#endif

// ------------------------------------------------------------------------------
// PDM Sampling & Decimation Specifications
// ------------------------------------------------------------------------------
#define PIONET_PDM_CLOCK_HZ         1024000 // 1.024 MHz PDM clock
#define PIONET_AUDIO_RATE_HZ        16000   // 16.0 kHz Target Audio sample rate
#define PIONET_OSR                  64      // Over-sampling ratio: 1.024 MHz / 16 kHz = 64

// Buffer sizing: 512 words * 32 bits = 16,384 PDM bits = 16.0 ms of audio
#define BUF_WORDS                   512

// ------------------------------------------------------------------------------
// PIO-Net Stage-1 Hardware Matched Filter Parameters
// Co-designed with Gigi's Bit-Stream Acoustic Simulation & NAS
// ------------------------------------------------------------------------------
#define PIONET_STAGE1_PATTERN       0xFFFFFFFFu  // Acoustic high-pressure bitstream signature
#define PIONET_STAGE1_THRESHOLD     21           // Minimum matching bits out of 32 (Hamming dist <= 11)
#define PIONET_LED_HOLD_WORDS       1000         // ~31 ms hold for human eye; set to 0 for oscilloscope probing
#define PIONET_STAGE1_ENERGY_THRESHOLD  58       // Base onset energy threshold; adapts dynamically with noise floor (min 58)
#define PIONET_DEV_DUMP_PDM         0            // Set to 1 in dev build to dump raw PDM words over USB on wake

/**
 * @brief Computes bit match count (popcount of bitwise XNOR) between word and pattern.
 * Uses ARM Cortex-M0+ compiler intrinsic __builtin_popcount for single-cycle efficiency.
 */
static inline uint32_t pionet_compute_match(uint32_t word, uint32_t pattern) {
    return (uint32_t)__builtin_popcount(~(word ^ pattern));
}

#ifdef __cplusplus
}
#endif

#endif // PIONET_CORE_MAIN_H
