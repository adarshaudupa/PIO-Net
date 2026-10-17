// ==============================================================================
// PIO-Net: RP2040 Programmable I/O (SM-A) PDM Audio Capture & Stage-1 Matched Filter
// Authors: Adarsh & Anukul (Embedded Systems) & Gigi Koneti (ML / Acoustic Lead)
//
// Microarchitectural Overview:
// 1. PIO SM0 runs pdm_reader.pio at 1.024 MHz, side-setting PDM CLK on GPIO 10 and
//    sampling PDM DATA on GPIO 11 on the rising clock edge into the Input Shift Register (ISR).
// 2. Autopush pushes 32-bit words into the 4-word RX FIFO every 31.25 microseconds.
// 3. The Cortex-M0+ core drains words from the FIFO and evaluates the Stage-1 popcount
//    matched filter against PIONET_STAGE1_PATTERN (0xFFFFFFFF) in real time.
// 4. If the Hamming similarity exceeds PIONET_STAGE1_THRESHOLD (>=22/32 bits), an acoustic
//    event is flagged and the onboard LED is illuminated.
// 5. Diagnostics are rate-limited to avoid serial transmission blocking, preventing
//    RX FIFO overflow and loss of phase continuity.
// ==============================================================================

#include "main.h"

#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/pio.h"
#include "hardware/gpio.h"
#include "pdm_reader.pio.h"

int main(void)
{
    // Initialize standard I/O (USB CDC and UART if enabled)
    stdio_init_all();

    // Initialize visual indicator LED
    gpio_init(PIONET_LED_PIN);
    gpio_set_dir(PIONET_LED_PIN, GPIO_OUT);
    gpio_put(PIONET_LED_PIN, 0);

    // Flash LED on startup for visual confirmation
    for (int i = 0; i < 3; i++) {
        gpio_put(PIONET_LED_PIN, 1);
        sleep_ms(60);
        gpio_put(PIONET_LED_PIN, 0);
        sleep_ms(60);
    }

    printf("\n=======================================================\n");
    printf("   PIO-Net: Embedded Acoustic Wake-Up Core (SM-A)     \n");
    printf("   SenSys 2027 Research Project                        \n");
    printf("   Authors: Adarsh, Anukul & Gigi                      \n");
    printf("=======================================================\n");
    printf("Configuration:\n");
    printf(" - PDM CLK Pin:  GPIO %d (Target: 1.024 MHz)\n", PIONET_PDM_CLK_PIN);
    printf(" - PDM DATA Pin: GPIO %d\n", PIONET_PDM_DATA_PIN);
    printf(" - Filter Mask:  0x%08X (Threshold: %d/32 bits)\n", 
           PIONET_STAGE1_PATTERN, PIONET_STAGE1_THRESHOLD);
    printf("=======================================================\n\n");

    // Load PIO assembly program into PIO0 instruction memory
    PIO pio = pio0;
    uint sm = 0;
    uint offset = pio_add_program(pio, &pdm_reader_program);

    // Initialize state machine with pins and clock divider
    pdm_reader_program_init(pio, sm, offset, PIONET_PDM_DATA_PIN, PIONET_PDM_CLK_PIN);
    printf("[PIO-Net] State Machine 0 initialized and capturing...\n");

    uint32_t total_words = 0;
    uint32_t total_triggers = 0;
    uint32_t ones_count = 0;
    uint32_t loop_count = 0;
    bool led_state = false;
    uint32_t led_hold_counter = 0;

    while (1) {
        // Read 32-bit word directly from PIO RX FIFO (blocking wait)
        uint32_t pdm_word = pio_sm_get_blocking(pio, sm);
        total_words++;
        ones_count += __builtin_popcount(pdm_word);

        // Stage-1 Matched Filter: Popcount similarity test
        uint32_t match = pionet_compute_match(pdm_word, PIONET_STAGE1_PATTERN);
        if (match >= PIONET_STAGE1_THRESHOLD) {
            total_triggers++;
            led_state = true;
            led_hold_counter = 1000; // Hold LED on for ~31 ms (1000 words * 31.25 us)
            gpio_put(PIONET_LED_PIN, 1);
        } else if (led_state) {
            if (led_hold_counter > 0) {
                led_hold_counter--;
            } else {
                led_state = false;
                gpio_put(PIONET_LED_PIN, 0);
            }
        }

        // Periodic telemetry reporting: Every 32,000 words (~1.0 second of audio)
        // Rate-limiting prevents serial FIFO backpressure from overflowing the PIO RX FIFO
        if (++loop_count >= 32000) {
            float bit_density = (float)ones_count / (float)(32000 * 32);
            printf("[Telemetry] Words: %lu | Density: %.3f | Stage-1 Triggers: %lu | Last Word: 0x%08lX\n",
                   (unsigned long)total_words,
                   bit_density,
                   (unsigned long)total_triggers,
                   (unsigned long)pdm_word);

            // Reset periodic counters
            loop_count = 0;
            ones_count = 0;
        }
    }

    return 0;
}
