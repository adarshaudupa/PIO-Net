// ==============================================================================
// PIO-Net Comparison Suite - Baseline B: Duty-Cycled Buffer-Wake (The Fair Competitor)
// Architecture: DMA Background Ingestion + CPU Periodic 10ms Wake VAD
// Target: ACM SenSys 2027 Baseline Comparison
//
// Operation:
// 1. Clock runs at 125 MHz.
// 2. PIO0 SM0 generates 1.024 MHz PDM clock and shifts in 32-bit data.
// 3. DMA streams PDM bitstream into 32KB circular SRAM buffer in background.
// 4. Core 0 sleeps in __wfi() during silence/noise.
// 5. Hardware timer alarm wakes Core 0 every 10 ms (100 Hz polling rate).
// 6. On each 10 ms wake, Core 0 decimates 160 samples, calculates mean |PCM| energy:
//    - If below threshold (silence/noise): Core 0 returns to __wfi() immediately (~35 µs active).
//    - If above threshold (speech onset): Core 0 captures 1.0s window, runs Stage 2 DSCNN,
//      and returns to 10 ms duty-cycled sleep.
// 7. Full headless mode (no USB CDC, no UART, status LED disabled).
// ==============================================================================

#include <stdio.h>
#include <math.h>
#include "pico/stdlib.h"
#include "main.h"
#include "hardware/pio.h"
#include "hardware/gpio.h"
#include "hardware/dma.h"
#include "hardware/sync.h"
#include "hardware/timer.h"
#include "hardware/irq.h"
#include "hardware/clocks.h"
#include "hardware/pll.h"
#include "hardware/resets.h"
#include "pdm_reader.pio.h"
#include "stage2_infer.h"
#include "stage2_features.h"

#define RING_BYTES          32768
#define RING_WORDS          (RING_BYTES / sizeof(uint32_t)) // 8192 words
#define RING_MASK           (RING_WORDS - 1)
#define PREROLL_WORDS       4800                            // 150 ms at 32000 words/s

static uint32_t __aligned(RING_BYTES) pdm_ring[RING_WORDS];

static int dma_chan;
static int timer_alarm_num = -1;
static volatile bool timer_fired = false;

static int32_t dc_filter_x = 0;
static int32_t dc_filter_y = 0;

static const char *CLASS_NAMES[12] = {
    "yes", "no", "stop", "go", "left", "right",
    "on", "off", "up", "down", "_silence_", "_unknown_"
};

static void timer_alarm_cb(uint alarm_num) {
    (void)alarm_num;
    timer_fired = true;
}

static void dma_init_ring(void) {
    dma_channel_config cfg = dma_channel_get_default_config(dma_chan);
    channel_config_set_transfer_data_size(&cfg, DMA_SIZE_32);
    channel_config_set_read_increment(&cfg, false);
    channel_config_set_write_increment(&cfg, true);
    channel_config_set_ring(&cfg, true, 15); // 32KB wrap
    channel_config_set_dreq(&cfg, pio_get_dreq(pio0, 0, false));

    dma_channel_configure(
        dma_chan,
        &cfg,
        pdm_ring,
        &pio0->rxf[0],
        0xFFFFFFFFu,
        true
    );
}

static void run_stage2_inference(const int16_t *pcm) {
    static float spectrogram[STAGE2_NUM_FRAMES * STAGE2_NUM_MELS];
    stage2_extract_spectrogram(pcm, spectrogram);

    float std_dev = stage2_features_get_last_std_dev();
    if (std_dev >= 0.20f) {
        stage2_classify_float(spectrogram);
    }
}

int main(void) {
    stdio_init_all();

    // -------------------------------------------------------------------------
    // Peripheral Clock Gating & Deep-Power Optimization (Identical across all builds)
    // -------------------------------------------------------------------------
    clock_stop(clk_usb);
    pll_deinit(pll_usb);
    clock_stop(clk_adc);

    reset_block(RESETS_RESET_SPI0_BITS | RESETS_RESET_SPI1_BITS |
                RESETS_RESET_I2C0_BITS | RESETS_RESET_I2C1_BITS |
                RESETS_RESET_PWM_BITS  |
                RESETS_RESET_UART0_BITS | RESETS_RESET_UART1_BITS |
                RESETS_RESET_USBCTRL_BITS);

    gpio_init(PIONET_LED_PIN);
    gpio_set_dir(PIONET_LED_PIN, GPIO_OUT);
    gpio_put(PIONET_LED_PIN, 0);

    // Initialize State Machine 0 for PDM clock generation at 125 MHz
    // Divider = 125 MHz / 2.048 MHz = 61.03515625 = 61 + 9/256
    PIO pio = pio0;
    uint sm = 0;
    uint offset = pio_add_program(pio, &pdm_reader_program);
    pdm_reader_program_init(pio, sm, offset, PIONET_PDM_DATA_PIN, PIONET_PDM_CLK_PIN);
    pio_sm_set_clkdiv_int_frac(pio, sm, 61, 9);

    dma_chan = dma_claim_unused_channel(true);
    dma_init_ring();

    // Claim dedicated hardware timer alarm for 10 ms periodic wakeups
    timer_alarm_num = hardware_alarm_claim_unused(true);
    hardware_alarm_set_callback(timer_alarm_num, timer_alarm_cb);
    irq_set_enabled(TIMER_IRQ_0 + timer_alarm_num, true);

    stage2_features_init();

    // Warm-up inference
    static int8_t dummy_input[STAGE2_NUM_FRAMES * STAGE2_NUM_MELS] = {0};
    stage2_classify(dummy_input);

    uint32_t read_idx = 0;
    static int16_t block_pcm[160]; // 10 ms block
    static int16_t capture_pcm[STAGE2_TOTAL_SAMPLES]; // 1.0s window

    // Arm initial 10 ms alarm
    hardware_alarm_set_target(timer_alarm_num, delayed_by_us(get_absolute_time(), 10000));

    while (1) {
        // Sleep in WFI until 10 ms timer alarm wakes Core 0
        while (!timer_fired) {
            __wfi();
        }
        timer_fired = false;

        uint32_t write_addr = dma_hw->ch[dma_chan].write_addr;
        uint32_t cur_w = ((write_addr - (uintptr_t)pdm_ring) >> 2) & RING_MASK;
        uint32_t avail = (cur_w + RING_WORDS - read_idx) & RING_MASK;

        // Process newly arrived 10 ms block (160 samples = 320 PDM words)
        if (avail >= 320) {
            uint32_t block_energy = 0;
            for (int i = 0; i < 160; i++) {
                uint32_t w0 = pdm_ring[read_idx];
                uint32_t w1 = pdm_ring[(read_idx + 1) & RING_MASK];
                read_idx = (read_idx + 2) & RING_MASK;

                int32_t ones = __builtin_popcount(w0) + __builtin_popcount(w1);
                int32_t x = (ones - 32) << 10;
                int32_t y = x - dc_filter_x + ((dc_filter_y * 32600) >> 15);
                dc_filter_x = x;
                dc_filter_y = y;
                if (y > 32767)  y = 32767;
                if (y < -32768) y = -32768;
                block_pcm[i] = (int16_t)y;

                block_energy += (y < 0) ? (uint32_t)(-y) : (uint32_t)y;
            }
            uint32_t mean_energy = block_energy / 160;

            // Software VAD Energy Threshold Check
            if (mean_energy > 1100) {
                // Speech onset detected!
                uint32_t trig_w = cur_w;
                uint32_t cap_read_idx = ((trig_w + RING_WORDS - PREROLL_WORDS) & RING_MASK) & ~1u;
                int cap_count = 0;

                while (cap_count < STAGE2_TOTAL_SAMPLES) {
                    uint32_t cur_write = dma_hw->ch[dma_chan].write_addr;
                    uint32_t cur_pos = ((cur_write - (uintptr_t)pdm_ring) >> 2) & RING_MASK;
                    uint32_t cap_avail = (cur_pos + RING_WORDS - cap_read_idx) & RING_MASK;

                    while (cap_avail >= 2 && cap_count < STAGE2_TOTAL_SAMPLES) {
                        uint32_t w0 = pdm_ring[cap_read_idx];
                        uint32_t w1 = pdm_ring[(cap_read_idx + 1) & RING_MASK];
                        cap_read_idx = (cap_read_idx + 2) & RING_MASK;
                        cap_avail -= 2;

                        int32_t ones = __builtin_popcount(w0) + __builtin_popcount(w1);
                        int32_t x = (ones - 32) << 10;
                        int32_t y = x - dc_filter_x + ((dc_filter_y * 32600) >> 15);
                        dc_filter_x = x;
                        dc_filter_y = y;
                        if (y > 32767)  y = 32767;
                        if (y < -32768) y = -32768;
                        capture_pcm[cap_count++] = (int16_t)y;
                    }

                    if (cap_count < STAGE2_TOTAL_SAMPLES) {
                        // Sleep briefly between batch drain intervals during capture
                        hardware_alarm_set_target(timer_alarm_num, delayed_by_us(get_absolute_time(), 8000));
                        while (!timer_fired) {
                            __wfi();
                        }
                        timer_fired = false;
                    }
                }

                // Run Stage 2 DSCNN inference
                run_stage2_inference(capture_pcm);

                // Advance read pointer to avoid immediate re-trigger on same utterance
                read_idx = ((dma_hw->ch[dma_chan].write_addr - (uintptr_t)pdm_ring) >> 2) & RING_MASK;
                dc_filter_x = 0;
                dc_filter_y = 0;
            }
        }

        // Re-arm hardware timer for next 10 ms wakeup
        hardware_alarm_set_target(timer_alarm_num, delayed_by_us(get_absolute_time(), 10000));
    }

    return 0;
}
