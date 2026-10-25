// ==============================================================================
// PIO-Net: RP2040 Programmable I/O PDM Audio Capture & Stage-1 Matched Filter
// Authors: Adarsh & Anukul (Embedded Systems) & Gigi Koneti (ML / Acoustic Lead)
// Project: ACM SenSys 2027
//
// Microarchitectural Overview (SenSys 2027 Architecture):
// 1. Quiescent Mode: Core 0 sleeps in __wfi(). Zero software energy gating.
// 2. PIO1 Stage-1 matched filter monitors PDM stream directly from GPIO pins.
// 3. PIO1_IRQ_0 fires on acoustic keyword onset, setting pio1_triggered and waking Core 0.
// 4. Core 0 acknowledges trigger, turns on status LED, and captures 1.0 second of PDM audio.
// 5. 64x boxcar popcount decimation generates 16,000 int16_t PCM samples.
// 6. Stage-2 extracts [39 x 16] Mel-spectrogram, quantizes to int8_t, and classifies.
// 7. Core 0 prints the predicted class and immediately returns to __wfi() sleep.
// ==============================================================================

#include "main.h"

#include <stdio.h>
#include <math.h>
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "hardware/pio.h"
#include "hardware/gpio.h"
#include "hardware/dma.h"
#include "hardware/irq.h"
#include "hardware/sync.h"
#include "hardware/timer.h"
#include "pdm_reader.pio.h"
#include "stage2_infer.h"
#include "stage2_features.h"
#include "pdm_stage1.pio.h"

#ifndef PIONET_TELEMETRY
#define PIONET_TELEMETRY 1   // Set to 0 for bench power measurements
#endif

// STEP 1: Free-running circular ring buffer in main SRAM (32768 bytes, 8192 uint32_t words, 256 ms of history)
#define RING_BYTES          32768
#define RING_WORDS          (RING_BYTES / sizeof(uint32_t)) // 8192 words
#define RING_MASK           (RING_WORDS - 1)               // 8191 (0x1FFF)
#define PREROLL_WORDS       4800                           // 150 ms at 32000 words/s

static uint32_t __aligned(RING_BYTES) pdm_ring[RING_WORDS];

// 12-Class MLPerf Tiny keyword labels
static const char *CLASS_NAMES[12] = {
    "yes", "no", "stop", "go", "left", "right",
    "on", "off", "up", "down", "_silence_", "_unknown_"
};

static int32_t dc_filter_x = 0;
static int32_t dc_filter_y = 0;

// Inter-core telemetry statistics
static volatile uint32_t stat_words       = 0;
static volatile uint32_t stat_ones        = 0;
static volatile uint32_t stat_triggers    = 0;
static volatile uint32_t pio_triggers     = 0;

static volatile uint32_t bench_latency_us = 0;
static volatile int8_t   bench_pred = 0;

static int dma_chan;
static int timer_alarm_num = -1;

static void alarm_dummy_cb(uint alarm_num) {
    (void)alarm_num;
}

// Lag tracking (write pointer lapping read pointer)
static volatile uint32_t lag_count = 0;
static volatile uint32_t core1_logged_lag = 0;

// 20-bin energy envelope (50 ms bins, 800 samples each, integer mean |pcm|)
static volatile uint16_t env_bins[20] = {0};
static volatile uint32_t env_print_seq = 0;
static volatile uint32_t core1_printed_seq = 0;

// STEP 1: One free-running ring DMA, 32768 bytes aligned to 32768 in main SRAM, trans_count 0xFFFFFFFF, NO DMA IRQ.
// SM0 keeps draining it continuously.
static void dma_init_ring(void) {
    dma_channel_config cfg = dma_channel_get_default_config(dma_chan);
    channel_config_set_transfer_data_size(&cfg, DMA_SIZE_32);
    channel_config_set_read_increment(&cfg, false);
    channel_config_set_write_increment(&cfg, true);
    channel_config_set_ring(&cfg, true, 15); // 32768 bytes (8192 words) ring wrap
    channel_config_set_dreq(&cfg, pio_get_dreq(pio0, 0, false));

    dma_channel_configure(
        dma_chan,
        &cfg,
        pdm_ring,
        &pio0->rxf[0],
        0xFFFFFFFFu,
        true // start immediately so PIO0 never stalls and PDM CLK stays active
    );
}

#if PIONET_TELEMETRY
static uint32_t core1_stack[1024];

static void telemetry_core1(void) {
    uint32_t prev_words = 0, prev_ones = 0;
    static uint32_t last_density_permille = 500;
    uint32_t tick_count = 0;

    while (1) {
        sleep_ms(1);
        tick_count++;

        // Core 1 lag logger: prints lag counter if write pointer lapped read_idx
        uint32_t current_lag = lag_count;
        if (current_lag > core1_logged_lag) {
            core1_logged_lag = current_lag;
            printf("\n>>> [CORE 1 LAG EVENT] Lag counter: %lu | write pointer lapped read_idx\n\n",
                   (unsigned long)current_lag);
        }

        // Step 2: Core 1 prints the 20-bin envelope line right after STAGE-2 INFER
        if (env_print_seq != core1_printed_seq) {
            printf(">>> [ENV] %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u\n",
                   (unsigned int)env_bins[0],  (unsigned int)env_bins[1],  (unsigned int)env_bins[2],  (unsigned int)env_bins[3],
                   (unsigned int)env_bins[4],  (unsigned int)env_bins[5],  (unsigned int)env_bins[6],  (unsigned int)env_bins[7],
                   (unsigned int)env_bins[8],  (unsigned int)env_bins[9],  (unsigned int)env_bins[10], (unsigned int)env_bins[11],
                   (unsigned int)env_bins[12], (unsigned int)env_bins[13], (unsigned int)env_bins[14], (unsigned int)env_bins[15],
                   (unsigned int)env_bins[16], (unsigned int)env_bins[17], (unsigned int)env_bins[18], (unsigned int)env_bins[19]);
            core1_printed_seq = env_print_seq;
        }

        if (tick_count >= 1000) {
            tick_count = 0;
            uint32_t words   = stat_words;
            uint32_t ones    = stat_ones;
            uint32_t d_words = words - prev_words;
            uint32_t d_ones  = ones - prev_ones;
            prev_words = words;
            prev_ones  = ones;

            if (d_words > 0) {
                last_density_permille = (uint32_t)(((uint64_t)d_ones * 1000ULL) / ((uint64_t)d_words * 32ULL));
            }
            uint32_t density_permille = last_density_permille;
            uint32_t lat_ms_int = bench_latency_us / 1000;
            uint32_t lat_ms_dec = (bench_latency_us % 1000) / 100;
            uint32_t pio1_pc = pio1->sm[0].addr;

            printf("[Telemetry] Density: 0.%03lu | S1: %lu (PC: %lu) | S2: %lu us (~%lu.%lu ms, pred %d) | Overruns: %lu\n",
                   (unsigned long)density_permille,
                   (unsigned long)pio_triggers,
                   (unsigned long)pio1_pc,
                   (unsigned long)bench_latency_us,
                   (unsigned long)lat_ms_int,
                   (unsigned long)lat_ms_dec,
                   (int)bench_pred,
                   (unsigned long)lag_count);
        }
    }
}
#endif

// Hardware PIO1 Stage-1 Wake Condition flag
static volatile bool pio1_triggered = false;

static void pio1_isr(void) {
    pio1_hw->irq = 1u; // Clear PIO IRQ 0
    pio_triggers++;
    pio1_triggered = true;
}

int main(void)
{
    stdio_init_all();

    gpio_init(PIONET_LED_PIN);
    gpio_set_dir(PIONET_LED_PIN, GPIO_OUT);
    gpio_put(PIONET_LED_PIN, 0);

    // Boot blink sequence
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
    printf(" - PDM CLK Pin:       GPIO %d (Target: 1.024 MHz)\n", PIONET_PDM_CLK_PIN);
    printf(" - PDM DATA Pin:      GPIO %d\n", PIONET_PDM_DATA_PIN);
    printf(" - Stage-1 Gate:      Hardware PIO1 Matched Filter (sole wake source)\n");
    printf(" - Stage-2 Core:      DSCNNMicro [39 x 16] Cortex-M0+ Classifier\n");
    printf("=======================================================\n\n");

    // Initialize State Machine 0 for PDM clock generation and data capture
    PIO pio = pio0;
    uint sm = 0;
    uint offset = pio_add_program(pio, &pdm_reader_program);
    pdm_reader_program_init(pio, sm, offset, PIONET_PDM_DATA_PIN, PIONET_PDM_CLK_PIN);
    printf("[PIO-Net] State Machine 0 initialized and capturing...\n");

    dma_chan = dma_claim_unused_channel(true);

    // Arm DMA in silent free-running ring mode: keeps PIO0 draining and PDM CLK alive with 0 CPU IRQs
    dma_init_ring();
    printf("[PIO-Net] DMA armed in silent background ring mode (32KB ring, trans_count 0xFFFFFFFF, no IRQ)...\n");

    // Claim dedicated hardware timer alarm for post-wake 8 ms sleep intervals
    timer_alarm_num = hardware_alarm_claim_unused(true);
    hardware_alarm_set_callback(timer_alarm_num, alarm_dummy_cb);
    irq_set_enabled(TIMER_IRQ_0 + timer_alarm_num, false);

    // Initialize Stage-2 Mel-spectrogram feature extractor
    stage2_features_init();

    // Dry-run Stage-2 inference benchmark
    static int8_t dummy_input[STAGE2_NUM_FRAMES * STAGE2_NUM_MELS] = {0};
    uint32_t t0 = time_us_32();
    bench_pred = stage2_classify(dummy_input);
    bench_latency_us = time_us_32() - t0;

    gpio_init(22);
    gpio_set_dir(22, GPIO_OUT);
    gpio_put(22, 0);

    // Initialize State Machine on PIO1 for Stage-1 Hardware Matched Filter
    PIO pio1_inst = pio1;
    uint sm1 = 0;
    uint offset1 = pio_add_program(pio1_inst, &pdm_stage1_program);
    pdm_stage1_program_init(pio1_inst, sm1, offset1, PIONET_PDM_CLK_PIN, PIONET_PDM_DATA_PIN);

    pio_set_irq0_source_enabled(pio1_inst, pis_interrupt0, true);
    irq_set_exclusive_handler(PIO1_IRQ_0, pio1_isr);
    irq_set_enabled(PIO1_IRQ_0, true);

#if PIONET_TELEMETRY
    multicore_launch_core1_with_stack(telemetry_core1, core1_stack, sizeof(core1_stack));
#endif

    printf("[PIO-Net] Core 0 entering low-power __wfi() sleep. Waiting for PIO1 hardware wake...\n\n");

    while (1) {
        // Task 3: Make pio1_triggered the sole wake condition.
        // When pio1_triggered is false, keep Core 0 sleeping in __wfi().
        gpio_put(PIONET_LED_PIN, 0);
        while (!pio1_triggered) {
            __wfi();
        }

        // Task 4: When pio1_triggered fires:
        // 1. Acknowledge trigger and turn on the status LED
        pio1_triggered = false;
        irq_set_enabled(PIO1_IRQ_0, false); // Mask PIO1 interrupt while CPU is capturing & processing
        gpio_put(PIONET_LED_PIN, 1);

        // Reset DC filter states between captures to prevent transient leakage
        dc_filter_x = 0;
        dc_filter_y = 0;

        // STEP 2: on PIO1 wake, read dma_hw->ch[dma_chan].write_addr, convert to a word index,
        // and set read_idx = write_idx - PREROLL_WORDS (modulo ring). PREROLL_WORDS is 4800 (150 ms).
        uint32_t write_addr = dma_hw->ch[dma_chan].write_addr;
        uint32_t write_idx = ((write_addr - (uintptr_t)pdm_ring) >> 2) & RING_MASK;
        uint32_t read_idx = ((write_idx + RING_WORDS - PREROLL_WORDS) & RING_MASK) & ~1u;

        // STEP 3 (replaces polling): after wake, loop { consume all words between read_idx and the DMA write pointer;
        // if 16000 samples done, break; arm a hardware timer alarm 8 ms ahead; __wfi(); }
        // No spin loop, no DMA IRQ. Before wake, CPU is only in __wfi() with DMA_IRQ_0 disabled;
        // PIO1_IRQ_0 is sole wake source. Keep a lag counter if write pointer would lap read_idx.
        static int16_t pcm[STAGE2_TOTAL_SAMPLES];
        int samples_captured = 0;

        irq_set_enabled(TIMER_IRQ_0 + timer_alarm_num, true);

        while (1) {
            uint32_t cur_write_addr = dma_hw->ch[dma_chan].write_addr;
            uint32_t cur_w = ((cur_write_addr - (uintptr_t)pdm_ring) >> 2) & RING_MASK;
            uint32_t avail = (cur_w + RING_WORDS - read_idx) & RING_MASK;

            if (avail >= RING_WORDS - 64) {
                lag_count++;
                read_idx = (cur_w + RING_WORDS - PREROLL_WORDS) & RING_MASK;
                read_idx &= ~1u;
                avail = PREROLL_WORDS;
            }

            while (avail >= 2 && samples_captured < STAGE2_TOTAL_SAMPLES) {
                uint32_t w0 = pdm_ring[read_idx];
                uint32_t w1 = pdm_ring[read_idx + 1];
                read_idx = (read_idx + 2) & RING_MASK;
                avail -= 2;

                int32_t ones = __builtin_popcount(w0) + __builtin_popcount(w1);
                stat_words += 2;
                stat_ones  += (uint32_t)ones;

                // 64x boxcar popcount decimation: ones is in [0, 64], center is 32
                // (ones - 32) is [-32, 32]. Shifting by 10 maps exactly to 16-bit range [-32768, 32768]
                int32_t x = (ones - 32) << 10;
                // 1st-order IIR DC blocker filter
                int32_t y = x - dc_filter_x + ((dc_filter_y * 32600) >> 15);
                dc_filter_x = x;
                dc_filter_y = y;
                if (y > 32767)  y = 32767;
                if (y < -32768) y = -32768;
                pcm[samples_captured++] = (int16_t)y;
            }

            if (samples_captured >= STAGE2_TOTAL_SAMPLES) {
                break;
            }

            // Arm a hardware timer alarm 8 ms ahead; __wfi()
            if (!hardware_alarm_set_target(timer_alarm_num, delayed_by_us(get_absolute_time(), 8000))) {
                __wfi();
            }
        }

        irq_set_enabled(TIMER_IRQ_0 + timer_alarm_num, false);
        hardware_alarm_cancel(timer_alarm_num);

        // Step 1: Compute integer mean |pcm| for each of 20 consecutive 50 ms bins (800 samples each)
        for (int b = 0; b < 20; b++) {
            uint32_t sum = 0;
            const int16_t *bin_samples = &pcm[b * 800];
            for (int s = 0; s < 800; s++) {
                int16_t sample = bin_samples[s];
                sum += (sample < 0) ? (uint32_t)(-sample) : (uint32_t)sample;
            }
            env_bins[b] = (uint16_t)(sum / 800);
        }

        printf("\n>>> [STAGE-1 WAKE] PIO1 Hardware Matched Filter Fired! Waking CPU!\n");

        // 3. Pass the PCM buffer to stage2_extract_spectrogram(pcm, spec) to generate the [39 x 16] Mel-spectrogram
        static float spectrogram[STAGE2_NUM_FRAMES * STAGE2_NUM_MELS];
        uint32_t t_feat = time_us_32();
        stage2_extract_spectrogram(pcm, spectrogram);
        uint32_t feat_lat = time_us_32() - t_feat;

        // 4. Pass high-precision spectrogram directly to Stage-2 classifier (scale 256.0f)
        int8_t pred = 10;
        float score = 0.0f;
        int8_t second_pred = 11;
        float second_score = 0.0f;
        uint32_t inf_lat = 0;

        float std_dev = stage2_features_get_last_std_dev();
        if (std_dev < 0.20f) {
            pred = 10; // _silence_
            score = 0.0f;
            second_pred = 11; // _unknown_
            second_score = 0.0f;
        } else {
            gpio_put(22, 1);
            uint32_t t_inf = time_us_32();
            pred = stage2_classify_float(spectrogram);
            score = stage2_get_last_score();
            second_pred = stage2_get_second_class();
            second_score = stage2_get_second_score();
            inf_lat = time_us_32() - t_inf;
            gpio_put(22, 0);
        }

        bench_pred = pred;
        bench_latency_us = inf_lat;
        stat_triggers++;

        // 5. Print the predicted class and return Core 0 to sleep
        printf(">>> [STAGE-2 INFER] Predicted Class: '%s' (class %d, score: %.2f) | Runner-up: '%s' (score: %.2f) | Feat: %lu us, Infer: %lu us\n",
               CLASS_NAMES[pred], pred, (double)score, CLASS_NAMES[second_pred], (double)second_score,
               (unsigned long)feat_lat, (unsigned long)inf_lat);

        // Step 2: Signal Core 1 to print the 20-bin envelope line right after STAGE-2 INFER
        env_print_seq++;
        while (core1_printed_seq != env_print_seq) {
            tight_loop_contents();
        }

        float margin = score - second_score;
        bool confirmed = (pred == 0 && score >= 0.80f && margin >= 0.30f);

        if (confirmed) {
            printf("***************************************************\n");
            printf("   🎉 [WAKEWORD DETECTED]: 'YES'! ACTIVATING CPU!   \n");
            printf("   (Score: %.2f, Runner-up '%s': %.2f, Margin: +%.2f)\n",
                   (double)score, CLASS_NAMES[second_pred], (double)second_score, (double)margin);
            printf("***************************************************\n");
            sleep_ms(1000); // Hold status LED on for 1.0 second on confirmed wakeword
        } else {
            if (pred == 0) {
                printf("    [REJECTED]: 'yes' candidate rejected (score: %.2f < 0.80 or margin: +%.2f < 0.30). Returning to sleep.\n",
                       (double)score, (double)margin);
            } else {
                printf("    [REJECTED]: Predicted '%s' (score: %.2f, margin: +%.2f) is not wakeword 'yes'. Returning to sleep.\n",
                       CLASS_NAMES[pred], (double)score, (double)margin);
            }
            sleep_ms(200);
        }

        gpio_put(PIONET_LED_PIN, 0);
        pio1_hw->irq = 1u; // Clear any pending hardware flag during processing
        pio1_triggered = false;
        irq_set_enabled(PIO1_IRQ_0, true); // Re-arm hardware interrupt for sleep
    }
}