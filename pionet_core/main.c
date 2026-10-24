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
#include "pdm_reader.pio.h"
#include "stage2_infer.h"
#include "stage2_features.h"
#include "pdm_stage1.pio.h"

#ifndef PIONET_TELEMETRY
#define PIONET_TELEMETRY 1   // Set to 0 for bench power measurements
#endif

// Double buffers isolated to dedicated SRAM banks to prevent AHB bus contention
static uint32_t __scratch_x("pdm_buf0") pdm_buf0[BUF_WORDS];
static uint32_t __scratch_y("pdm_buf1") pdm_buf1[BUF_WORDS];
static uint32_t * const pdm_buf[2] = { pdm_buf0, pdm_buf1 };

// buf_state: 0 = free (DMA write), 1 = full (CPU pending), 2 = active (CPU processing)
static volatile uint8_t  buf_state[2] = {0, 0};
static volatile uint8_t  dma_filling  = 0;
static volatile bool     dma_stalled  = false;
static volatile uint32_t overrun_count = 0;

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

// DMA Interrupt Service Routine: continuously ping-pongs buffers
static void dma_isr(void) {
    dma_hw->ints0 = 1u << dma_chan;

    buf_state[dma_filling] = 1;

    uint8_t next = 1 - dma_filling;
    dma_filling = next;
    buf_state[next] = 0; // Ensure next buffer is ready so DMA streams continuously without stalling
    dma_channel_set_trans_count(dma_chan, BUF_WORDS, false);
    dma_channel_set_write_addr(dma_chan, pdm_buf[dma_filling], true);
}

#if PIONET_TELEMETRY
static uint32_t core1_stack[1024];

static void telemetry_core1(void) {
    uint32_t prev_words = 0, prev_ones = 0;
    static uint32_t last_density_permille = 500;
    while (1) {
        sleep_ms(1000);
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

        printf("[Telemetry] Density: 0.%03lu | S1: %lu (PC: %lu) | S2: %lu us (~%lu.%lu ms, pred %d)\n",
               (unsigned long)density_permille,
               (unsigned long)pio_triggers,
               (unsigned long)pio1_pc,
               (unsigned long)bench_latency_us,
               (unsigned long)lat_ms_int,
               (unsigned long)lat_ms_dec,
               (int)bench_pred);
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

    dma_channel_config cfg = dma_channel_get_default_config(dma_chan);
    channel_config_set_transfer_data_size(&cfg, DMA_SIZE_32);
    channel_config_set_read_increment(&cfg, false);
    channel_config_set_write_increment(&cfg, true);
    channel_config_set_dreq(&cfg, pio_get_dreq(pio, sm, false));

    dma_channel_configure(
        dma_chan,
        &cfg,
        pdm_buf[0],
        &pio->rxf[sm],
        BUF_WORDS,
        false
    );

    dma_channel_set_irq0_enabled(dma_chan, true);
    irq_set_exclusive_handler(DMA_IRQ_0, dma_isr);
    irq_set_enabled(DMA_IRQ_0, true);

    dma_channel_start(dma_chan);
    printf("[PIO-Net] DMA armed, streaming to double buffer...\n");

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
        printf("\n>>> [STAGE-1 WAKE] PIO1 Hardware Matched Filter Fired! Waking CPU!\n");

        // 2. Decimate 1.0 second of incoming PDM audio into a 16,000-sample int16_t PCM buffer using 64x boxcar popcount
        static int16_t pcm[STAGE2_TOTAL_SAMPLES];
        int samples_captured = 0;

        while (samples_captured < STAGE2_TOTAL_SAMPLES) {
            uint8_t proc = 0xFF;
            while (proc == 0xFF) {
                for (uint8_t i = 0; i < 2; i++) {
                    if (buf_state[i] == 1) { proc = i; break; }
                }
                if (proc == 0xFF) {
                    __wfi();
                }
            }
            buf_state[proc] = 2; // active processing
            uint32_t *buf = pdm_buf[proc];

            int to_take = 256;
            if (samples_captured + to_take > STAGE2_TOTAL_SAMPLES) {
                to_take = STAGE2_TOTAL_SAMPLES - samples_captured;
            }

            uint32_t buf_ones = 0;
            for (int s = 0; s < to_take; s++) {
                uint32_t w0 = buf[s * 2];
                uint32_t w1 = buf[s * 2 + 1];
                int32_t ones = __builtin_popcount(w0) + __builtin_popcount(w1);
                buf_ones += ones;
                // 64x boxcar popcount decimation: ones is in [0, 64], center is 32
                int32_t x = (ones - 32) << 10;
                // 1st-order IIR DC blocker filter
                int32_t y = x - dc_filter_x + ((dc_filter_y * 32600) >> 15);
                dc_filter_x = x;
                dc_filter_y = y;
                if (y > 32767)  y = 32767;
                if (y < -32768) y = -32768;
                pcm[samples_captured + s] = (int16_t)y;
            }
#if PIONET_DEV_DUMP_PDM
            for (int w = 0; w < to_take * 2; w++) {
                printf("%08lx\n", (unsigned long)buf[w]);
            }
#endif
            samples_captured += to_take;

            stat_words += BUF_WORDS;
            stat_ones  += buf_ones;
            buf_state[proc] = 0;

            uint32_t irq_state = save_and_disable_interrupts();
            if (dma_stalled) {
                dma_stalled = false;
                dma_filling = proc;
                dma_channel_set_trans_count(dma_chan, BUF_WORDS, false);
                dma_channel_set_write_addr(dma_chan, pdm_buf[proc], true);
            }
            restore_interrupts(irq_state);
        }

        // 3. Pass the PCM buffer to stage2_extract_spectrogram(pcm, spec) to generate the [39 x 16] Mel-spectrogram
        static float spectrogram[STAGE2_NUM_FRAMES * STAGE2_NUM_MELS];
        uint32_t t_feat = time_us_32();
        stage2_extract_spectrogram(pcm, spectrogram);
        uint32_t feat_lat = time_us_32() - t_feat;

        // 4. Quantize the float spectrogram to int8_t and pass it to stage2_classify()
        static int8_t spectrogram_int8[STAGE2_NUM_FRAMES * STAGE2_NUM_MELS];
        for (int i = 0; i < STAGE2_NUM_FRAMES * STAGE2_NUM_MELS; i++) {
            int32_t val = (int32_t)roundf(spectrogram[i] * 32.0f);
            if (val > 127) val = 127;
            if (val < -128) val = -128;
            spectrogram_int8[i] = (int8_t)val;
        }

        gpio_put(22, 1);
        uint32_t t_inf = time_us_32();
        int8_t pred = stage2_classify(spectrogram_int8);
        float score = stage2_get_last_score();
        int8_t second_pred = stage2_get_second_class();
        float second_score = stage2_get_second_score();
        uint32_t inf_lat = time_us_32() - t_inf;
        gpio_put(22, 0);

        bench_pred = pred;
        bench_latency_us = inf_lat;
        stat_triggers++;

        // 5. Print the predicted class and return Core 0 to sleep
        printf(">>> [STAGE-2 INFER] Predicted Class: '%s' (class %d, score: %.2f) | Runner-up: '%s' (score: %.2f) | Feat: %lu us, Infer: %lu us\n",
               CLASS_NAMES[pred], pred, (double)score, CLASS_NAMES[second_pred], (double)second_score,
               (unsigned long)feat_lat, (unsigned long)inf_lat);

        float margin = score - second_score;
        bool confirmed = (pred == 0 && score >= 1.20f && margin >= 0.40f);

        if (confirmed) {
            printf("***************************************************\n");
            printf("   🎉 [WAKEWORD DETECTED]: 'YES'! ACTIVATING CPU!   \n");
            printf("   (Score: %.2f, Runner-up '%s': %.2f, Margin: +%.2f)\n",
                   (double)score, CLASS_NAMES[second_pred], (double)second_score, (double)margin);
            printf("***************************************************\n");
            sleep_ms(1000); // Hold status LED on for 1.0 second on confirmed wakeword
        } else {
            if (pred == 0) {
                printf("    [REJECTED]: 'yes' candidate rejected (score: %.2f < 1.20 or margin: +%.2f < 0.40). Returning to sleep.\n",
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