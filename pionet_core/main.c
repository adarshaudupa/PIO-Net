// ==============================================================================
// PIO-Net: RP2040 Programmable I/O (SM-A) PDM Audio Capture & Stage-1 Matched Filter
// Authors: Adarsh & Anukul (Embedded Systems) & Gigi Koneti (ML / Acoustic Lead)
//
// Microarchitectural Overview:
// 1. PIO SM0 runs pdm_reader.pio at 1.024 MHz, side-setting PDM CLK on GPIO 10 and
//    sampling PDM DATA on GPIO 11 on rising edge into the Input Shift Register (ISR).
// 2. Autopush pushes 32-bit words into the 4-word RX FIFO every 31.25 microseconds.
// 3. DMA transfers words from PIO RX FIFO to RAM automatically without CPU polling.
// 4. Double-buffering: DMA fills one buffer while Core 0 executes Stage-1 sub-block
//    boxcar energy integration on the completed buffer.
// 5. DMA ISR hands completed buffers to Core 0. Main loop restarts DMA only after a stall.
// 6. Core 1 handles USB CDC telemetry to prevent blocking the Core 0 audio pipeline.
// ==============================================================================

#include "main.h"

#include <stdio.h>
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "hardware/pio.h"
#include "hardware/gpio.h"
#include "hardware/dma.h"
#include "hardware/irq.h"
#include "hardware/sync.h"
#include "pdm_reader.pio.h"
#include "stage2_infer.h"
#include "pdm_stage1.pio.h"

#ifndef PIONET_TELEMETRY
#define PIONET_TELEMETRY 1   // Set to 0 for bench power measurements
#endif

#define PIONET_STAGE1_ENERGY_THRESHOLD 90

// Double buffers isolated to dedicated SRAM banks to prevent AHB bus contention
static uint32_t __scratch_x("pdm_buf0") pdm_buf0[BUF_WORDS];
static uint32_t __scratch_y("pdm_buf1") pdm_buf1[BUF_WORDS];
static uint32_t * const pdm_buf[2] = { pdm_buf0, pdm_buf1 };

// buf_state: 0 = free (DMA write), 1 = full (CPU pending), 2 = active (CPU processing)
static volatile uint8_t  buf_state[2] = {0, 0};
static volatile uint8_t  dma_filling  = 0;
static volatile bool     dma_stalled  = false;
static volatile uint32_t overrun_count = 0;
static uint32_t led_hold = 0;

// Inter-core telemetry statistics
static volatile uint32_t stat_words       = 0;
static volatile uint32_t stat_ones        = 0;
static volatile uint32_t stat_triggers    = 0;
static volatile uint32_t stat_peak_energy = 0;
static volatile uint32_t pio_triggers     = 0;

static volatile uint32_t bench_latency_us = 0;
static volatile int8_t   bench_pred = 0;

static int dma_chan;

static void dma_isr(void) {
    dma_hw->ints0 = 1u << dma_chan;

    buf_state[dma_filling] = 1;

    uint8_t next = 1 - dma_filling;
    if (buf_state[next] == 0) {
        dma_filling = next;
        dma_channel_set_trans_count(dma_chan, BUF_WORDS, false);
        dma_channel_set_write_addr(dma_chan, pdm_buf[dma_filling], true);
    } else {
        overrun_count++;
        dma_stalled = true;
    }
}

#if PIONET_TELEMETRY
static uint32_t core1_stack[1024];

static void telemetry_core1(void) {
    uint32_t prev_words = 0, prev_ones = 0;
    while (1) {
        sleep_ms(1000);
        uint32_t words   = stat_words;
        uint32_t ones    = stat_ones;
        uint32_t d_words = words - prev_words;
        uint32_t d_ones  = ones - prev_ones;
        prev_words = words;
        prev_ones  = ones;

        // Integer permille: 501 = 0.501 (eliminates 64-bit float stack corruption)
        uint32_t density_permille = d_words ? (uint32_t)(((uint64_t)d_ones * 1000ULL) / ((uint64_t)d_words * 32ULL)) : 0;
        uint32_t peak = stat_peak_energy;
        stat_peak_energy = 0;

        uint32_t lat_ms_int = bench_latency_us / 1000;
        uint32_t lat_ms_dec = (bench_latency_us % 1000) / 100;

        uint32_t pio1_pc = pio1->sm[0].addr;
        printf("[Telemetry] Density: 0.%03lu | Peak: %lu | S1: %lu (PC: %lu) | S2: %lu us (~%lu.%lu ms, pred %d) | Overruns: %lu\n",
               (unsigned long)density_permille,
               (unsigned long)peak,
               (unsigned long)pio_triggers,
               (unsigned long)pio1_pc,
               (unsigned long)bench_latency_us,
               (unsigned long)lat_ms_int,
               (unsigned long)lat_ms_dec,
               (int)bench_pred,
               (unsigned long)overrun_count);
    }
}
#endif

static volatile bool     pio1_triggered = false;

static void pio1_isr(void) {
    pio1_hw->irq = 1u;
    pio_triggers++;
    pio1_triggered = true;
}

int main(void)
{
    stdio_init_all();

    gpio_init(PIONET_LED_PIN);
    gpio_set_dir(PIONET_LED_PIN, GPIO_OUT);
    gpio_put(PIONET_LED_PIN, 0);

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
    printf(" - Energy Threshold:  %d (16-word boxcar deviation)\n", PIONET_STAGE1_ENERGY_THRESHOLD);
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
    printf("[PIO-Net] DMA armed, capturing to double buffer...\n");
    // Dry-run Stage-2 inference benchmark
    static int8_t dummy_input[39 * 16] = {0};
    uint32_t t0 = time_us_32();
    bench_pred = stage2_classify(dummy_input);
    bench_latency_us = time_us_32() - t0;;

    gpio_init(22);
    gpio_set_dir(22, GPIO_OUT);
    gpio_put(22, 0);

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

    // -------------------------------------------------------------------------
    // Adaptive DC baseline & noise floor tracking
    // Mirrors Gigi's EnergyZCRVAD noise floor IIR from vad_reference.py.
    // PDM sigma-delta quantization noise variance changes 30x depending on
    // the mic's DC bias (which drifts between power cycles), so no fixed
    // -------------------------------------------------------------------------
    // Adaptive DC baseline & noise floor tracking
    // -------------------------------------------------------------------------
    static int32_t  dc_center_q8    = (265 << 8); // Sub-bit DC baseline in Q8 format
    static int32_t  dc_center       = 265;        // DC ones-per-sub-block
    static uint32_t noise_floor     = 0;          // Adaptive silence energy baseline
    static uint32_t warmup_accum    = 0;          // Accumulator for warmup energy
    static uint32_t warmup_ones     = 0;          // Accumulator for warmup ones
    static uint8_t  warmup          = 30;         // ~480 ms calibration window at boot
    static uint8_t  cooldown        = 0;          // Post-trigger debounce

    while (1) {
        uint8_t proc = 0xFF;
        for (uint8_t i = 0; i < 2; i++) {
            if (buf_state[i] == 1) { proc = i; break; }
        }

        if (proc == 0xFF) {
            uint32_t s = save_and_disable_interrupts();
            if (buf_state[0] != 1 && buf_state[1] != 1) {
                __wfi();
            }
            restore_interrupts(s);
            continue;
        }

        buf_state[proc] = 2;

        uint32_t *buf = pdm_buf[proc];
        uint32_t buf_ones = 0, buf_energy = 0, buf_triggers = 0;

        // 32 sub-blocks of 16 words (512 bits = 0.5 ms per sub-block)
        for (uint32_t b = 0; b < 32; b++) {
            uint32_t block_ones = 0;
            for (uint32_t j = 0; j < 16; j++) {
                uint32_t word = buf[b * 16 + j];
                uint32_t ones = __builtin_popcount(word);
                buf_ones += ones;
                block_ones += ones;
            }
            int32_t dev = (int32_t)block_ones - dc_center;
            if (dev < 0) dev = -dev;
            buf_energy += (uint32_t)dev;
        }

        if (buf_energy > stat_peak_energy) {
            stat_peak_energy = buf_energy;
        }

        // ---- Calibration / Warmup ----
        if (warmup > 0) {
            warmup_accum += buf_energy;
            warmup_ones  += buf_ones;
            warmup--;
            if (warmup == 0) {
                noise_floor  = warmup_accum / 30;
                int32_t mean = (int32_t)(warmup_ones / (30 * 32));
                dc_center_q8 = mean << 8;
                dc_center    = mean;
            }
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
            continue;
        }

        // Compute adaptive threshold:
        // Silence is ~25-38. Speech is ~72-105. Minimum threshold of 58
        // completely eliminates false blinks in silence while catching normal speech.
        uint32_t margin = noise_floor / 2;
        if (margin < 25) margin = 25;
        uint32_t adaptive_threshold = noise_floor + margin;
        if (adaptive_threshold < 58) adaptive_threshold = 58;

        // Adapt DC baseline and silence noise floor ONLY during silence (non-trigger frames)
        if (buf_energy < adaptive_threshold) {
            // Track DC center gently (sub-bit precision with Q8 fixed point)
            int32_t current_mean_q8 = (int32_t)((buf_ones << 8) / 32);
            dc_center_q8 += (current_mean_q8 - dc_center_q8) >> 5;
            dc_center = (dc_center_q8 + 128) >> 8;

            // Symmetrical, steady noise floor tracker without downward leakage
            if (buf_energy > noise_floor) {
                uint32_t step = (buf_energy - noise_floor) >> 4;
                if (step < 1) step = 1;
                noise_floor += step;
            } else if (buf_energy < noise_floor) {
                uint32_t step = (noise_floor - buf_energy) >> 4;
                if (step < 1) step = 1;
                noise_floor -= step;
            }
        }

        // Buffer proc has been fully read. Release it back to DMA immediately
        // so DMA can ping-pong without stalling if Stage-2 inference runs (~28.7 ms).
        stat_words    += BUF_WORDS;
        stat_ones     += buf_ones;
        stat_triggers += buf_triggers;

        buf_state[proc] = 0;
        uint32_t irq_state = save_and_disable_interrupts();
        if (dma_stalled) {
            dma_stalled = false;
            dma_filling = proc;
            dma_channel_set_trans_count(dma_chan, BUF_WORDS, false);
            dma_channel_set_write_addr(dma_chan, pdm_buf[proc], true);
        }
        restore_interrupts(irq_state);

        // Cooldown management
        if (cooldown > 0) cooldown--;

        // Stage-1 Acoustic Energy Gate (adaptive threshold + PIO1 hardware trigger)
        if ((buf_energy > adaptive_threshold || pio1_triggered) && cooldown == 0) {
            pio1_triggered = false;
            printf(">>> [STAGE-1 WAKE] Speech Detected! Energy: %lu (floor: %lu, thresh: %lu)\n",
                   (unsigned long)buf_energy,
                   (unsigned long)noise_floor,
                   (unsigned long)adaptive_threshold);
            gpio_put(22, 1);
            bench_pred = stage2_classify((const int8_t *)dummy_input);
            gpio_put(22, 0);
            stat_triggers++;
            led_hold = 20;
            gpio_put(PIONET_LED_PIN, 1);
            cooldown = 4;  // Skip 4 frames (~64 ms) to debounce speech
        } else if (led_hold > 0) {
            led_hold--;
            if (led_hold == 0) {
                gpio_put(PIONET_LED_PIN, 0);
            }
        }
    }
}