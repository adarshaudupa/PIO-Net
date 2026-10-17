// ==============================================================================
// PIO-Net: RP2040 Programmable I/O (SM-A) PDM Audio Capture & Stage-1 Matched Filter
// Authors: Adarsh & Anukul (Embedded Systems) & Gigi Koneti (ML / Acoustic Lead)
//
// Microarchitectural Overview:
// 1. PIO SM0 runs pdm_reader.pio at 1.024 MHz, side-setting PDM CLK on GPIO 10 and
//    sampling PDM DATA on GPIO 11 on the rising clock edge into the Input Shift Register (ISR).
// 2. Autopush pushes 32-bit words into the 4-word RX FIFO every 31.25 microseconds.
// 3. DMA channel transfers words from PIO RX FIFO to RAM automatically.
//    CPU is not involved in data movement — zero polling overhead.
// 4. Double-buffering: DMA fills one buffer while CPU runs Stage-1 matched filter
//    on the completed buffer. No gap in PDM capture between buffers.
// 5. DMA completion interrupt triggers matched filter evaluation on the filled buffer.
// ==============================================================================

#include "main.h"

#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/pio.h"
#include "hardware/gpio.h"
#include "hardware/dma.h"
#include "hardware/irq.h"
#include "pdm_reader.pio.h"

// Double buffers: DMA ping-pongs between these
static uint32_t pdm_buf[2][BUF_WORDS];
static volatile uint8_t dma_filling = 0; // Buffer DMA is currently writing to
static volatile uint8_t  buf_ready    = 0;  // which buffer CPU should process
static volatile bool     buf_flag     = false; // set by ISR when buffer is full
uint32_t led_hold = 0;

static int dma_chan;

static void dma_isr(void) {
    // Clear the interrupt
    dma_hw->ints0 = 1u << dma_chan;

    // Mark which buffer is ready for CPU
    buf_ready   = dma_filling;
    dma_filling = 1 - dma_filling;   // Flip DMA to the other buffer
    buf_flag  = true;

    // Rearm DMA to fill the other buffer
    dma_channel_set_write_addr(dma_chan, pdm_buf[1 - buf_ready], true);
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
    printf(" - PDM CLK Pin:  GPIO %d (Target: 1.024 MHz)\n", PIONET_PDM_CLK_PIN);
    printf(" - PDM DATA Pin: GPIO %d\n", PIONET_PDM_DATA_PIN);
    printf(" - Filter Mask:  0x%08X (Threshold: %d/32 bits)\n",
           PIONET_STAGE1_PATTERN, PIONET_STAGE1_THRESHOLD);
    printf("=======================================================\n\n");

    // Init PIO
    PIO pio = pio0;
    uint sm = 0;
    uint offset = pio_add_program(pio, &pdm_reader_program);
    pdm_reader_program_init(pio, sm, offset, PIONET_PDM_DATA_PIN, PIONET_PDM_CLK_PIN);
    printf("[PIO-Net] State Machine 0 initialized and capturing...\n");

    // Claim a DMA channel
    dma_chan = dma_claim_unused_channel(true);

    // Configure DMA
    dma_channel_config cfg = dma_channel_get_default_config(dma_chan);
    channel_config_set_transfer_data_size(&cfg, DMA_SIZE_32);       // 32-bit words
    channel_config_set_read_increment(&cfg, false);                  // always read from same FIFO address
    channel_config_set_write_increment(&cfg, true);                  // advance through RAM buffer
    channel_config_set_dreq(&cfg, pio_get_dreq(pio, sm, false));    // PIO RX DREQ — triggers on each word

    dma_channel_configure(
        dma_chan,
        &cfg,
        pdm_buf[0],             // initial write destination: buffer 0
        &pio->rxf[sm],          // read from PIO RX FIFO
        BUF_WORDS,              // 512 transfers then interrupt
        false                   // don't start yet
    );

    // Set up DMA completion interrupt
    dma_channel_set_irq0_enabled(dma_chan, true);
    irq_set_exclusive_handler(DMA_IRQ_0, dma_isr);
    irq_set_enabled(DMA_IRQ_0, true);

    // Start DMA
    dma_channel_start(dma_chan);

    printf("[PIO-Net] DMA armed, capturing to double buffer...\n");

    uint32_t total_words    = 0;
    uint32_t total_triggers = 0;
    uint32_t ones_count     = 0;
    uint32_t report_count   = 0;

    while (1) {
        // Wait for DMA to signal a full buffer
        if (!buf_flag) {
            continue; // CPU is idlly spinning here, we can use __wfi() to make it sleep till interrupt is fired
        }
        buf_flag = false;

        // Process the completed buffer
        uint32_t *buf = pdm_buf[buf_ready];
        for (uint32_t i = 0; i < BUF_WORDS; i++) {
            uint32_t pdm_word = buf[i];
            total_words++;
            ones_count += __builtin_popcount(pdm_word);

            uint32_t match = pionet_compute_match(pdm_word, PIONET_STAGE1_PATTERN);
            if (match >= PIONET_STAGE1_THRESHOLD) {
                total_triggers++;
                led_hold = 1000; // Hold LED on for ~31 ms (1000 words * 31.25 us)
                gpio_put(PIONET_LED_PIN, 1);
            } else if (led_hold > 0) {
                led_hold--;
            } else {
                gpio_put(PIONET_LED_PIN, 0);
            }
        }

        // Telemetry every ~1 second (32000 words)
        report_count += BUF_WORDS;
        if (report_count >= 32000) {
            float bit_density = (float)ones_count / (float)(report_count * 32);
            printf("[Telemetry] Words: %lu | Density: %.3f | Stage-1 Triggers: %lu\n",
                   (unsigned long)total_words,
                   bit_density,
                   (unsigned long)total_triggers);
            report_count = 0;
            ones_count   = 0;
        }
    }
}