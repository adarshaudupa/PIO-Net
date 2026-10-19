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
#include "hardware/sync.h"
#include "pdm_reader.pio.h"

// Double buffers: DMA ping-pongs between these
static uint32_t pdm_buf[2][BUF_WORDS];
static volatile uint8_t dma_filling = 0; // Buffer DMA is currently writing to
static volatile uint8_t buf_state[2] = {0, 0}; // Both buffers will be free at the start
static volatile uint32_t overrun_count = 0;
uint32_t led_hold = 0;

static int dma_chan;

static void dma_isr(void) {
    dma_hw->ints0 = 1u << dma_chan;

    // Mark the buffer DMA just finished as full (hand off to CPU)
    buf_state[dma_filling] = 1;

    // Find next free buffer for DMA
    uint8_t next = 1 - dma_filling;
    if (buf_state[next] == 0) {
        // Next buffer is free — use it
        dma_filling = next;
        dma_channel_set_trans_count(dma_chan, BUF_WORDS, false);
        dma_channel_set_write_addr(dma_chan, pdm_buf[dma_filling], true);
    } else {
        // Both buffers occupied — overrun, DMA stalls
        overrun_count++;
        // Don't rearm. PIO FIFO will fill and stall PIO state machine.
        // Main loop must call dma_rearm() after releasing a buffer.
    }
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
        // Find a full buffer to process
        uint8_t proc = 0xFF;
        for (uint8_t i = 0; i < 2; i++) {
            if (buf_state[i] == 1) { proc = i; break; }
        }
        if (proc == 0xFF) { __wfi(); continue; }

        // Lock it — mark as active so ISR won't rearm DMA into it
        buf_state[proc] = 2;

        // Process
        uint32_t *buf = pdm_buf[proc];
        for (uint32_t i = 0; i < BUF_WORDS; i++) {
            // ... matched filter, LED, counts ...
        }

        // Release buffer — DMA can now fill it again
        buf_state[proc] = 0;

        // If DMA stalled due to overrun, restart it now
        if (!dma_channel_is_busy(dma_chan)) {
            dma_filling = proc;
            buf_state[proc] = 0; // still free
            dma_channel_set_trans_count(dma_chan, BUF_WORDS, false);
            dma_channel_set_write_addr(dma_chan, pdm_buf[dma_filling], true);
            dma_channel_start(dma_chan);
        }
    }
}