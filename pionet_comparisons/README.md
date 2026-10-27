# PIO-Net Comparison Suite (ACM SenSys 2027)

This directory provides standalone firmware targets for reproducing the empirical power and latency benchmarks reported in the PIO-Net paper. All configurations share identical peripheral clock gating (`pll_deinit(pll_usb)`, `clock_stop(clk_adc)`, and `reset_block` for unused peripherals) and headless operation to guarantee an apples-to-apples comparison.

---

### Configurations

1. **Config A: Continuous Software VAD (`config_a_continuous`)**
   - **Architecture:** Traditional edge AI polling.
   - **System Clock:** 125 MHz continuously.
   - **Operation:** Core 0 runs at 100% duty cycle (no `__wfi()`). It decimate incoming PDM audio into PCM via boxcar popcounting and evaluates sliding-window energy in software. On speech detection, it runs the Stage-2 DSCNN classifier and immediately returns to polling.

2. **Config B: Duty-Cycled Buffer-Wake (`config_b_dutycycle`)**
   - **Architecture:** Classical low-power MCU sleep + periodic wake.
   - **System Clock:** 125 MHz.
   - **Operation:** PDM bitstream streams into a circular DMA buffer in the background while Core 0 sleeps in `__wfi()`. A hardware timer interrupt wakes Core 0 every 10 ms (100 Hz). The CPU decimates the 10 ms block and computes energy:
     - *Silence/Noise:* CPU returns to `__wfi()` in ~35 µs.
     - *Speech Onset:* CPU captures 1.0 s of audio, runs Stage-2 DSCNN inference, and returns to duty-cycled sleep.

3. **Config C: PIO-Net Two-Gear DVFS (`config_c_pionet`)**
   - **Architecture:** Proposed hardware-software co-design.
   - **System Clock:** **24 MHz quiescent listening**, upscaling to **125 MHz burst** on trigger.
   - **Operation:** PDM generation and Stage-1 acoustic matched filtering run autonomously in PIO state machines. Core 0 experiences **zero wakeups** during silence/noise. On acoustic match, PIO triggers an IRQ that switches `clk_sys` to 125 MHz, ingests speech, runs Stage-2 DSCNN, and drops back to 24 MHz.

---

### Empirical Measurements (INA226 @ VSYS = 4.08V)

| Metric | Config A: Continuous Polling | Config B: Duty-Cycled Wake | Config C: PIO-Net (Ours) |
| :--- | :---: | :---: | :---: |
| **Quiescent System Clock** | 125 MHz | 125 MHz | **24 MHz** |
| **Burst System Clock** | 125 MHz | 125 MHz | **125 MHz** |
| **Acoustic Trigger Method** | Continuous Software VAD | Periodic 10 ms Software VAD | **PIO1 Silicon Matched Filter** |
| **CPU Wakeups in Silence** | Continuous (100% active) | 100 wakes/sec (every 10 ms) | **0 wakes/sec (Pure silicon)** |
| **Quiescent Current ($I_{\text{listen}}$)** | **23.90 mA** | **20.55 mA** | **7.05 mA** |
| **Quiescent Power ($P_{\text{listen}}$)** | **97.4 mW** | **83.8 mW** | **28.8 mW** |
| **Inference Burst Current ($I_{\text{burst}}$)** | **24.42 mA** | **24.38 mA** | **24.65 mA** |
| **Inference Burst Power ($P_{\text{burst}}$)** | **99.5 mW** | **99.2 mW** | **100.8 mW** |
| **Quiescent Power Reduction** | *Baseline (1.00×)* | *14.0% saved ($1.16\times$)* | **70.5% saved ($3.39\times$ lower)** |
| **Advantage over Duty-Cycled** | — | *Baseline (1.00×)* | **65.7% saved ($2.91\times$ lower)** |

---

### Building the Targets

```bash
mkdir build && cd build
cmake -G Ninja -DPICO_BOARD=pico ..
ninja
```

Generated UF2 files:
- `config_a_continuous.uf2`
- `config_b_dutycycle.uf2`
- `config_c_pionet.uf2`
