# 🎙️ PIO-Net: Ultra-Low-Power Cascaded Acoustic Wake-Up on RP2040 Microcontrollers

[![Tests](https://img.shields.io/badge/Tests-34%2F34%20Passing-brightgreen)](verification/)
[![Target Conference](https://img.shields.io/badge/Target-ACM%20SenSys%202027-blue)](https://sensys27.hotcrp.com/)
[![Target Symposium](https://img.shields.io/badge/Target-IIT%20Bombay%20ResCon%202026-orange)](https://unstop.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Authors:** Gigi Koneti\*, Adarsha Udupa Baikady\*, Anukul Anand\*  
*\*All authors contributed equally to this work (Co-First Authorship).*

---

## 📌 Overview

**PIO-Net** is an ultra-low-power, two-stage cascaded keyword wake-up system designed for the Raspberry Pi RP2040 microcontroller. 

Continuous acoustic keyword spotting (KWS) typically demands tens of milliwatts, draining portable batteries within hours. PIO-Net solves this by decoupling acoustic tripwire monitoring from the main CPU using a **Two-Gear Dynamic Voltage and Frequency Scaling (DVFS)** architecture co-designed with the RP2040's **Programmable I/O (PIO)** hardware subsystem:

1. **Quiescent Listening Gear (24 MHz):**
   - System clock is downscaled to **24 MHz** with all unused peripherals gated (`pll_usb` off, `clk_adc` stopped, UART/SPI/I2C/PWM held in reset).
   - **Stage 1 (Hardware 1-Bit Energy Gate):** PIO state machine 1 directly ingests the 1.024 MHz 1-bit Pulse Density Modulation (PDM) stream and evaluates a dual-tier energy gate in pure silicon ($\ge 19$ ones within a 32-clock window, requiring 12 consecutive qualifying blocks for ~0.2–0.4 ms persistence).
   - Core 0 remains in deep `__wfi()` sleep with **0 Hz CPU wakeups** during silence and ambient noise, drawing only **7.00 mA (28.6 mW @ 4.08V)**.

2. **Active Inference Burst Gear (125 MHz):**
   - On an acoustic match, PIO asserts an IRQ that triggers a glitchless clock transition (safely parked on 12 MHz `clk_ref`) upscaling the system to **125 MHz**.
   - **Stage 2 (On-Demand M0+ Classifier):** Core 0 captures 1.0 s of decimated audio, extracts a [39 frames $\times$ 16 Mel bins] spectrogram in **Feat ~87 ms**, and executes an INT8 quantized depthwise separable convolutional neural network (`DSCNNMicro`) in **Infer ~24 ms at 125 MHz** (~110 ms total compute burst, drawing 24.65 mA).
   - Once classification finishes, the clock immediately downscales back to 24 MHz and Core 0 returns to sleep.

```
[Digital PDM Microphone]
           │ (1.024 MHz 1-bit PDM stream)
           ▼
┌─────────────────────────────────────────────────────────────┐
│ RP2040 PIO Subsystem (State Machine 1)                      │
│ • Operating Clock: 24 MHz Quiescent System Clock            │
│ • Dual-tier temporal energy gate in silicon                 │
│ • Inner block: >= 19 ones within 32 PDM clocks              │
│ • Outer persistence: 12 consecutive blocks (~0.2-0.4 ms)    │
└──────────────────────────┬──────────────────────────────────┘
                           │ (12 consecutive blocks qualified?)
                           ├────────────── No  ──> Reset counter, stay asleep (0 Hz CPU wakeups)
                           ▼ Yes (Acoustic Tripwire Fired / IRQ 0)
┌─────────────────────────────────────────────────────────────┐
│ Two-Gear DVFS Controller (Glitchless Frequency Upscale)     │
│ • Transitions clk_sys from 24 MHz -> 125 MHz                │
│ • DMA circular buffer drains audio into SRAM                │
└──────────────────────────┬──────────────────────────────────┘
                           │ Wake-up IRQ
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ ARM Cortex-M0+ Core 0 @ 125 MHz (Burst Inference)           │
│ • Audio Capture: 1.0 s (16,000 PCM samples)                 │
│ • Feature Extractor: Mel Spectrogram (Feat ~87 ms)          │
│ • Stage-2 Classifier: DSCNNMicro INT8 (Infer ~24 ms)        │
│ • Total Wake-to-Decision Latency: ~1.1 s                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ├─> Keyword Detected: Action trigger
                           ▼
               [Downscale to 24 MHz & Return to WFI Sleep]
```

---

## 🚀 Key Results & Performance Benchmark

### Empirical Hardware Telemetry (INA226 External Power Monitor @ VSYS = 4.08V)

Physical supply voltage and current were captured in real-time using a calibrated high-side INA226 current sensor sampled by an external DAQ over dedicated hardware runs across all three firmware targets:

| Metric | Config A: Continuous Software VAD | Config B: Duty-Cycled Wake | Config C: PIO-Net (Ours) |
| :--- | :---: | :---: | :---: |
| **Quiescent System Clock** | 125 MHz | 125 MHz | **24 MHz** |
| **Burst System Clock** | 125 MHz | 125 MHz | **125 MHz** |
| **CPU Wake Rate in Silence** | 100% Active Polling | 100 Hz Timer Wakeups | **0 Hz (Zero CPU Wakeups)** |
| **Measured Idle Current ($I_{\text{listen}}$)** | **23.90 mA** | **20.55 mA** | **7.00 mA** |
| **Idle Power @ 4.08V ($P_{\text{listen}}$)** | **97.5 mW** | **83.8 mW** | **28.6 mW** |
| **Peak Inference Current ($I_{\text{burst}}$)** | **24.42 mA** | **24.30 mA** | **24.65 mA** |
| **Peak Inference Power ($P_{\text{burst}}$)** | **99.6 mW** | **99.1 mW** | **100.6 mW** |
| **Idle Power Savings vs. Config A** | *Baseline (0%)* | *14.0% saved ($1.16\times$)* | **70.7% saved ($3.41\times$ lower)** |
| **Idle Power Savings vs. Config B** | — | *Baseline (0%)* | **65.9% saved ($2.93\times$ lower)** |

### Operational Battery Lifetime Comparison

Assuming a standard portable power source (e.g., 2× AAA pack or 500 mAh 3.7V/4.1V LiPo cell) under realistic acoustic conditions (99.0% silence/ambient noise, 1.0% keyword activation):

| Firmware Architecture | Quiescent Power | Active Burst Power | Effective Average Power | Battery Lifetime (500 mAh LiPo) |
| :--- | :---: | :---: | :---: | :---: |
| **Config A (Continuous Software VAD)** | 97.5 mW | 99.6 mW | 97.5 mW | **20.9 hours (0.87 days)** |
| **Config B (Duty-Cycled 10ms Wake)** | 83.8 mW | 99.1 mW | 84.0 mW | **24.3 hours (1.01 days)** |
| **Config C: PIO-Net (Two-Gear DVFS)** | **28.6 mW** | **100.6 mW** | **29.3 mW** | **69.6 hours (2.90 days)** |

> **Key Architectural Finding:** Standard software duty-cycling (Config B) only yields a modest 14% power reduction over continuous polling (Config A) because the 125 MHz clock tree and peripheral crossbar continue dissipating dynamic charge ($C \cdot V^2 \cdot f$) during `__wfi()`. PIO-Net achieves a **$3.41\times$ power reduction** by offloading the wake tripwire to autonomous PIO silicon, enabling true 24 MHz idle downscaling.

---

### Pipeline Stage & Latency Audit (Cortex-M0+ @ 125 MHz)

| Pipeline Stage / Metric | Basis | Measured Value | Implementation / Hardware Setup |
|---|---|---|---|
| **Stage-1 Detection Window** | **MEASURED** | $\ge 19$ ones within 32 clocks | Inner sub-block threshold evaluated in PIO1 (`pdm_stage1.pio`) |
| **Stage-1 Persistence Gate** | **MEASURED** | 12 consecutive blocks | Persistence duration: ~0.2–0.4 ms at 1.024 MHz PDM clock before IRQ assert |
| **Quiescent CPU Duty Cycle** | **MEASURED** | 0.0% | Core 0 in `__wfi()`, silent ring DMA keeps microphone clock alive (0 CPU IRQs) |
| **Audio Capture Window** | **MEASURED** | 1.0 s | 16,000 PCM samples decimated at 64x popcount from 1.024 MHz stream |
| **Stage-2 Feature Extraction** | **MEASURED** | Feat ~87 ms | [39 $\times$ 16] Mel-spectrogram with 32-bit BFP FFT on Cortex-M0+ |
| **Stage-2 Classifier Inference** | **MEASURED** | Infer ~24 ms | DSCNNMicro INT8 inference at 125 MHz on Cortex-M0+ hardware multiplier |
| **Total Wake-to-Decision** | **MEASURED** | ~1.1 s | From Stage-1 IRQ assert to classification output (including 1.0 s capture) |
| **Stage-2 Classification Acc.** | **MEASURED** | 82.4% | Full-resolution Q8 features on Speech Commands 12-class benchmark |

---

## 📂 Repository Structure

```
PIO-Net/
├── ahb_sim/                 # AHB 4-port crossbar cycle-accurate bus simulator
├── baseline/                # 3rd-order CIC decimator, Energy VAD, and WebRTC VAD
│   └── vad_reference.py     # System configurations power model (Config A, B, PIO-Net)
├── dataset/                 # Dataset curation and PDM conversion
│   ├── download_curate.py   # Benchmark curation & noise suite generator
│   └── pcm_to_pdm.py        # 2nd-order Sigma-Delta 1.024 MHz PDM modulator
├── dma_sim/                 # DMA ping-pong controller cycle-accurate simulator
├── docs/                    # Research reports, surveys, and publication figures
│   ├── figures/             # Camera-ready vector PDFs and high-res PNGs
│   │   ├── pionet_confusion_matrix.pdf
│   │   ├── pionet_det_curves.pdf
│   │   └── pionet_energy_waterfall.pdf
│   ├── Gigi_Research_and_Implementation_Report.md  # Comprehensive research chronicle
│   ├── Detection_Metrics_Report.md
│   ├── Noise_Robustness_Report.md
│   └── Go_NoGo_Checkpoint_Report.md
├── energy_model/            # Physical RP2040 dynamic & static energy profiler
│   ├── clap_detector.py     # Real-time streaming INA226 current/voltage telemetry parser
│   └── yes_test_log.txt     # Empirical hardware INA226 capture logs
├── evaluation/              # Sim-to-real noise robustness and visual generation
│   ├── evaluate_noise_robustness.py  # ITU-T P.56 active speech & RIR SNR sweeps
│   └── generate_detection_metrics.py # ACM publication figures generator
├── nas/                     # Stage-1 matched filter optimization
│   ├── train_nas.py         # Differentiable PyTorch STE with Dirichlet regularization
│   ├── pionet_pattern.h     # Exported C header for RP2040 PIO
│   └── pionet_pattern.json  # Metadata and threshold calibration
├── paper/                   # Camera-ready ACM LaTeX snippets
│   └── tables_and_figures.tex
├── pio_sim/                 # RP2040 PIO state machine cycle-accurate simulator
├── pionet_comparisons/      # Standalone SenSys 2027 benchmark targets (Config A, B, C)
│   ├── CMakeLists.txt       # Unified Ninja/CMake build file
│   ├── config_a_cpu_continuous.c # 125 MHz 100% active polling baseline
│   ├── config_b_dutycycle.c # 125 MHz 10ms periodic timer buffer-wake baseline
│   ├── config_c_pionet.c    # 24 MHz -> 125 MHz Two-Gear DVFS PIO-Net target
│   └── README.md            # Empirical measurement protocol and verification data
├── pionet_core/             # Production RP2040 C firmware implementation
│   ├── CMakeLists.txt       # Firmware build with clock wrap & peripheral gating
│   ├── main.c               # Two-gear DVFS executive, PDM DMA ring, and Stage-2 driver
│   ├── pdm_reader.pio       # PIO0 state machine: 1.024 MHz PDM clock & bit capture
│   ├── pdm_stage1.pio       # PIO1 state machine: Autonomous 1-bit temporal energy gate
│   ├── stage2_features.c    # 16-channel Mel-filterbank with Radix-2 BFP FFT
│   └── stage2_infer.c       # INT8 CMSIS-NN DSCNN inference engine
├── stage2/                  # Stage-2 DSCNNMicro neural network
│   ├── model.py             # PyTorch architecture
│   ├── train_stage2.py      # Training, BatchNorm folding, and CMSIS-NN exporter
│   ├── model_data.h         # Exported INT8 CMSIS-NN C header
│   └── model_config.json    # Hardware memory & latency budget audit
├── verification/            # Automated test suite (34 unit and system tests)
│   ├── gonogo_checkpoint.py # Official Go/No-Go feasibility evaluator
│   └── test_*.py            # PyTest automated unit test modules
├── Makefile                 # One-command build and verification entrypoint
├── .gitignore               # Strict exclude list for dataset bloat and caches
└── README.md
```

---

## 🛠️ Quickstart & Reproduction

### 1. Installation
Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/GiGiKoneti/PIO-Net.git
cd PIO-Net
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install torch numpy scipy soundfile matplotlib pytest
```

### 2. Run Complete Verification Suite (34/34 Passing)
```bash
make verify
```

### 3. Build & Flash Comparison Targets
To replicate the SenSys 2027 power benchmark on an RP2040 board with an INA226 current monitor:
```bash
cd pionet_comparisons
mkdir build && cd build
cmake -G Ninja -DPICO_BOARD=pico ..
ninja
```
Flashing targets:
- `config_a_continuous.uf2`: 125 MHz Continuous Software Polling (measures ~23.90 mA)
- `config_b_dutycycle.uf2`: 125 MHz Duty-Cycled 10ms Wake (measures ~20.55 mA)
- `config_c_pionet.uf2`: Two-Gear DVFS PIO-Net (measures ~7.00 mA quiescent, 24.65 mA burst)

---

## 📄 License & Citation

This project is licensed under the MIT License.

```bibtex
@inproceedings{koneti2027pionet,
  title     = {PIO-Net: Ultra-Low-Power Cascaded Acoustic Wake-Up on Microcontrollers via Programmable I/O Gating},
  author    = {Koneti, Gigi and Baikady, Adarsha Udupa and Anand, Anukul},
  booktitle = {Proceedings of the International Conference on Embedded Artificial Intelligence and Sensing Systems (ACM SenSys)},
  year      = {2027}
}
```
