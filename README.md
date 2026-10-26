# 🎙️ PIO-Net: Ultra-Low-Power Cascaded Acoustic Wake-Up on RP2040 Microcontrollers

[![Tests](https://img.shields.io/badge/Tests-34%2F34%20Passing-brightgreen)](verification/)
[![Target Conference](https://img.shields.io/badge/Target-ACM%20SenSys%202027-blue)](https://sensys27.hotcrp.com/)
[![Target Symposium](https://img.shields.io/badge/Target-IIT%20Bombay%20ResCon%202026-orange)](https://unstop.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Authors:** Gigi Koneti\*, Adarsh\*, Anukul\*  
*\*All authors contributed equally to this work (Co-First Authorship).*

---

## 📌 Overview

**PIO-Net** is an ultra-low-power, two-stage cascaded keyword wake-up system designed for the Raspberry Pi RP2040 microcontroller. 

Continuous acoustic keyword spotting (KWS) typically demands tens of milliwatts, draining portable batteries in days. PIO-Net solves this by offloading acoustic tripwire monitoring to the RP2040's **Programmable I/O (PIO)** hardware subsystem:
1. **Stage 1 (Hardware 1-Bit Energy Gate):** The PIO state machine directly ingests the 1.024 MHz 1-bit Pulse Density Modulation (PDM) stream from a digital microphone and evaluates a dual-tier energy gate in hardware ($\ge 19$ ones within a 32-clock window, requiring 12 consecutive qualifying blocks for a persistence of about 0.2–0.4 ms) before asserting an interrupt, while the main CPU stays in deep sleep (`__wfi()`).
2. **Stage 2 (On-Demand M0+ Classifier):** When triggered by the PIO interrupt, ARM Cortex-M0+ Core 0 wakes up, captures 1.0 s of decimated audio, extracts a [39 frames $\times$ 16 Mel bins] spectrogram in **Feat ~87 ms**, and executes an INT8 quantized depthwise separable convolutional neural network (`DSCNNMicro`) in **Infer ~24 ms at 125 MHz**, achieving a total wake-to-decision latency of **about 1.1 s** (including the 1.0 s capture).

```
[Digital PDM Microphone]
           │ (1.024 MHz 1-bit PDM stream)
           ▼
┌─────────────────────────────────────────────────────────────┐
│ RP2040 PIO Subsystem (State Machine 1)                      │
│ • Dual-tier temporal energy gate                            │
│ • Inner block: >= 19 ones within 32 PDM clocks              │
│ • Outer persistence: 12 consecutive blocks (~0.2-0.4 ms)    │
└──────────────────────────┬──────────────────────────────────┘
                           │ (12 consecutive blocks qualified?)
                           ├────────────── No  ──> Reset counter, stay asleep (0% CPU load)
                           ▼ Yes (Acoustic Tripwire Fired / IRQ 0)
┌─────────────────────────────────────────────────────────────┐
│ DMA Controller (Circular Ping-Pong Buffer)                  │
│ • Transfers audio words into SRAM                           │
│ • Zero AHB crossbar contention with CPU execution           │
└──────────────────────────┬──────────────────────────────────┘
                           │ Wake-up IRQ
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ ARM Cortex-M0+ Core 0 @ 125 MHz (Wakes from Deep Sleep)     │
│ • Audio Capture: 1.0 s (16,000 PCM samples)                 │
│ • Feature Extractor: Mel Spectrogram (Feat ~87 ms)          │
│ • Stage-2 Classifier: DSCNNMicro INT8 (Infer ~24 ms)        │
│ • Total Wake-to-Decision Latency: ~1.1 s                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                [Target Keyword Detected]
```

---

## 🚀 Key Results & Performance Benchmark

### Empirical Hardware Measurements (RP2040 Board @ 125 MHz)

| Pipeline Stage / Metric | Basis | Measured Value | Implementation / Hardware Setup |
|---|---|---|---|
| **Stage-1 Detection Window** | **MEASURED** | $\ge 19$ ones within 32 clocks | Inner sub-block threshold evaluated in PIO1 (`pdm_stage1.pio`) |
| **Stage-1 Persistence Gate** | **MEASURED** | 12 consecutive blocks | Persistence duration: about 0.2–0.4 ms at 1.024 MHz PDM clock before IRQ assert |
| **Quiescent CPU Duty Cycle** | **MEASURED** | 0.0% | Core 0 in `__wfi()`, silent ring DMA keeps microphone clock alive (0 CPU IRQs) |
| **Audio Capture Window** | **MEASURED** | 1.0 s | 16,000 PCM samples decimated at 64x popcount from 1.024 MHz stream |
| **Stage-2 Feature Extraction** | **MEASURED** | Feat ~87 ms | [39 $\times$ 16] Mel-spectrogram with 32-bit BFP FFT on Cortex-M0+ |
| **Stage-2 Classifier Inference** | **MEASURED** | Infer ~24 ms | DSCNNMicro INT8 inference at 125 MHz on Cortex-M0+ hardware multiplier |
| **Total Wake-to-Decision** | **MEASURED** | ~1.1 s | From Stage-1 IRQ assert to classification output (including 1.0 s capture) |
| **Stage-2 Classification Acc.** | **MEASURED** | 82.4% | Full-resolution Q8 features on Speech Commands 12-class benchmark |

### Comparative System Analysis

| Architectural Pipeline | Basis | VAD Tripwire | CPU Duty Cycle | Power (mW) [SIM] | 2x AAA Lifetime [SIM] | Macro Recall [SIM] | False-Wake Rate [SIM] |
|---|---|---|---|---|---|---|---|
| **Config A (Continuous CPU)** | **SIM** | WebRTC VAD | 100.0% | 65.01 mW | 39.2 h (1.6 days) | 90.8% | Baseline |
| **Config B (Periodic 20ms)** | **SIM** | Energy VAD | 12.6% | 26.67 mW | 95.6 h (4.0 days) | 88.2% | Baseline |
| **PIO-Net (Cascaded Analytical)** | **SIM** | 1-Bit PIO Gate | 0.0% (quiescent) | 17.95 mW | 142.1 h (5.9 days) | 90.8% | 0.0 / hour |

* **Hardware Stage-1 Gate [MEASURED]:** Gate requires $\ge 19$ ones within 32 clocks, 12 consecutive blocks, persistence about 0.2–0.4 ms at 1.024 MHz PDM clock.
* **Measured On-Board Latencies [MEASURED]:** Feat ~87 ms, Infer ~24 ms at 125 MHz; wake-to-decision about 1.1 s including a 1.0 s capture.
* **Quiescent CPU Load [MEASURED]:** 0.0% CPU duty cycle in sleep with silent background ring DMA keeping the PDM clock active with 0 CPU IRQs.
* **Power Reduction [SIM]:** 72.4% power reduction in analytical simulation (from 65.01 mW down to 17.95 mW, extending modeled 2x AAA battery lifetime from 39.2 h to 142.1 h / 5.9 days).
* **Noise Suppression [SIM]:** 99.74% noise variance suppression evaluated in acoustic simulation suites.
* **Bus Contention [SIM]:** Zero crossbar arbitration contention verified across 5,000 AHB simulation cycles.

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
git clone https://github.com/gigikoneti/PIO-Net.git
cd PIO-Net
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy scipy soundfile matplotlib pytest
```

### 2. Run Complete Verification Suite (34/34 Passing)
```bash
make verify
```

### 3. Run Microarchitecture Simulators (PIO, DMA, AHB)
```bash
make sim
```

### 4. Regenerate All Publication Figures
```bash
python evaluation/generate_detection_metrics.py
```
Outputs vector PDFs and high-resolution PNGs directly to `docs/figures/`.

---

## 📄 License & Citation

This project is licensed under the MIT License.

```bibtex
@inproceedings{koneti2027pionet,
  title     = {PIO-Net: Ultra-Low-Power Cascaded Acoustic Wake-Up on Microcontrollers via Programmable I/O Gating},
  author    = {Koneti, Gigi and Adarsh and Anukul},
  booktitle = {Proceedings of the International Conference on Embedded Artificial Intelligence and Sensing Systems (ACM SenSys)},
  year      = {2027}
}
```
