"""
==============================================================================
PIO-Net: Acoustic Wake-Up & Power Profiling - Supply Current Clap Detector
Authors: Adarsh & Anukul (Embedded Systems) & Gigi Koneti (ML / Acoustic Lead)
Project: ACM SenSys 2027

Microarchitectural & Power Context:
----------------------------------
The RP2040 runs PIO-Net with Core 0 quiescent in __wfi() (~22.8 mA @ 4.12V).
When an acoustic clap occurs:
1. PIO1 hardware matched filter trips on the PDM bitstream energy burst.
2. Core 0 wakes up, activates the status LED, and captures 1.0 s of audio
   via DMA double-buffering (current rises to ~23.5 - 24.5 mA).
3. Stage-2 feature extraction and DSCNNMicro INT8 inference run at 125 MHz
   (current surges to ~26.7 - 27.2 mA for ~110 ms, causing supply V-sag).
4. An external INA226 power monitor sampled by an STM32 DAQ (pionet_daq)
   streams 'V: x.xxx V | I: xx.xx mA' over UART.

This module provides a robust streaming parser and two-stage clap detection
state machine with baseline EMA tracking, consecutive-sample blip rejection,
hysteresis release, merge windowing, and optional V-drop confirmation.
==============================================================================
"""

import re
from typing import List, Dict, Tuple, Optional, Union
import numpy as np


class ClapDetector:
    """
    Two-stage clap detector operating on INA226 current and voltage telemetry.

    Sample Rate Derivation:
    -----------------------
    pionet_daq main loop timing:
    - HAL_Delay(20):              20.0 ms
    - INA226 I2C reads @ 400 kHz: ~0.2 ms
    - UART2 TX @ 115200 baud:     ~2.1 ms (for 25-byte 'V: ... | I: ...\r\n')
    Total sample period Ts:       ~22.3 ms -> Fs ≈ 44.8 Hz (nominal 45 Hz).
    """

    def __init__(
        self,
        sample_rate_hz: float = 45.0,
        trigger_delta_ma: float = 0.40,
        hysteresis_delta_ma: float = 0.20,
        min_consecutive_samples: int = 2,
        merge_window_samples: int = 10,
        v_drop_threshold_v: float = 0.003,
        ema_alpha: float = 0.05,
    ):
        """
        Initialize the ClapDetector.

        Args:
            sample_rate_hz: DAQ sample rate in Hz (~45.0 Hz from pionet_daq timing).
            trigger_delta_ma: Current rise above baseline to trigger (~0.40 mA).
            hysteresis_delta_ma: Current rise above baseline to release (~0.20 mA).
            min_consecutive_samples: Consecutive samples required to trigger (2 rejects 1-sample blips).
            merge_window_samples: Max sample gap across which active decay tails are merged (10 samples ≈ 220 ms).
            v_drop_threshold_v: Minimum voltage dip to confirm secondary power sag (e.g. 0.003 V = 3 mV).
            ema_alpha: Exponential moving average coefficient for idle baseline updates.
        """
        self.sample_rate_hz = sample_rate_hz
        self.trigger_delta_ma = trigger_delta_ma
        self.hysteresis_delta_ma = hysteresis_delta_ma
        self.min_consecutive_samples = min_consecutive_samples
        self.merge_window_samples = merge_window_samples
        self.v_drop_threshold_v = v_drop_threshold_v
        self.ema_alpha = ema_alpha

        # State machine variables
        self.state = "IDLE"  # "IDLE", "PENDING", "ACTIVE", "MERGE_WAIT"
        self.baseline_i: Optional[float] = None
        self.baseline_v: Optional[float] = None

        self.consecutive_count: int = 0
        self.samples_since_release: int = 0
        self.sample_index: int = 0

        # Pending event (candidate during consecutive-sample verification)
        self.pending_start_idx: int = 0
        self.pending_samples: List[int] = []
        self.pending_i: List[float] = []
        self.pending_v: List[float] = []

        # Active event tracking
        self.current_event: Optional[Dict] = None
        self.detected_claps: List[Dict] = []

    def reset(self) -> None:
        """Reset internal state machine and event lists."""
        self.state = "IDLE"
        self.baseline_i = None
        self.baseline_v = None
        self.consecutive_count = 0
        self.samples_since_release = 0
        self.sample_index = 0
        self.pending_samples = []
        self.pending_i = []
        self.pending_v = []
        self.current_event = None
        self.detected_claps = []

    @staticmethod
    def parse_serial_stream(stream_text: str) -> List[Tuple[float, float]]:
        """
        Robust regex parser extracting (V, I) tuples from serial log stream.

        Handles:
        - Clean records with newlines (\\r\\n or \\n)
        - Records missing newlines that arrive glued together as '...mAV: x.xxx V | I: xx.xx mA...'
        - Extra spaces, tabs, or corrupted prefixes.

        Returns:
            List of (voltage_V, current_mA) tuples.
        """
        pattern = re.compile(r"V:\s*([0-9.]+)\s*V\s*\|\s*I:\s*([0-9.]+)\s*mA")
        records = []
        for match in pattern.finditer(stream_text):
            try:
                v = float(match.group(1))
                i = float(match.group(2))
                records.append((v, i))
            except (ValueError, IndexError):
                continue
        return records

    def process_sample(self, v: float, i: float) -> Optional[Dict]:
        """
        Process a single (V, I) measurement sample through the state machine.

        Returns:
            Dict representing a completed Clap event if one was finalized on this sample, else None.
        """
        idx = self.sample_index
        self.sample_index += 1

        # Initialize baseline on first sample
        if self.baseline_i is None or self.baseline_v is None:
            self.baseline_i = i
            self.baseline_v = v
            return None

        base_i: float = self.baseline_i
        base_v: float = self.baseline_v

        trig_thresh = base_i + self.trigger_delta_ma
        release_thresh = base_i + self.hysteresis_delta_ma

        emitted_event = None

        if self.state == "IDLE":
            # Update baseline ONLY during confirmed idle state
            self.baseline_i = (1.0 - self.ema_alpha) * base_i + self.ema_alpha * i
            self.baseline_v = (1.0 - self.ema_alpha) * base_v + self.ema_alpha * v

            if i >= trig_thresh:
                # Potential onset: enter PENDING state to verify multi-sample duration
                self.state = "PENDING"
                self.consecutive_count = 1
                self.pending_start_idx = idx
                self.pending_samples = [idx]
                self.pending_i = [i]
                self.pending_v = [v]
            else:
                self.consecutive_count = 0

        elif self.state == "PENDING":
            if i >= trig_thresh:
                self.consecutive_count += 1
                self.pending_samples.append(idx)
                self.pending_i.append(i)
                self.pending_v.append(v)

                if self.consecutive_count >= self.min_consecutive_samples:
                    # Verified trigger! Duration exceeds single-sample blip limit.
                    self.state = "ACTIVE"
                    self.current_event = {
                        "clap_id": len(self.detected_claps) + 1,
                        "start_idx": self.pending_start_idx,
                        "baseline_i": base_i,
                        "baseline_v": base_v,
                        "samples": list(self.pending_samples),
                        "i_values": list(self.pending_i),
                        "v_values": list(self.pending_v),
                        "peak_i": max(self.pending_i),
                        "min_v": min(self.pending_v),
                    }
                    self.pending_samples = []
                    self.pending_i = []
                    self.pending_v = []
            else:
                # Single-sample blip rejected! Drop back to IDLE without triggering.
                self.state = "IDLE"
                self.consecutive_count = 0
                self.pending_samples = []
                self.pending_i = []
                self.pending_v = []
                # Update baseline with this non-triggering sample
                self.baseline_i = (1.0 - self.ema_alpha) * base_i + self.ema_alpha * i
                self.baseline_v = (1.0 - self.ema_alpha) * base_v + self.ema_alpha * v

        elif self.state == "ACTIVE":
            # Event is ongoing: collect samples and track peak I and min V
            if self.current_event is not None:
                self.current_event["samples"].append(idx)
                self.current_event["i_values"].append(i)
                self.current_event["v_values"].append(v)
                self.current_event["peak_i"] = max(self.current_event["peak_i"], i)
                self.current_event["min_v"] = min(self.current_event["min_v"], v)

            if i < release_thresh:
                # Signal dropped below hysteresis release threshold
                self.state = "MERGE_WAIT"
                self.samples_since_release = 0

        elif self.state == "MERGE_WAIT":
            self.samples_since_release += 1
            if self.current_event is not None:
                self.current_event["samples"].append(idx)
                self.current_event["i_values"].append(i)
                self.current_event["v_values"].append(v)
                self.current_event["peak_i"] = max(self.current_event["peak_i"], i)
                self.current_event["min_v"] = min(self.current_event["min_v"], v)

            if i >= trig_thresh:
                # Re-surge within merge window: re-activate existing event (merges decay tail / secondary burst)
                self.state = "ACTIVE"
                self.samples_since_release = 0
            elif self.samples_since_release >= self.merge_window_samples:
                # Merge window expired with signal staying low: finalize and emit event
                emitted_event = self._finalize_event()
                self.state = "IDLE"
                self.consecutive_count = 0
                self.current_event = None

        return emitted_event

    def _finalize_event(self) -> Optional[Dict]:
        """Calculate event metrics and append to detected_claps list."""
        if self.current_event is None:
            return None

        evt = self.current_event
        release_thresh = evt["baseline_i"] + self.hysteresis_delta_ma

        # Determine last active sample before the trailing release gap
        last_active_idx = evt["start_idx"]
        for s_idx, val in zip(evt["samples"], evt["i_values"]):
            if val >= release_thresh:
                last_active_idx = s_idx

        evt["end_idx"] = last_active_idx
        evt["duration_samples"] = max(1, evt["end_idx"] - evt["start_idx"] + 1)
        evt["duration_ms"] = evt["duration_samples"] * (1000.0 / self.sample_rate_hz)
        evt["delta_i"] = evt["peak_i"] - evt["baseline_i"]
        evt["v_drop"] = evt["baseline_v"] - evt["min_v"]
        evt["v_confirmed"] = evt["v_drop"] >= self.v_drop_threshold_v

        self.detected_claps.append(evt)
        return evt

    def finish(self) -> List[Dict]:
        """Flush any pending or active event at the end of stream."""
        if self.state in ("ACTIVE", "MERGE_WAIT") and self.current_event is not None:
            self._finalize_event()
            self.state = "IDLE"
            self.current_event = None
        return self.detected_claps


    def process_stream(self, stream_text: str) -> List[Dict]:
        """
        Convenience method: parse full stream text and process all samples.

        Returns:
            List of detected clap event dictionaries.
        """
        records = self.parse_serial_stream(stream_text)
        for v, i in records:
            self.process_sample(v, i)
        return self.finish()


def generate_benchmark_log(include_glued: bool = True) -> Tuple[str, List[Dict]]:
    """
    Generate synthetic test data matching the empirical log specification:
    - Idle: V ≈ 4.124 V, I ≈ 22.80 - 22.87 mA with ±0.05 mA noise.
    - Clap 1: I peaks at 24.65 mA over ~6 samples. V dips to ~4.116 V.
    - Clap 2 (strongest): I = 26.7 - 27.15 mA for ~5 samples, decay tail of 23.1 - 24.2 mA
      for ~5 more. V dips to 4.112 V.
    - Clap 3 (weakest): I peaks at 23.57 mA (+0.73 mA over idle).
    - Isolated single-sample blips of ~+0.5 mA (23.30 - 23.37 mA) during idle.
    """
    rng = np.random.RandomState(42)
    records = []

    def add_idle(n_samples: int):
        for _ in range(n_samples):
            v = 4.124 + rng.uniform(-0.001, 0.001)
            i = 22.835 + rng.uniform(-0.035, 0.035)
            records.append((round(v, 3), round(i, 2)))

    def add_blip(blip_i: float = 23.35):
        # 1-sample isolated spike
        records.append((4.122, blip_i))

    # 1. Initial Idle
    add_idle(30)

    # 2. Isolated Blip 1 (NOT a clap)
    add_blip(23.32)
    add_idle(25)

    # 3. Clap 1: Peaks at 24.65 mA over ~6 samples, V dips to 4.116 V
    c1_i = [23.40, 24.10, 24.65, 24.40, 23.85, 23.15]
    c1_v = [4.122, 4.118, 4.116, 4.117, 4.119, 4.121]
    for v, i in zip(c1_v, c1_i):
        records.append((v, i))

    # 4. Idle between Clap 1 and Clap 2
    add_idle(35)

    # 5. Isolated Blip 2 (NOT a clap)
    add_blip(23.37)
    add_idle(20)

    # 6. Clap 2 (strongest): 5 samples @ 26.7 - 27.15 mA + 5-sample decay tail @ 23.1 - 24.2 mA
    c2_peak_i = [26.70, 26.95, 27.15, 27.05, 26.80]
    c2_peak_v = [4.113, 4.112, 4.112, 4.112, 4.113]
    c2_tail_i = [24.20, 23.80, 23.60, 23.35, 23.10]
    c2_tail_v = [4.117, 4.118, 4.119, 4.120, 4.122]
    for v, i in zip(c2_peak_v + c2_tail_v, c2_peak_i + c2_tail_i):
        records.append((v, i))

    # 7. Idle between Clap 2 and Clap 3
    add_idle(40)

    # 8. Isolated Blip 3 (NOT a clap)
    add_blip(23.30)
    add_idle(20)

    # 9. Clap 3 (weakest): peaks at 23.57 mA (+0.73 mA over baseline), V dips to 4.120 V
    c3_i = [23.30, 23.57, 23.45, 23.12]
    c3_v = [4.122, 4.120, 4.121, 4.123]
    for v, i in zip(c3_v, c3_i):
        records.append((v, i))

    # 10. Trailing Idle
    add_idle(30)

    # Format into serial string
    if include_glued:
        # Intentionally glue some records together as '...mAV: ...' to verify parser robustness
        lines = []
        for idx, (v, i) in enumerate(records):
            rec = f"V: {v:.3f} V | I: {i:.2f} mA"
            # Randomly omit newline for ~30% of records to simulate glued stream
            if idx % 3 == 0 and idx < len(records) - 1:
                lines.append(rec)  # No newline, next record concatenates directly
            else:
                lines.append(rec + "\r\n")
        stream_text = "".join(lines)
    else:
        stream_text = "\r\n".join(f"V: {v:.3f} V | I: {i:.2f} mA" for v, i in records) + "\r\n"

    ground_truth = [
        {"clap_id": 1, "approx_peak_i": 24.65, "min_v": 4.116},
        {"clap_id": 2, "approx_peak_i": 27.15, "min_v": 4.112},
        {"clap_id": 3, "approx_peak_i": 23.57, "min_v": 4.120},
    ]

    return stream_text, ground_truth


if __name__ == "__main__":
    print("=== PIO-Net Clap Detection Benchmark ===")
    stream, truth = generate_benchmark_log(include_glued=True)
    detector = ClapDetector(
        sample_rate_hz=45.0,
        trigger_delta_ma=0.40,
        hysteresis_delta_ma=0.20,
        min_consecutive_samples=2,
        merge_window_samples=10,
        v_drop_threshold_v=0.003,
    )
    claps = detector.process_stream(stream)

    print(f"Total claps detected: {len(claps)} (Expected: 3)")
    for c in claps:
        print(
            f"  - Clap #{c['clap_id']}: Peak I = {c['peak_i']:.2f} mA "
            f"(Delta: +{c['delta_i']:.2f} mA), Min V = {c['min_v']:.3f} V "
            f"(Drop: -{c['v_drop']*1000:.1f} mV, V-Confirmed: {c['v_confirmed']}), "
            f"Duration: {c['duration_samples']} samples ({c['duration_ms']:.1f} ms)"
        )
