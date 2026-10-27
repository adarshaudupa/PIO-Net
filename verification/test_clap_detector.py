"""
==============================================================================
PIO-Net Unit & Verification Suite: Supply Current Clap Detector
Authors: Adarsh & Anukul (Embedded Systems) & Gigi Koneti (ML / Acoustic Lead)
Project: ACM SenSys 2027
==============================================================================
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from energy_model.clap_detector import ClapDetector, generate_benchmark_log


def test_parser_glued_stream():
    """Verify that serial stream parser extracts records with 100% accuracy even when glued without newlines."""
    glued_text = (
        "V: 4.124 V | I: 22.83 mAV: 4.120 V | I: 24.65 mAV: 4.112 V | I: 27.15 mA"
        "V: 4.124 V | I: 22.85 mA\r\nV: 4.120 V | I: 23.57 mA\nV: 4.124 V | I: 22.82 mA"
    )
    records = ClapDetector.parse_serial_stream(glued_text)
    assert len(records) == 6, f"Expected 6 records, parsed {len(records)}"
    assert records[0] == (4.124, 22.83)
    assert records[1] == (4.120, 24.65)
    assert records[2] == (4.112, 27.15)
    assert records[3] == (4.124, 22.85)
    assert records[4] == (4.120, 23.57)
    assert records[5] == (4.124, 22.82)


def test_reject_single_sample_blips():
    """Verify that isolated single-sample current spikes (+0.5 mA) are rejected and never trigger claps."""
    detector = ClapDetector(
        sample_rate_hz=45.0,
        trigger_delta_ma=0.40,
        hysteresis_delta_ma=0.20,
        min_consecutive_samples=2,
    )

    # 10 idle samples, 1 blip at +0.55 mA (23.38 mA), 10 idle samples
    samples = [(4.124, 22.83)] * 10 + [(4.122, 23.38)] + [(4.124, 22.84)] * 10
    for v, i in samples:
        detector.process_sample(v, i)
    claps = detector.finish()

    assert len(claps) == 0, f"Expected 0 claps (blip rejected), got {len(claps)}"


def test_benchmark_3_claps_glued_and_standard():
    """Verify exact 3-clap detection on benchmark dataset in both glued and newline-separated formats."""
    for is_glued in [True, False]:
        stream, _ = generate_benchmark_log(include_glued=is_glued)
        detector = ClapDetector(
            sample_rate_hz=45.0,
            trigger_delta_ma=0.40,
            hysteresis_delta_ma=0.20,
            min_consecutive_samples=2,
            merge_window_samples=10,
            v_drop_threshold_v=0.003,
        )
        claps = detector.process_stream(stream)

        assert len(claps) == 3, f"Glued={is_glued}: Expected exactly 3 claps, found {len(claps)}"

        # Validate Clap 1
        c1 = claps[0]
        assert abs(c1["peak_i"] - 24.65) < 0.05, f"Clap 1 peak expected ~24.65 mA, got {c1['peak_i']}"
        assert c1["min_v"] <= 4.118, f"Clap 1 expected V dip to ~4.116 V, got {c1['min_v']}"
        assert c1["v_confirmed"], "Clap 1 should have V-drop confirmed"

        # Validate Clap 2 (strongest + merged tail)
        c2 = claps[1]
        assert abs(c2["peak_i"] - 27.15) < 0.05, f"Clap 2 peak expected ~27.15 mA, got {c2['peak_i']}"
        assert c2["min_v"] <= 4.114, f"Clap 2 expected V dip to ~4.112 V, got {c2['min_v']}"
        assert c2["duration_samples"] >= 8, f"Clap 2 decay tail must be merged, duration was {c2['duration_samples']}"

        # Validate Clap 3 (weakest)
        c3 = claps[2]
        assert abs(c3["peak_i"] - 23.57) < 0.05, f"Clap 3 peak expected ~23.57 mA, got {c3['peak_i']}"
        assert c3["delta_i"] >= 0.50, f"Clap 3 delta expected >= 0.5 mA, got {c3['delta_i']}"


def test_merge_window_decay_tail():
    """Verify that a decay tail within the merge window (10 samples) counts as ONE clap, not two."""
    detector = ClapDetector(
        sample_rate_hz=45.0,
        trigger_delta_ma=0.40,
        hysteresis_delta_ma=0.20,
        min_consecutive_samples=2,
        merge_window_samples=10,
    )

    # Idle (22.84 mA) -> Peak (27.0 mA for 3 samples) -> drops to 23.0 mA (1 sample)
    # -> Tail (23.8 mA for 3 samples) -> Idle
    trace = (
        [(4.124, 22.84)] * 15
        + [(4.112, 27.00)] * 3
        + [(4.120, 23.00)] * 2   # Below release (22.84 + 0.20 = 23.04)
        + [(4.117, 23.80)] * 3   # Re-surge within merge window (< 10 samples)
        + [(4.124, 22.84)] * 20
    )

    for v, i in trace:
        detector.process_sample(v, i)
    claps = detector.finish()

    assert len(claps) == 1, f"Expected 1 merged clap, found {len(claps)}"
    assert claps[0]["peak_i"] == 27.00


def test_clap3_weak_margin_audit():
    """
    Audit Clap 3 margin:
    Peak I is ~23.57 mA (+0.73 mA above ~22.84 mA idle).
    Single-sample blips reach ~23.37 mA (+0.53 mA above idle).
    Trigger threshold is baseline + 0.40 mA (= 23.24 mA).
    Safety margin between Clap 3 and trigger threshold:
    23.57 - 23.24 = 0.33 mA.
    Margin between blip and trigger threshold:
    23.37 - 23.24 = +0.13 mA (crosses single-sample threshold, rejected ONLY by consecutive duration requirement!).
    """
    baseline_i = 22.84
    trig_thresh = baseline_i + 0.40  # 23.24 mA
    blip_peak = 23.37
    clap3_peak = 23.57

    # Clap 3 margin over trigger threshold
    clap3_margin = clap3_peak - trig_thresh
    assert clap3_margin > 0.30, f"Clap 3 margin over trigger should be > 0.30 mA, got {clap3_margin:.2f} mA"

    # Blip exceeds threshold by 0.13 mA, so consecutive sample check is critical
    blip_excess = blip_peak - trig_thresh
    assert blip_excess > 0.10, "Blips exceed amplitude threshold, proving consecutive check is mandatory"


if __name__ == "__main__":
    print("Running Clap Detector verification tests...")
    test_parser_glued_stream()
    print("  [PASS] test_parser_glued_stream")
    test_reject_single_sample_blips()
    print("  [PASS] test_reject_single_sample_blips")
    test_benchmark_3_claps_glued_and_standard()
    print("  [PASS] test_benchmark_3_claps_glued_and_standard")
    test_merge_window_decay_tail()
    print("  [PASS] test_merge_window_decay_tail")
    test_clap3_weak_margin_audit()
    print("  [PASS] test_clap3_weak_margin_audit")
    print("\n>>> ALL CLAP DETECTOR TESTS PASSED (5/5) <<<")

