#!/usr/bin/env python3
"""
Unit test for DS-6C-2F-E:
Verify Figure 5 dynamically updates when input summary metrics change,
proving that hardcoded placeholder data has been removed.
"""
import os
import sys
import json
import hashlib
import tempfile
import unittest

# Import the figure generator
EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(EXP_DIR, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from generate_figures import generate_all_figures

class TestFigure5Dynamic(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.processed_dir = os.path.join(self.test_dir.name, "processed")
        self.traces_dir = os.path.join(self.test_dir.name, "traces")
        self.output_dir = os.path.join(self.test_dir.name, "figures")
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.traces_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_figure5_changes_with_input_metrics(self):
        # Scenario 1: Metric Set A
        metric_a = {
            "timing_reconciliation": {
                "simulated_device_latency_ms": 64.5,
                "simulated_ddr5_time_ms": 62.4,
                "simulated_internal_compute_comm_ms": 2.1,
                "analytical_compute_time_ms": 0.0045,
                "host_wallclock_seconds": 160.0
            }
        }
        with open(os.path.join(self.processed_dir, "summary_metrics.json"), "w") as f:
            json.dump(metric_a, f)

        generate_all_figures(self.processed_dir, self.traces_dir, self.output_dir)
        
        fig5_png = os.path.join(self.output_dir, "fig5_compute_vs_memory_overlap.png")
        fig5_pdf = os.path.join(self.output_dir, "fig5_compute_vs_memory_overlap.pdf")
        
        self.assertTrue(os.path.exists(fig5_png), "Figure 5 PNG must exist")
        self.assertTrue(os.path.exists(fig5_pdf), "Figure 5 PDF must exist")
        
        with open(fig5_png, "rb") as f:
            hash_a = hashlib.sha256(f.read()).hexdigest()

        # Scenario 2: Metric Set B (different values)
        metric_b = {
            "timing_reconciliation": {
                "simulated_device_latency_ms": 128.9,
                "simulated_ddr5_time_ms": 124.8,
                "simulated_internal_compute_comm_ms": 4.1,
                "analytical_compute_time_ms": 0.0090,
                "host_wallclock_seconds": 320.0
            }
        }
        with open(os.path.join(self.processed_dir, "summary_metrics.json"), "w") as f:
            json.dump(metric_b, f)

        generate_all_figures(self.processed_dir, self.traces_dir, self.output_dir)
        
        with open(fig5_png, "rb") as f:
            hash_b = hashlib.sha256(f.read()).hexdigest()

        # Assert that the image changed when input metrics changed
        self.assertNotEqual(
            hash_a, hash_b,
            "Figure 5 output hash must differ when input metrics change (hardcoded placeholder eliminated)."
        )
        print(f"[TEST PASS] Figure 5 dynamic check passed: Hash A ({hash_a[:12]}...) != Hash B ({hash_b[:12]}...)")

if __name__ == "__main__":
    unittest.main()
