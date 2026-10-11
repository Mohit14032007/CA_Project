# DeepSeek-V3 5-Step Controlled Full-Depth Experiment (DS-6C-4)

**Run Root Directory:** `reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/`  
**Execution Binary:** `build/run_deepseekv3_mixtral_style`  
**Effective Configuration:** `configs/deepseekv3_b1_full_run_5step.yaml`  
**Execution Status:** `SUCCESS` (Exit Code `0`, Termination Reason: `NORMAL`)  
**Completed At:** `2026-10-11T15:28:01Z`  

---

## 1. Workload Specification

- **Model Architecture:** DeepSeek-V3-shaped (60 total layers: 3 dense attention layers + 57 MoE layers).
- **Expert Routing:** 256 routed experts per MoE layer, top-$k = 8$, 1 shared expert per MoE layer.
- **Precision:** FP16 accounting (`precision_byte: 2`), expert footprint = 88,080,384 bytes (~88.08 MB).
- **Total Routed Expert Inventory:** 14,592 experts = 1,285,268,963,328 bytes (~1.285 TB).
- **Hardware Profile:** Single NVIDIA A100-SXM4-80GB class device (`num_node: 1`, `num_device: 1`).
- **Memory Subsystem:**
  - On-chip HBM3: 80,000,000,000 bytes (~74.51 GiB / 80 GB) LRU expert cache.
  - Off-chip DDR5: 2.75 TB / 2.5 TiB backing store (10,485,760 rows, row utilization 46.76%).
- **Serving / Sequence Parameters:**
  - Batch size: 1 (`max_batch_size: 1`).
  - Context / Sequence Length: `input_len: 2048`, `output_len: 10`.
  - Sequential Decoding Iterations: 5 decoding steps (`iter: 5`, tokens 2048 to 2052).
- **Diagnostic / Visualization:**
  - Gantt export disabled (`export_gantt: false`).
  - Raw telemetry and CSV outputs enabled.

---

## 2. Empirical Execution Metrics

- **Host Wall-Clock Runtime:** **21,159.0 seconds** (**5.877 hours** / 352.65 minutes).
  - Pre-flight estimate: 21,000 to 22,600 seconds (5.83 to 6.28 hours).
  - Variance: -0.8% from midpoint estimate.
- **Watchdog Timeout:** 25,200 seconds (7.0 hours).
  - Safety margin remaining at exit: **4,041.0 seconds** (**67.35 minutes**).
- **Host Resource Utilization:**
  - Peak Process RSS: **1,084.7 MB** (~1.08 GB).
  - Minimum Available System RAM: **4,741 MB** (well above 2,560 MB threshold).
- **Routing Selections:** Exactly **2,280 routing events** ($5 \times 456\text{ selections}$).
- **Cache Operation Events:**
  - Total Cache Events: **10,342 events**
  - Cache Hits: **4,635**
  - Cache Misses: **2,205**
  - Cache Inserts: **2,205**
  - Cache Evictions: **1,297**
  - Peak Resident Footprint: **79,976,988,672 bytes** (74.48 GiB / ~79.98 GB, exactly 908 resident experts).
  - Total Data Loaded from DDR5: **194,217,246,720 bytes** (~194.22 GB / 180.88 GiB).
- **Hardware Simulated Timings:**
  - Total Simulated Device Latency: **8,713.53 ms** (mean per-token latency: 1,742.70 ms).
  - Simulated DDR5 Transfer Time: **8,601.93 ms** (98.72% of device latency).
  - Simulated Internal Forward Time: **111.60 ms** (1.28% of device latency).
  - Analytical Roofline Compute: **0.6775 ms** ($677.54\ \mu\text{s}$, or $135.51\ \mu\text{s/token}$).
  - Memory-Wall Disparity Ratio: **$\approx 12,696\times$**.

---

## 3. Directory Layout & Key Artifacts

```
full_run_5step_20261011/
├── configs/
│   └── deepseekv3_b1_full_run_5step.yaml       # Run YAML configuration snapshot
├── config_snapshot/
│   ├── dram_config_DDR5_retry.yaml             # Ramulator DDR5 configuration
│   ├── dram_config_HBM3_80GB.yaml              # Ramulator HBM3 configuration
│   └── checksums.sha256                        # Configuration checksums
├── logs/
│   ├── run.log                                 # Complete execution console log
│   ├── completion_status.txt                   # Machine-readable status summary
│   └── preflight_summary.txt                   # Preflight launch verification record
├── raw/
│   └── completion_status.txt                   # Status copy
├── traces/
│   ├── expert_routing.csv                      # Detailed routing decisions (2,281 lines)
│   ├── expert_load.csv                         # DRAM load telemetry (18,091 lines)
│   └── expert_cache.csv                        # LRU cache events (10,343 lines)
├── csv/
│   └── deepseekV3_synthesis_..._ramul_decode.csv # Canonical simulator CSV (5 rows)
├── processed/
│   ├── summary_metrics.json                    # Reconciled metrics and breakdowns
│   ├── summary_metrics.csv                     # Flattened summary table
│   ├── advanced_analysis.json                  # Multi-token sequential analysis
│   ├── integrity_checks.json                   # Automated invariant check (PASS)
│   ├── per_token_metrics.csv                   # Per-token selection counts
│   ├── per_layer_loads.csv                     # Per-layer DDR5 transfer times
│   ├── per_layer_routing.csv                   # Per-layer routing statistics
│   ├── per_expert_metrics.csv                  # Per-expert access statistics
│   ├── expert_frequency.csv                    # Routed expert selection frequencies
│   ├── cache_event_counts.csv                  # Action event breakdown (HIT/MISS/INSERT/EVICT)
│   └── temporal_routing_jaccard.csv            # Sequential transition Jaccard similarities
├── figures/
│   ├── fig1_temporal_routing_correlation.{png,pdf} # Temporal routing correlation
│   ├── fig2_multi_token_cache_occupancy.{png,pdf}  # Multi-token cache occupancy
│   ├── fig3_lru_eviction_distribution.{png,pdf}    # Cache event composition
│   ├── fig4_per_token_ddr5_latency.{png,pdf}       # Per-layer DDR5 latency
│   └── fig5_compute_vs_memory_overlap.{png,pdf}    # Reconciled timing disparity
├── manifests/
│   ├── preflight_audit.md                      # Preflight launch gate audit
│   └── full_run_manifest.md                    # Complete provenance & artifact inventory
├── report/
│   └── full_run_results.md                     # Comprehensive results report
└── README.md                                   # This document
```

---

## 4. Post-Run Validation Verification

All post-run validation checks passed:
1. `completion_status.txt`: `STATUS=SUCCESS`, `EXIT_CODE=0`.
2. Trace rows: exactly 2,280 routing events + header, 18,090 load rows + header, 10,342 cache rows + header.
3. Automated integrity checks: `integrity_checks.json` returned `status: PASS`.
4. Residency invariants: Max resident bytes strictly capped at 79,976,988,672 bytes ($\le 80\text{ GB}$).
5. Monotonically advancing tokens: verified tokens `[2048, 2049, 2050, 2051, 2052]`.
