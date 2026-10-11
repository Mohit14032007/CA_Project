# DeepSeek-V3 Full-Model Memory-System Baseline

**Directory:** `reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/`  
**Simulator:** LLMSimulator (Cycle-accurate GPU & Ramulator 2 DRAM subsystem)  
**Host Target:** Single NVIDIA A100 GPU (80 GB HBM3) paired with Secondary Off-Chip DDR5 DRAM  
**Execution Date:** October 10, 2026  
**Status:** COMPLETED (SUCCESS — Exit Code 0)  

---

## 1. Objective and Architectural Scope

This experiment executes the authoritative full-model memory-system baseline for the DeepSeek-V3 architecture in LLMSimulator. The primary objective is to evaluate memory tiering dynamics, off-chip DDR5 batch loading latencies, and HBM LRU cache residency under a full 60-layer MoE transformer workload.

### Model Architecture & Experiment Classification

> [!IMPORTANT]
> **Workload Provenance & Architectural Classification:**  
> This experiment is explicitly classified as a **DeepSeek-V3-shaped, FP16, synthetic-routing memory-system simulation**, designed to quantify hardware memory traffic and queuing under realistic scale. It is **not** an end-to-end token inference run of the official weights checkpoint.

### Architectural Parameters & Reconciliation:

| Architectural Parameter | LLMSimulator Baseline Value | Official DeepSeek-V3 Checkpoint | Reconciliation & Provenance Rationale |
| :--- | :--- | :--- | :--- |
| **Total Layers** | **60** | 61 | Configured in `src/model/model_config.h` (`num_layers = 60`). |
| **Dense Layers** | **3** (Layers 0–2) | 3 | Preserved from `first_k_dense = 3`. |
| **MoE Layers** | **57** (Layers 3–59) | 58 | Exactly $60 - 3 = 57$ MoE layers. |
| **Routed Experts / Layer** | **256** | 256 | Exact architectural match (`num_routed_expert = 256`). |
| **Top-K Routing** | **8** | 8 | Top-8 routed experts activated per token. |
| **Shared Experts** | **1** | 1 | Shared expert is permanently resident in HBM; routed off-chip tier handles the 256 routed experts. |
| **Weight Precision** | **FP16 (2 Bytes/weight)** | FP8 (1 Byte/weight) | Standard precision modeled throughout LLMSimulator memory pipeline. |
| **Expert FFN Structure** | **3-way SwiGLU** ($W_1, W_3, W_2$) | 3-way SwiGLU | Three matrices: Gate ($W_1$), Up ($W_3$), Down ($W_2$). |
| **Hidden / Interm. Dim** | Hidden: 7168, Interm: 2048 | Hidden: 7168, Interm: 2048 | Shape $[7168, 2048]$ for $W_1, W_3$; $[2048, 7168]$ for $W_2$. |
| **Single Expert Footprint** | **88,080,384 Bytes ($\approx 88.08$ MB)** | $\approx 44.04$ MB (FP8) | $3 \times (7168 \times 2048 \times 2\text{ B}) = 88,080,384\text{ B}$. |
| **Cold Experts per Token** | **456 routed experts** | 464 routed experts | $57\text{ MoE layers} \times 8\text{ top-k} = \mathbf{456\text{ experts}}$ (Pass 1). |
| **Cold Weight Footprint** | **40,164,655,104 Bytes ($\approx 40.16$ GB)** | $\approx 20.43$ GB | $456 \times 88,080,384\text{ B} = \mathbf{40.16\text{ GB}}$. |
| **HBM Cache Capacity** | **80,000,000,000 Bytes (80 GB)** | 80 GB | Cold footprint consumes **50.21%** of HBM cache. Zero evictions expected for single token. |
| **Total Model Footprint** | **1,285,268,963,328 Bytes ($\approx 1.285$ TB)** | $\approx 653.9$ GB (FP8) | 14,592 total routed experts across 57 layers fit within the 2.75 TB DDR5 address space. |

---

## 2. Execution Run Metadata

- **Run ID:** `deepseekv3_full_baseline_20261010`
- **Execution Command:** `./reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/scripts/run_full_baseline.sh`
- **Source Revision:** `f75d362ae157f81ef2d3bd4055c9b22e34882a9f`
- **Executable:** `build/test_deepseekv3_full_baseline`
- **Active Configurations:**
  - DDR5 Config: `config_snapshot/dram_config_DDR5_retry.yaml` (TraceRecorder disabled)
  - HBM Config: `config_snapshot/dram_config_HBM3_80GB.yaml`
- **Start Time (UTC):** `2026-10-10T14:28:11Z`
- **End Time (UTC):** `2026-10-10T15:42:10Z`
- **Total Host Runtime:** `4,439 seconds` ($\approx 73.98$ minutes)
  - Pass 1 (Cold Start): $4,431.96\text{ s}$
  - Pass 2 (Warm Replay): $3.25\text{ s}$
- **Simulated Hardware Time (Pass 1 DDR5):** $1,778,905,300\text{ ns}$ ($\approx 1.7789\text{ seconds}$)
- **Pass 2 Clock Advances:** DDR5 Delta = $0\text{ ns}$, HBM Delta = $0\text{ ns}$ (100% cache hits)
- **Exit Code:** `0`
- **Completion Status:** `SUCCESS` (Recorded in `logs/ds5c11_completion_status.txt` and `raw/ds5c10_completion_status.txt`)

### Resource & Environment Observations:
- **Available System RAM at Start:** $9,079\text{ MB}$
- **Minimum Available RAM Observed:** $5,132\text{ MB}$ (remained safely above the $2.5\text{ GB}$ / $2,560\text{ MB}$ guard)
- **Peak Process RSS:** $1,082,875,904\text{ bytes}$ ($\approx 1.08\text{ GB}$)
- **Disk Space Available:** $929\text{ GB}$ (Free space unaffected; run consumed $< 10\text{ MB}$)
- **Watchdog Timeout Limit:** $7,200\text{ seconds}$ (Run finished with $\approx 46$ minutes of headroom)

---

## 3. Raw Telemetry Artifacts

All raw simulation telemetry is preserved in `raw/`:

| Raw File | Row Count | File Size | SHA256 Checksum | Description |
| :--- | :--- | :--- | :--- | :--- |
| [`raw/expert_routing.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/raw/expert_routing.csv) | 913 lines | 11,355 bytes | `229fc7522e212f356d2846d79a72b6172b395f3ca9a8f194de8187d0a696e259` | 912 routing selections (456 Pass 1 + 456 Pass 2 replay) + 1 header |
| [`raw/expert_load.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/raw/expert_load.csv) | 6,385 lines | 464,179 bytes | `d984dd17a3c412bcbf97c7a0c44bab185bb7f70b4ba06933bd259fa178d28644` | Per-tensor and batch-level transfer records |
| [`raw/expert_cache.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/raw/expert_cache.csv) | 3,193 lines | 72,902 bytes | `3302371500fe0e5cb79c23d4b9e86cce1b107b25482ed042e6b4bdb73d02d181` | HBM cache MISS, INSERT, and HIT transitions |
| [`raw/ds5c10_completion_status.txt`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/raw/ds5c10_completion_status.txt) | 23 lines | 555 bytes | `f6e2ff8b133858b9ecaf50c89d33c689429f2a3539146f6b647ffb275b65e859` | C++ simulator hardware timers and line counters |
| [`logs/ds5c11_completion_status.txt`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/logs/ds5c11_completion_status.txt) | 22 lines | 1,708 bytes | `6de4d7b226d2cade755555e27b452fbbcbeb7aa9f8ba52d6da7971f1d45f5ae0` | Runner execution metadata and system metrics |

### Interpretation Metadata:
1. **`expert_routing.csv`:** Contains the token/layer/expert routing assignments. Pass 1 contains 456 decisions ($57\text{ layers} \times 8\text{ top-k}$); Pass 2 replayed the identical 456 decisions deterministically via `cached_routing`.
2. **`expert_load.csv`:** Contains individual tensor rows (`W1_gate`, `W3_up`, `W2_down`) and aggregate rows (`all_weights`). A cold miss issues a single off-chip batch of 88.08 MB for the 3 weights. Subsequent sister executions and warm replay accesses are recorded as HBM hits.
3. **`expert_cache.csv`:** Records every cache state transition. At the conclusion of Pass 1, exactly 456 experts are resident, with a total resident size of $40,164,655,104\text{ bytes}$ ($\approx 40.16\text{ GB}$). No evictions occurred.

---

## 4. Directory Structure

```
deepseekv3_full_baseline_20261010/
├── README.md                           # This experiment guide and execution documentation
├── config_snapshot/                   # Frozen hardware & memory configuration files
│   ├── checksums.sha256                # Cryptographic verification hashes
│   ├── dram_config_DDR5_retry.yaml     # Production DDR5 Ramulator 2 configuration (TraceRecorder disabled)
│   └── dram_config_HBM3_80GB.yaml      # Production HBM3 Ramulator 2 configuration
├── manifests/                          # Audit records, validation reports, and schemas
│   ├── ds5c12_analysis_validation.md   # Final DS-5C-12 10-gate validation audit manifest
│   ├── input_integrity_record.md       # Pre/post-run telemetry checksum audit
│   ├── source_change_provenance.md     # Git diff and explanation of all C++ changes
│   ├── raw_telemetry_schema.md         # Schema descriptions for routing, load, and cache logs
│   ├── completion_status_schema.md     # Specification of machine-readable status file
│   └── go_nogo_checklist.md            # Preflight checklist and gate decisions
├── logs/                               # Execution logs
│   ├── build.log                       # Target compilation output
│   ├── ds5c11_completion_status.txt    # Runner execution metadata
│   └── full_baseline_run.log           # Full stdout/stderr and progress telemetry
├── raw/                                # Raw unmodified simulation telemetry
│   ├── expert_routing.csv              # Token-to-expert routing assignments (913 lines)
│   ├── expert_load.csv                 # Detailed per-tensor and batch DDR5/HBM transfers (6,385 lines)
│   ├── expert_cache.csv                # Exact HBM cache MISS, INSERT, HIT timeline (3,193 lines)
│   ├── ds5c10_completion_status.txt    # C++ harness hardware timing status
│   └── ds5c11_completion_status.txt    # Copy of runner execution metadata
├── processed/                          # Validated processed datasets and metrics
│   ├── summary_metrics.json            # Machine-readable comprehensive summary metrics
│   ├── summary_metrics.csv             # Tabular flattened summary metrics
│   ├── per_layer_metrics.csv           # Layer-by-layer routing and DDR5 latency aggregations
│   ├── per_expert_metrics.csv          # Per-expert cold load sequence and duration table
│   ├── routing_counts.csv              # Expert selection frequencies across passes
│   ├── cache_event_counts.csv          # Discrete counts of cache actions (MISS, INSERT, HIT, EVICT)
│   ├── load_duration_metrics.csv       # Percentiles and latency distributions of off-chip batches
│   └── integrity_checks.json           # Programmatic assertion checks (all 8 passed)
├── figures/                            # Publication-quality figures (300 DPI PNG and vector PDF)
│   ├── README.md                       # Comprehensive figure index and description guide
│   ├── fig1_routing_coverage.png / .pdf
│   ├── fig2_expert_selection_heatmap.png / .pdf
│   ├── fig3_cold_loaded_experts_per_layer.png / .pdf
│   ├── fig4_offchip_bytes_per_layer.png / .pdf
│   ├── fig5_cache_event_composition.png / .pdf
│   ├── fig6_cache_occupancy_evolution.png / .pdf
│   ├── fig7_ddr5_batch_duration_distribution.png / .pdf
│   ├── fig8_per_layer_simulated_ddr5_time.png / .pdf
│   ├── fig9_host_vs_simulated_time.png / .pdf
│   └── fig10_expert_load_timeline.png / .pdf
├── report/                             # Final research report
│   └── results.md                      # Comprehensive academic baseline research report
└── scripts/                            # Self-contained reproduction scripts
    ├── run_full_baseline.sh            # Safe runner script (7,200s watchdog + RAM/disk guards)
    ├── analyze_full_baseline.py        # Post-run metrics processing and aggregation script
    ├── generate_graphs.py              # Publication-quality figure generation script
    └── process_metrics.py              # Schema-normalized parser utility
```

---

## 5. Processed Datasets

All processed metrics are derived programmatically from `raw/` telemetry without modifying raw records:

| Dataset File | Description | Records | Key Columns / Summary |
| :--- | :--- | :---: | :--- |
| [`processed/summary_metrics.json`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/processed/summary_metrics.json) | Complete baseline metrics in JSON | 1 doc | Architecture, host timing, simulated DDR5 timing, bandwidth, row counts |
| [`processed/summary_metrics.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/processed/summary_metrics.csv) | Flattened tabular summary metrics | 30 rows | Key-value pairs across categories |
| [`processed/per_layer_metrics.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/processed/per_layer_metrics.csv) | Metrics for all 57 MoE layers | 57 rows | `model_layer_index`, `offchip_batches`, `offchip_bytes`, `simulated_ddr5_duration_ms` |
| [`processed/per_expert_metrics.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/processed/per_expert_metrics.csv) | Sequence of 456 cold-loaded experts | 456 rows | `sequence_order`, `moe_layer_id`, `expert_id`, `offchip_bytes`, `simulated_duration_ms` |
| [`processed/routing_counts.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/processed/routing_counts.csv) | Expert activation frequencies | 256 rows | `expert_id`, `selection_count_pass1`, `selection_count_pass2`, `total_selections` |
| [`processed/cache_event_counts.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/processed/cache_event_counts.csv) | Breakdown of cache actions | 4 rows | `action`, `count` (456 MISS, 456 INSERT, 2280 HIT, 0 EVICT) |
| [`processed/load_duration_metrics.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/processed/load_duration_metrics.csv) | Statistical distribution of batch latency | 15 rows | `mean_ms` (3.9011 ms), `min_ns`, `p50_median_ns`, `max_ns`, `effective_bandwidth_gb_per_sec` (22.58 GB/s) |
| [`processed/integrity_checks.json`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/processed/integrity_checks.json) | Assertion verification flags | 9 keys | All 8 assertions `true`, `overall_integrity_passed: true` |

---

## 6. Publication Figures

Ten high-resolution (300 DPI) publication figures and vector PDFs are saved in `figures/`:

1. **Figure 1: Routing Coverage by Layer** ([`PNG`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/figures/fig1_routing_coverage.png) / [`PDF`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/figures/fig1_routing_coverage.pdf)) — Confirms exact top-8 selection across 57 MoE layers in Pass 1 and Pass 2.
2. **Figure 2: Expert Selection Heatmap** ([`PNG`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/figures/fig2_expert_selection_heatmap.png) / [`PDF`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/figures/fig2_expert_selection_heatmap.pdf)) — $57 \times 256$ routing matrix visualization for synthetic PRNG routing.
3. **Figure 3: Cold-Loaded Experts Per Layer** ([`PNG`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/figures/fig3_cold_loaded_experts_per_layer.png) / [`PDF`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/figures/fig3_cold_loaded_experts_per_layer.pdf)) — Shows exactly 8 off-chip batch loads per layer (total = 456).
4. **Figure 4: Off-Chip Bytes Per Layer** ([`PNG`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/figures/fig4_offchip_bytes_per_layer.png) / [`PDF`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/figures/fig4_offchip_bytes_per_layer.pdf)) — Quantifies 704.64 MB off-chip volume transferred per layer (total = 40.16 GB).
5. **Figure 5: Cache Event Composition** ([`PNG`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/figures/fig5_cache_event_composition.png) / [`PDF`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/figures/fig5_cache_event_composition.pdf)) — 456 MISS, 456 INSERT, 2280 HIT, 0 EVICT.
6. **Figure 6: Cache Occupancy Evolution** ([`PNG`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/figures/fig6_cache_occupancy_evolution.png) / [`PDF`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/figures/fig6_cache_occupancy_evolution.pdf)) — Occupancy rise to 40.16 GB during Pass 1 and constant plateau during Pass 2 against 80 GB limit.
7. **Figure 7: DDR5 Batch Duration Distribution** ([`PNG`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/figures/fig7_ddr5_batch_duration_distribution.png) / [`PDF`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/figures/fig7_ddr5_batch_duration_distribution.pdf)) — Latency distribution ($3.9011 \pm 0.0003\text{ ms}$).
8. **Figure 8: Per-Layer Simulated DDR5 Time** ([`PNG`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/figures/fig8_per_layer_simulated_ddr5_time.png) / [`PDF`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/figures/fig8_per_layer_simulated_ddr5_time.pdf)) — Uniform ~31.21 ms simulated hardware DDR5 transfer time per layer.
9. **Figure 9: Host Time vs. Simulated Memory Time** ([`PNG`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/figures/fig9_host_vs_simulated_time.png) / [`PDF`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/figures/fig9_host_vs_simulated_time.pdf)) — Host CPU time ($4,439\text{ s}$) vs. simulated DRAM time ($1.7789\text{ s}$) demonstrating $\approx 2,491\times$ slowdown.
10. **Figure 10: Expert-Load Timeline** ([`PNG`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/figures/fig10_expert_load_timeline.png) / [`PDF`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/figures/fig10_expert_load_timeline.pdf)) — Cumulative simulated hardware DRAM time progression over the 456 cold batches.

Detailed index available in [`figures/README.md`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/figures/README.md).

---

## 7. Results Audit & Validation Summary

| Audit Item | Expected Value | Observed & Verified Value | Status |
| :--- | :--- | :--- | :---: |
| **Pass 1 Routing Decisions** | 456 ($57 \times 8$) | Exactly 456 decisions | **VERIFIED** |
| **Pass 2 Routing Decisions** | 456 (cached replay) | Exactly 456 decisions (row-by-row identical) | **VERIFIED** |
| **Cold Expert Batches (Pass 1)** | 456 | Exactly 456 DDR5 batches ($88.08\text{ MB}$ each) | **VERIFIED** |
| **Warm Replay Batches (Pass 2)** | 0 off-chip | Exactly 0 DDR5 batches ($100\%$ HBM hits) | **VERIFIED** |
| **Cache Events Total** | 3,192 | 456 MISS + 456 INSERT + 2280 HIT + 0 EVICT | **VERIFIED** |
| **Total Cold Data Volume** | 40,164,655,104 Bytes | $40,164,655,104\text{ Bytes}$ ($40.1647\text{ GB}$) | **VERIFIED** |
| **Final HBM Cache Occupancy** | 50.21% (of 80 GB) | $50.2058\%$ (Zero evictions) | **VERIFIED** |
| **Simulated Hardware DDR5 Time** | $\approx 1.7789\text{ s}$ | $1,778,905,340\text{ ns}$ ($1.778905\text{ s}$) | **VERIFIED** |
| **Host CPU Wall-Clock Time** | $4,439\text{ s}$ | $4,431.96\text{ s}$ (Pass 1) + $3.25\text{ s}$ (Pass 2) | **VERIFIED** |
| **Raw Telemetry Checksums** | Bit-for-bit unchanged | All SHA-256 hashes match original DS-5C-11 records | **VERIFIED** |

Full validation gate details documented in [`manifests/ds5c12_analysis_validation.md`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/manifests/ds5c12_analysis_validation.md).

---

## 8. Analysis Reproduction Instructions

To reproduce the analysis and regenerate all figures from the raw telemetry:

```bash
# 1. Run post-processing analysis and generate processed datasets:
python3 reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/scripts/analyze_full_baseline.py \
    reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010

# 2. Generate all 10 publication figures (PNG and PDF):
python3 reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/scripts/generate_graphs.py \
    reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010
```

---

## 9. Limitations & Scope Constraints

1. **Synthetic Routing:** PRNG Zipfian/uniform routing evaluates hardware scaling and bus contention; it does not evaluate input prompt text semantics or multi-token KV cache interactions.
2. **Ideal Interconnect / On-Chip Latency:** HBM hits are modeled as zero nanoseconds in this tier to isolate secondary DDR5 DRAM bus transfer dynamics.
3. **Capacity & Eviction Regimes:** Since the 40.16 GB cold working set fits within the 80 GB HBM budget (50.21% occupancy), this baseline evaluates cold loading without inter-token LRU cache thrashing.

