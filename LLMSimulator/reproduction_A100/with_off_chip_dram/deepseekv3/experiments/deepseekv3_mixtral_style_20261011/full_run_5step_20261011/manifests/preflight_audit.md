# DS-6C-4 Preflight Audit & Launch Gate Manifest

**Date:** 2026-10-11  
**Timestamp (UTC):** 2026-10-11T09:31:30Z  
**Repository:** `/home/mohman/ca_project/LLMSimulator`  
**Git Commit:** `f75d362ae157f81ef2d3bd4055c9b22e34882a9f`  
**Harness Executable:** `build/run_deepseekv3_mixtral_style` (SHA-256: `819a752527b536224b1467a297d72158281da5c7d5e54e7f792a0d521c4e1093`)  
**Run Directory:** `reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/`  
**Effective Configuration:** `reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/configs/deepseekv3_b1_full_run_5step.yaml` (SHA-256: `1bdcaee0f058c448a69f94e256998dc44da6acca5fdbd01b74db41b3a3954f6d`)  

---

## 1. Workload Specifications

- **Model Architecture:** DeepSeek-V3 (60 layers: 3 dense + 57 MoE layers).
- **Expert Configuration:** 256 routed experts per MoE layer, top-$k = 8$, 1 shared expert.
- **Precision:** FP16 accounting (`precision_byte: 2`), expert footprint = 88,080,384 bytes (~88.08 MB).
- **Total Routed Experts:** 14,592 experts = 1,285,268,963,328 bytes (~1.285 TB).
- **Backing Memory:** Off-chip DDR5 (2.75 TB / 2.5 TiB across 10,485,760 rows, row utilization 46.76%).
- **Cache Configuration:** On-chip HBM3 80,000,000,000 bytes (~74.51 GiB / 80 GB) LRU cache.
- **Serving / Sequence:**
  - Batch size: 1 (`max_batch_size: 1`).
  - Sequence: `input_len: 2048`, `output_len: 10`.
  - Sequential decoding iterations: 5 decoding steps (`iter: 5`).
- **Gantt / Visualization:** Disabled (`export_gantt: false`).

---

## 2. Source & Checksum Provenance

| Artifact | Path | SHA-256 Checksum |
|:---|:---|:---|
| **Executable** | `build/run_deepseekv3_mixtral_style` | `819a752527b536224b1467a297d72158281da5c7d5e54e7f792a0d521c4e1093` |
| **Run Config** | `configs/deepseekv3_b1_full_run_5step.yaml` | `1bdcaee0f058c448a69f94e256998dc44da6acca5fdbd01b74db41b3a3954f6d` |
| **DDR5 Memory Config** | `full_run_5step_20261011/config_snapshot/dram_config_DDR5_retry.yaml` | `7ef030f82d7f654339aa0ab1ff31abce311322c59aee693eb57774e071db9d97` |
| **HBM3 Memory Config** | `full_run_5step_20261011/config_snapshot/dram_config_HBM3_80GB.yaml` | `e7ae1fb6802016fba28dd8d2cac439a15c01d2615c2063973eacb628903e79c6` |
| **Git Revision** | Commit HEAD | `f75d362ae157f81ef2d3bd4055c9b22e34882a9f` |

---

## 3. Preflight System Health & Invariant Checks

1. **Available RAM:** 6,401 MB ($\ge 2,560\text{ MB}$ safety threshold) $\rightarrow$ **PASS**.
2. **Available Disk:** 909 GB ($\ge 5\text{ GB}$ safety threshold) $\rightarrow$ **PASS**.
3. **No Active Simulation Locks:** No active `*.lock` files $\rightarrow$ **PASS**.
4. **Configuration Validation:** `./build/run_deepseekv3_mixtral_style ... --validate-config` exited with code 0 $\rightarrow$ **PASS**.
5. **DDR5 Mapping Invariant:** Validated via `test_production_ddr5_mapping` (exit 0) $\rightarrow$ **PASS**.
6. **LRU Cache Unit Test:** Validated via `test_lru` (exit 0) $\rightarrow$ **PASS**.
7. **Figure 5 Dynamic Generation:** Validated via `test_figure5_dynamic.py` (exit 0) $\rightarrow$ **PASS**.

---

## 4. Runtime Estimation & Launch Gate Status

- **Estimated Host Runtime:** ~22,500 seconds (6.25 hours).
- **Realistic Completion Range:** 21,000 to 22,600 seconds (5.83 to 6.28 hours).
- **Expected Completion Timestamp (UTC):** ~15:45 UTC (range: 15:20 to 15:47 UTC).
- **Watchdog Setting:** `WATCHDOG_TIMEOUT=25200` (7.0 hours = 25,200 seconds).
- **Safety Headroom:** $25,200\text{ s} - 22,500\text{ s} = 2,700\text{ s} = \mathbf{45.0\text{ minutes}}$ ($\ge 45\text{ min}$ requirement strictly met).
- **Launch Gate Evaluation:** **GO / AUTHORIZED TO LAUNCH**.
