# DS-5C-10 Final Preflight: Model Fidelity and Full-Run Authorization

**Repository:** `/home/mohman/ca_project/LLMSimulator`  
**Execution Context:** Final Full-Workload Preflight Checkpoint  
**Target Executable:** `build/test_deepseekv3_full_baseline`  
**Planned Output Directory:** `reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/`  
**Date:** October 10, 2026  
**Final Decision:** **GO (PREFLIGHT PASSED — AWAITING USER LAUNCH AUTHORIZATION)**  

---

## 1. Intended Model Configuration & Fidelity Reconciliation

### 1.1 Experiment Classification
The experiment is formally classified as:
**Option 1: A DeepSeek-V3-shaped, FP16, synthetic-routing memory-system study.**

It evaluates the architectural memory dynamics (off-chip DDR5 batch transfers, queuing, and HBM LRU residency) of a full 60-layer transformer matching DeepSeek-V3 tensor dimensions and routing topology. It is **not** an end-to-end token inference run of the official weights checkpoint.

### 1.2 Authoritative Architectural Parameter Values:
- **Total Layers:** 60
- **Dense Layers:** 3 (Layers 0, 1, 2)
- **MoE Layers:** 57 (Layers 3 through 59)
- **Routed Experts per MoE Layer:** 256
- **Top-K Routing:** 8 activated experts per token
- **Shared Experts:** 1 (modeled permanently resident in HBM)
- **Weight Precision:** FP16 (2 bytes per weight)
- **Expert Architecture:** 3-way SwiGLU FFN ($W_1$ `gate_proj`, $W_3$ `up_proj`, $W_2$ `down_proj`)
- **Dimensions:** HiddenDim = 7168, IntermediateDim = 2048
- **Single Expert Weight Footprint:** $3 \times (7168 \times 2048 \times 2) = \mathbf{88,080,384\text{ bytes}}$ ($\approx 88.08\text{ MB}$)
- **Total Model Routed Footprint:** $57 \times 256 \times 88,080,384\text{ B} = \mathbf{1,285,268,963,328\text{ bytes}}$ ($\approx 1.285\text{ TB}$)
- **Cold Routed Experts per Token:** $57 \times 8 = \mathbf{456\text{ experts}}$
- **Cold Routed Footprint per Token:** $456 \times 88,080,384\text{ B} = \mathbf{40,164,655,104\text{ bytes}}$ ($\approx \mathbf{40.16\text{ GB}}$)
- **HBM Cache Budget:** 80 GB ($80,000,000,000\text{ bytes}$)
- **HBM Cache Occupancy:** $40.164\text{ GB} / 80\text{ GB} = \mathbf{50.21\%}$ (Zero evictions expected)
- **Simulated Hardware DDR5 Latency:** $456 \times 3.901028\text{ ms} = \mathbf{1.7788\text{ seconds}}$ ($1,778,868,800\text{ ns}$)

### 1.3 Deviations from Official DeepSeek-V3 Production Model
1. **Total Layers:** 60 in LLMSimulator vs 61 in Official DeepSeek-V3 (57 MoE layers vs 58 MoE layers).
2. **Precision:** FP16 (2 bytes/weight) in LLMSimulator vs FP8 (1 byte/weight + scaling) in Official DeepSeek-V3.
3. **Routing Provenance:** Synthetic PRNG routing (Zipfian/Random with deterministic token replay) vs token-embedding gate softmax.
4. **Shared Expert:** 1 shared expert resident in HBM; off-chip DDR5 tier serves the 256 routed experts.

---

## 2. Regression Test Gate Status

All 6 required regression test targets have built and passed with exit code 0:

| Regression Target | Command | Exit Code | Verified Scope |
| :--- | :--- | :---: | :--- |
| **Production GPU DDR5 Mapping** | `./build/test_production_ddr5_mapping` | **0** | Verified `getAddrVec` on `ddr5_32gb_x16_2ch`, row boundaries, row bounds, and 0% collisions across 10,000 addresses. |
| **Cache Residency Remediation** | `./build/test_cache_residency_remediation` | **0** | Verified empty activations bypass cache; cold loads mark residency only on completion; warm hits promote MRU; evictions respect capacity. |
| **LRU Cache & Capacity** | `./build/test_lru` | **0** | Verified variable-size eviction, duplicate protection, oversized entry rejection, and 908-expert budget. |
| **Extended DDR5 Mapping** | `./build/test_ddr5_mapping` | **0** | Verified extended DDR5 row indexing up to row 9,125,888. |
| **Minimal DDR5 Timing** | `./build/test_minimal_timing` | **0** | Verified 3-way Ramulator batching and cache bypass ($3.901\text{ ms}$). |
| **Staged Module Graph (Stage 1)** | `./build/test_staged_module_graph` | **0** | Verified 2 MoE layers, exactly 16 resident experts ($1.409\text{ GB}$), row-for-row replay routing identity, and 0 ns replay DDR5 delta. |

---

## 3. Explicit Full-Run Target Build Verification

- **Executable:** `build/test_deepseekv3_full_baseline`
- **Source:** `eval/test_deepseekv3_full_baseline.cpp`
- **Build Command:**
  ```bash
  cmake --build build --target test_deepseekv3_full_baseline -j"$(nproc)"
  ```
- **Exit Code:** `0`
- **Binary Status:** Present, executable, size: 520,368 bytes (Build log saved to `logs/build.log`).

---

## 4. Host Runtime Estimation & Resource Budget

- **Empirical Host Simulation Rate:** $\approx 10.259\text{ seconds per cold expert}$ ($164.14\text{ s} / 16\text{ experts}$ measured in Stage 1).
- **Workload Scale:** 456 cold experts across 57 MoE layers.
- **Estimated Host Runtime:** $456 \times 10.259\text{ s} \approx \mathbf{4,678\text{ seconds}}$ ($\approx \mathbf{78\text{ minutes}}$).
- **Runtime Uncertainty Range:** **72 to 90 minutes** (dependent on DRAM queuing and host CPU scheduling).
- **Configured Hard Timeout:** **7,200 seconds (120 minutes)** $\rightarrow$ **30 to 48 minutes safety headroom**.
- **System Resource Check:**
  - Available System RAM: **6.27 GB** ($\ge 2.5\text{ GB}$ threshold) — **HEALTHY**
  - Available Disk Space: **909 GB** ($\ge 5.0\text{ GB}$ threshold) — **HEALTHY**
  - Lock File Guard: Active in `scripts/run_full_baseline.sh`

---

## 5. Experiment Directory Preparation

The dedicated experiment root is fully prepared and populated:
`reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/`

- `README.md`: Complete reproduction and architectural documentation.
- `config_snapshot/`: Frozen `dram_config_DDR5_retry.yaml` and `dram_config_HBM3_80GB.yaml` with verified sha256 checksums.
- `manifests/`:
  - `source_change_provenance.md`: Full diffs of `src/module/linear.cpp`, `src/hardware/linear_impl.cpp`, `src/module/expert.cpp`, `src/scheduler/sequence.cpp`.
  - `raw_telemetry_schema.md`: Detailed schema definitions for routing, load, and cache CSVs.
  - `completion_status_schema.md`: Specification of machine-readable status file.
- `scripts/`:
  - `run_full_baseline.sh`: Self-contained execution script with 7,200s watchdog, RAM/disk guards, and 60s progress logging.
  - `process_metrics.py`: Normalized telemetry analysis script.
- `logs/`: Populated with `build.log`.

---

## 6. Known Limitations & Scope Constraints

1. **Synthetic Routing:** Expert selections use PRNG-generated routing (Zipfian/Random) over a single decode token. Results measure hardware memory queuing and bandwidth, not natural language prompt clustering.
2. **Zero-Modeled HBM Latency:** GPU on-chip HBM accesses are modeled as 0 ns in this configuration to isolate off-chip DDR5 memory overheads.
3. **No Evictions in Baseline:** Because the 40.16 GB cold working set fits within the 80 GB HBM budget (50.21% occupancy), this baseline evaluates cold loading without inter-token LRU cache thrashing.

---

## 7. Preflight Decision

**DECISION: GO (ALL PREFLIGHT GATES SATISFIED)**

The model configuration is reconciled, all regressions pass, the full-run binary is compiled and verified, resources are ample, and the experiment framework is in place.

**Awaiting user authorization to initiate the full-model baseline execution.**
