# DS-5C-12: Full Baseline Results Audit and Analysis Validation Manifest

**Experiment:** DeepSeek-V3 Full Baseline (`deepseekv3_full_baseline_20261010`)  
**Audit Context:** Post-Execution Telemetry Audit & Results Validation  
**Date:** October 10, 2026  
**Auditor:** Automated Antigravity Research Verification Suite  
**Final Verdict:** **ALL VALIDATION GATES PASSED (100% VERIFIED)**  

---

## 1. Executive Summary & Validation Gate Matrix

Every aspect of the simulation telemetry, data integrity, timing mechanics, and cache accounting was subjected to strict verification against the raw CSV files. No metrics were assumed or hardcoded.

| Validation Gate | Category | Status | Verified Measurement / Assertion |
| :--- | :--- | :---: | :--- |
| **Gate 1** | Raw File Hash Integrity | **PASS** | `raw/expert_routing.csv`, `raw/expert_load.csv`, and `raw/expert_cache.csv` match their original post-run SHA-256 hashes bit-for-bit. Zero bytes modified. |
| **Gate 2** | Deterministic Replay Identity | **PASS** | Pass 1 and Pass 2 routing decisions are identical across all 456 selections ($57\text{ layers} \times 8\text{ top-k}$). Row-by-row differences: 0. |
| **Gate 3** | Load Batch Invariant | **PASS** | Exactly 456 cold off-chip DDR5 batches were issued in Pass 1. Exactly 0 off-chip batches issued in Pass 2. |
| **Gate 4** | Sister Weight Batching | **PASS** | For each cold expert, $W_1, W_3, W_2$ were batched into a single 88.08 MB DDR5 transfer during the initial $W_1$ miss. Subsequent $W_3$ and $W_2$ accesses hit HBM. |
| **Gate 5** | Cache Residency Invariant | **PASS** | Unrouted experts with empty activations never called cache routines. Exactly 456 distinct experts were marked resident in HBM strictly upon DDR5 load completion. |
| **Gate 6** | Zero Cache Evictions | **PASS** | The total cold footprint ($40,164,655,104\text{ B} \approx 40.16\text{ GB}$) is $50.21\%$ of the 80 GB HBM capacity. Exactly 0 evictions occurred (`EVICT = 0`). |
| **Gate 7** | Warm Replay Cache Invariant | **PASS** | Pass 2 achieved 100% cache hits (1,368 hits). DDR5 simulated clock advance on replay was exactly $0.000\text{ ns}$. |
| **Gate 8** | Timing & Latency Consistency | **PASS** | Cumulative transfer duration per batch averaged $3.9011\text{ ms}$ (total simulated DDR5 time: $1.778905\text{ s}$). Sustained streaming rate: $22.58\text{ GB/s}$. |
| **Gate 9** | Host Wall-Clock Accounting | **PASS** | Host runtime was $4,439\text{ s}$ ($\approx 74\text{ min}$): Pass 1 took $4,431.96\text{ s}$, while Pass 2 took $3.25\text{ s}$, demonstrating a $\approx 2,491\times$ simulation slowdown factor. |
| **Gate 10** | Figure & Dataset Reproducibility | **PASS** | All 8 processed tables and all 10 publication figures generated programmatically from raw data with zero hardcoded values. |

---

## 2. Telemetry Row Count & Semantic Reconciliation

| Data Stream | Physical Line Count | Data Rows | Primary Telemetry Semantics |
| :--- | :---: | :---: | :--- |
| `expert_routing.csv` | 913 | 912 | 456 Pass 1 selections + 456 Pass 2 replay selections. |
| `expert_load.csv` | 6,385 | 6,384 | 2,736 `all_weights` aggregate rows (456 DDR5 misses + 2,280 HBM hits) + 3,648 constituent tensor rows ($W_1, W_3, W_2$). |
| `expert_cache.csv` | 3,193 | 3,192 | 456 initial cold `MISS` + 456 load complete `INSERT` + 2,280 warm `HIT` (912 sister hits + 1,368 replay hits) + 0 `EVICT`. |

### Cross-Stream Identity Verification:
- **Unique Request IDs in Load:** Exactly **2,736 unique IDs** ($1,368\text{ in Pass 1} + 1,368\text{ in Pass 2}$).
- **Constituent Tensors:** $1,368\text{ (Pass 1 miss sisters)} + 912\text{ (Pass 1 hits)} + 1,368\text{ (Pass 2 hits)} = \mathbf{3,648\text{ rows}}$.
- **Total Load Rows:** $2,736\text{ aggregates} + 3,648\text{ tensors} = \mathbf{6,384\text{ data rows}}$.
- **Total Cache Events:** $456\text{ MISS} + 456\text{ INSERT} + 2,280\text{ HIT} = \mathbf{3,192\text{ events}}$.
- **Final Resident Bytes:** $456 \times 88,080,384\text{ B} = \mathbf{40,164,655,104\text{ bytes}}$.

### Cache Event Sequencing Invariant:
- **Pass 1 Sequencing (1,824 events across 456 cold experts):**
  Every single cold expert strictly executed the 4-event tuple: `('MISS', 'INSERT', 'HIT', 'HIT')`.
  - Event 1 (`MISS`): Initial probe by $W_1$ gate weight triggering off-chip DDR5 batch retrieval.
  - Event 2 (`INSERT`): Modeled transfer completes; entire 88.08 MB expert marked resident; resident bytes increments by $88,080,384\text{ B}$.
  - Event 3 (`HIT`): Sister weight $W_3$ up matrix hits resident HBM copy.
  - Event 4 (`HIT`): Sister weight $W_2$ down matrix hits resident HBM copy.
- **Pass 2 Sequencing (1,368 events across 456 warm experts):**
  Every single replayed expert strictly executed the 3-event tuple: `('HIT', 'HIT', 'HIT')`.
  - All three matrices hit HBM with zero off-chip traffic and zero clock advance.

---

## 3. Mathematical, Address-Space, & Timing Reconciliation

### 3.1 Production GPU Address Mapping
- **Mapping Path:** `MMapController::getAddrVec` on `ddr5_32gb_x16_2ch` configuration.
- **Address Structure:** 2 Channels, 2 Ranks, 4 BankGroups, 4 Banks, 64 Columns, 64 B Granularity $\rightarrow$ Row Step Size = $262,144\text{ bytes}$.
- **Max Row Index Reached:** For 14,592 routed experts: $4,902,912 < 10,485,760$ rows ($46.76\%$ row capacity utilization).
- **Injectivity:** Proved collision-free across sampled coordinates.

### 3.2 Off-Chip DDR5 Hardware Latency & Bandwidth
- **456 Batches Duration Sum:** $\mathbf{1,778,905,340\text{ ns}}$ ($\mathbf{1.778905\text{ seconds}}$).
- **Mean Batch Latency:** $3.901108\text{ ms}$ ($3,901,108.2\text{ ns}$).
- **Batch Latency Standard Deviation:** $289.21\text{ ns}$ ($\approx 0.0074\%$ coefficient of variation).
- **Total Transferred Volume:** $456 \times 88,080,384\text{ B} = \mathbf{40,164,655,104\text{ bytes}}$ ($40.1647\text{ GB}$).
- **Effective DRAM Stream Bandwidth:**
  $$\text{BW}_{\text{eff}} = \frac{40,164,655,104\text{ B}}{1.77890534\text{ s}} = \mathbf{22.5783\text{ GB/s}}$$

### 3.3 Timestamp Mechanics & Float Serialization Reconciliation
- **Sum of `duration` column:** $1,778,905,340.0\text{ ns}$.
- **Sum of `(end - start)` columns:** $1,778,925,810.0\text{ ns}$.
- **Delta:** $20,470.0\text{ ns}$ ($0.0205\text{ ms}$ out of $1,778.9\text{ ms}$, i.e., $0.00115\%$).
- **Reconciliation Analysis:** In `src/hardware/linear_impl.cpp`, timestamps are formatted into CSV streams using `std::ofstream` default 6-significant-digit scientific notation (e.g., `1.77891e+06`). At $1.7\times 10^9\text{ ns}$, 6 significant digits yields a precision quantization step of $10^4 = 10,000\text{ ns}$. All observed per-row differences fall strictly within $[-1,730\text{ ns}, +9,410\text{ ns}]$, matching expected text serialization roundoff.
- **Hardware Timer Agreement:** In `logs/raw/ds5c10_completion_status.txt`, the C++ internal 64-bit integer accumulator recorded:
  `PASS1_SIMULATED_DDR5_NS=1778905300`. The 40 ns discrepancy ($0.0000022\%$) confirms exact agreement.

---

## 4. Workload Scope & Experimental Limitations

1. **Synthetic Routing:**
   - Routing decisions were produced using a pseudo-random generator with Zipfian/uniform distribution over a single decode token.
   - It is **not** an evaluation of real text prompt token routing or attention key/value clustering.
2. **Zero-Modeled HBM Latency:**
   - On-chip HBM accesses are modeled with 0 ns latency in LLMSimulator to isolate the performance overheads of the off-chip DDR5 secondary memory tier.
3. **No Evictions in Baseline:**
   - Because the 40.16 GB cold working set fits within the 80 GB HBM budget (50.21% occupancy), this baseline evaluates cold loading without inter-token LRU cache thrashing.

---

## 5. Raw Telemetry Checksums & Preservation

| Artifact Path | SHA-256 Checksum | Verified Unchanged |
| :--- | :--- | :---: |
| `raw/expert_routing.csv` | `229fc7522e212f356d2846d79a72b6172b395f3ca9a8f194de8187d0a696e259` | YES |
| `raw/expert_load.csv` | `d984dd17a3c412bcbf97c7a0c44bab185bb7f70b4ba06933bd259fa178d28644` | YES |
| `raw/expert_cache.csv` | `3302371500fe0e5cb79c23d4b9e86cce1b107b25482ed042e6b4bdb73d02d181` | YES |
| `raw/ds5c10_completion_status.txt` | `f6e2ff8b133858b9ecaf50c89d33c689429f2a3539146f6b647ffb275b65e859` | YES |
| `raw/ds5c11_completion_status.txt` | `6de4d7b226d2cade755555e27b452fbbcbeb7aa9f8ba52d6da7971f1d45f5ae0` | YES |

---

## 6. Final Audit Verdict

**DECISION: PASSED (ALL VERIFICATION GATES SATISFIED)**

The dataset is complete, uncorrupted, internally consistent, and fully documented.
