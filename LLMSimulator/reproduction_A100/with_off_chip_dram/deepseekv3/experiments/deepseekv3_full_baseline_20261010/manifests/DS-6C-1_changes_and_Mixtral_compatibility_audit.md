# DS-6C-1: DeepSeek-V3 Memory-System Changes Audit and Mixtral-Style Migration Assessment

**Audited Repository:** `/home/mohman/ca_project/LLMSimulator`  
**Audited Baseline Reference:** `reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/`  
**Audited Mixtral Reference:** `reproduction_A100/with_off_chip_dram/results/b1_full/` & `reproduction_A100/with_off_chip_dram/configs/b1_full.yaml`  
**Base Git Commit:** `f75d362ae157f81ef2d3bd4055c9b22e34882a9f`  
**Audit Date:** October 10, 2026  
**Auditor:** Antigravity Research Engineering Suite (Strict Read-Only Source & System Inspection)  
**Status:** COMPLETE — PHASE 1 AUDIT DELIVERABLE  

---

## 1. Executive Summary

This audit assesses the validated memory-system modifications implemented for DeepSeek-V3 in `LLMSimulator` and evaluates how to integrate them into a complete, reproducible experiment that follows the workflow, configuration conventions, output organization, and execution rigor established in our earlier Mixtral (`b1_full`) experiment.

### Key Audit Findings:

1. **Genuine Implementation of 8 Memory-System Changes:**
   A thorough code-level audit confirms that all eight claimed DeepSeek-V3 memory-system changes are genuinely implemented in C++ across `src/dram/`, `src/hardware/`, `src/module/`, and `src/scheduler/`. These encompass:
   - DDR5 address-space expansion to $2.75\text{ TB}$ ($10,485,760$ rows) to fit the $1.285\text{ TB}$ footprint of 14,592 routed experts without row overflow or aliasing.
   - A finite software LRU cache (`Device::resident_experts` / `lru_expert_list`) tracking byte-accurate residency against an 80 GB capacity budget with eviction logging.
   - Activation gating (`input->getSize() > 0`) preventing unrouted experts from touching cache or triggering false memory loads.
   - Corrected residency recording point, ensuring experts are registered in HBM cache strictly after Ramulator 2 DDR5 loading finishes.
   - Sister-weight batching ($W_1\text{_gate}, W_3\text{_up}, W_2\text{_down}$) consolidated into a single $88.08\text{ MB}$ off-chip transfer and simulated via batched PIM requests in `DRAMInterface`.
   - Dynamic off-chip DDR5 YAML loading alongside HBM3 80GB timing configuration.
   - Routing capture and deterministic replay using `BatchedSequence::cached_routing`.
   - File-based telemetry generation (`expert_routing.csv`, `expert_load.csv`, `expert_cache.csv`) completely decoupled from simulation correctness.

2. **Divergence Between Mixtral and DeepSeek-V3 Baselines:**
   - **Driver & Configuration:** The Mixtral experiment was driven by the generic `./build/run` executable (compiled from `eval/test.cpp`), loading all model, system, and hardware parameters dynamically from a single authoritative YAML file (`configs/b1_full.yaml`). The completed DeepSeek-V3 baseline used a dedicated test executable `./build/test_deepseekv3_full_baseline` (from `eval/test_deepseekv3_full_baseline.cpp`) with model parameters hardcoded in C++, loading only the DDR5 YAML via the `OFFCHIP_DRAM_CFG` environment variable.
   - **Workload Structure:** Mixtral executed 5 iterations of multi-token sequential decoding (`input_len: 2048`, `output_len: 128`, `iter: 5`, batch size 32) using `Cluster::runIteration`. The DeepSeek-V3 baseline evaluated a single decoding token across two manual passes (`cluster->run(metadata)`): Pass 1 (cold start, 456 experts loaded from DDR5) and Pass 2 (deterministic replay of the identical token, producing 100% cache hits).
   - **Cache Behavior:** Mixtral used an unbounded `std::set` (infinite residency tracker, where 256 experts $\approx 90.2\text{ GB}$ exceeded physical HBM without eviction). DeepSeek-V3 preserves a finite 80 GB LRU cache; however, in the 1-token 2-pass baseline, only $40.16\text{ GB}$ (456 experts) was accessed, meaning cache eviction was never triggered during the full run.

3. **Critical Scaling Warning for Phase 2 Migration:**
   Simulating 456 expert DDR5 loads ($40.16\text{ GB}$) in Ramulator 2 required **73.5 minutes of host CPU wall-clock time** in Pass 1. If DeepSeek-V3 were executed with Mixtral's full parameters (128 decode tokens $\times$ 5 iterations $\times$ batch size 32), the host CPU would be required to simulate thousands of gigabytes of cycle-accurate DDR5 traffic, requiring hundreds of hours. A calibrated token count and batch size must be selected to allow realistic multi-token cache thrashing and eviction within an acceptable runtime budget.

---

## 2. Detailed Inventory of DeepSeek-V3 Memory-System Changes

### Change 1: Expanded Off-Chip DDR5 Address-Space Capacity and Mapping
- **Source Files & Locations:**
  - `src/dram/memory_config.h`: lines 64–72 (`MemoryConfig ddr5_32gb_x16_2ch`).
  - `src/dram/mmap_controller.h`: lines 48–75 (`MMapController::addrToVec`) and lines 80–109 (`MMapController::addrToVecLOGIC`).
- **Implementation Mechanism:**
  - Standard dual-channel DDR5 32Gb has $2^{17} = 131,072$ rows per bank, providing $34.36\text{ GB}$ of capacity. DeepSeek-V3 has 14,592 routed experts across 57 MoE layers ($57 \times 256$). At FP16 precision, each 3-matrix expert is $88,080,384\text{ bytes}$, requiring $1,285,268,963,328\text{ bytes} \approx 1.285\text{ TB}$.
  - To prevent row address truncation in Ramulator 2, `num_row` was scaled by $80\times$ from $131,072$ to $10,485,760$.
  - Total addressable capacity is:
    $$\text{Capacity} = 2\text{ ch} \times 2\text{ ranks} \times 4\text{ BG} \times 4\text{ banks} \times 64\text{ cols} \times 64\text{ B granul} \times 10,485,760\text{ rows} = 2,748,779,069,440\text{ B} \approx 2.75\text{ TB} \ (2.5\text{ TiB})$$
  - In `MMapController::addrToVec` (used by GPU memory operations), the address decomposition order is `channel -> rank -> bankgroup -> bank -> col -> row`. With a row step size of $262,144\text{ bytes}$, the highest expert address $1,285,268,963,328$ maps to row index $4,902,912$, which is strictly less than $10,485,760$ ($46.76\%$ row capacity utilization).
- **Working Tree Status:** Fully present and active.

### Change 2: Expert-Cache Lookup, Insertion, Eviction, and LRU Replacement
- **Source Files & Locations:**
  - `src/hardware/device.h`: lines 121–136 (definitions of `lru_expert_list`, `resident_experts`, `cache_capacity_bytes`, `resident_bytes`, `reset_expert_cache`).
  - `src/hardware/device.cpp`: lines 269–298 (`Device::is_expert_resident`) and lines 300–332 (`Device::mark_expert_resident`).
- **Implementation Mechanism:**
  - Replaces Mixtral's unbounded `std::set<std::pair<int, int>>` with an LRU cache.
  - Double-ended queue `lru_expert_list` stores tuples of `<layer_id, expert_id, expert_size>`.
  - Hash map `resident_experts` maps `<layer_id, expert_id>` to iterators in `lru_expert_list`.
  - Cache lookup (`is_expert_resident`): On hit, moves the node to the back of `lru_expert_list` (MRU position), updates the hash map iterator, logs `layer_id,expert_id,HIT,resident_bytes` to `expert_cache.csv`, and returns `true`. On miss, logs `MISS` and returns `false`.
  - Cache insertion & eviction (`mark_expert_resident`): If the expert is not resident, checks `resident_bytes + expert_size > cache_capacity_bytes`. While over capacity, pops front nodes (LRU), erases from `resident_experts`, subtracts expert size from `resident_bytes`, and logs `EVICT`. Then inserts the incoming expert at the back (MRU), adds to `resident_bytes`, and logs `INSERT`.
  - Cache capacity defaults to $80 \times 10^9\text{ bytes}$ ($80\text{ GB}$), overridable via `DEBUG_CACHE_CAPACITY` environment variable.
- **Working Tree Status:** Fully present and active.

### Change 3: Execution-Path Gating for Empty or Inactive Expert Inputs
- **Source Files & Locations:**
  - `src/module/linear.cpp`: lines 45–56 (`Linear::forward`).
- **Implementation Mechanism:**
  - In MoE layers, modules are created for all 256 routed experts. When tokens are dispatched to top-8 experts, unrouted experts receive activation tensors with `input->getSize() == 0`.
  - Prior to this change, `Linear::forward` checked `is_expert_resident` and marked residency for all 256 experts unconditionally, polluting the cache with 512 unrouted experts during Pass 1.
  - The gating condition `if (A->is_expert_weight && input->getSize() > 0)` guarantees that unrouted experts never probe cache residency, never target `OFFCHIP_DRAM`, and never alter cache state.
- **Working Tree Status:** Fully present and active.

### Change 4: Point of Expert Residency Recording Relative to DRAM Loading
- **Source Files & Locations:**
  - `src/module/linear.cpp`: lines 48–56.
  - `src/hardware/linear_impl.cpp`: lines 81–95 (`LinearExecutionGPU`).
- **Implementation Mechanism:**
  - In earlier versions, `mark_expert_resident` was invoked inside `Linear::forward` before the module graph executed or before Ramulator returned, marking weights resident even if transfer had not occurred.
  - `mark_expert_resident` was removed completely from `Linear::forward`.
  - In `LinearExecutionGPU`, `device->mark_expert_resident` is now called strictly after `issueRamulatorBatch` returns with a valid `ExecStatus` for `OFFCHIP_DRAM`.
- **Working Tree Status:** Fully present and active.

### Change 5: W1, W2, W3 Sister-Weight Request Generation and Transfer Accounting
- **Source Files & Locations:**
  - `src/module/expert.cpp`: lines 101–118 (`ExpertFFN::ExpertFFN`).
  - `src/hardware/linear_impl.cpp`: lines 66–95, 100–135 (`LinearExecutionGPU`).
  - `src/hardware/layer_impl.cpp`: lines 43–85 (`issueRamulatorBatch`).
  - `src/dram/dram_interface.cpp`: lines 36–78 (`DRAMInterface::HandleRequest`).
  - `src/module/status.h`: lines 42, 53 (`ExecStatus::per_request_duration`).
- **Implementation Mechanism:**
  - In `ExpertFFN`, expert sub-matrices are assigned distinct names: `W1_gate`, `W3_up`, and `W2_down`. $W_1$ links sister weights: `w1->sister_weights = {w3, w2}`.
  - When $W_1$ misses the cache in `linear.cpp`, `w1->load_sisters = true`.
  - In `LinearExecutionGPU`, $W_1, W_3, W_2$ are bundled into `batch_tensors` ($29.36\text{ MB} \times 3 = 88.08\text{ MB}$) with common `expert_layer_id`.
  - `issueRamulatorBatch` bypasses the execution timing cache for `OFFCHIP_DRAM` (`bypass_cache = true`) and submits all three `DRAMRequest` objects to `device->run_ramulator_batch`.
  - In `DRAMInterface::HandleRequest`, PIM commands for all requests in the batch are queued before invoking `run()` once. The interface tracks `min_start` and `max_end` across all PIM requests, assigning `duration = max_end - min_start` and populating `per_request_duration`.
  - In `LinearExecutionGPU`, telemetry logs three individual rows (`W1_gate`, `W3_up`, `W2_down`) with their per-request durations, followed by an aggregate row (`all_weights`) with total size $88,080,384\text{ B}$ and batch duration.
- **Working Tree Status:** Fully present and active.

### Change 6: HBM and DDR5 Configuration Loading
- **Source Files & Locations:**
  - `src/hardware/device.cpp`: lines 34–40, 62–73 (`Device::Device`).
  - `eval/test.cpp`: lines 126–128 (`main`).
  - `eval/test_deepseekv3_full_baseline.cpp`: lines 55–56.
- **Implementation Mechanism:**
  - On-chip HBM: Hardcoded in `device.cpp:35` to load `./dram_config_HBM3_80GB.yaml` when `gpu_gen == "A100"`.
  - Off-chip DDR5: Loaded via `SystemConfig::offchip_dram_cfg_path`. In `eval/test.cpp:126`, loaded from YAML `config["system"]["optimization"]["offchip_dram_cfg_path"]`. In `test_deepseekv3_full_baseline.cpp`, loaded from `getenv("OFFCHIP_DRAM_CFG")`.
- **Working Tree Status:** Fully present and active.

### Change 7: Routing Capture, Replay, Token Identification, and Pass-Control Logic
- **Source Files & Locations:**
  - `src/scheduler/sequence.h`: line 132 (`std::map<std::pair<int, int>, std::vector<int>> cached_routing`).
  - `src/scheduler/sequence.cpp`: lines 173–240 (`BatchedSequence::update_expert`).
- **Implementation Mechanism:**
  - Telemetry logs routing decisions to `expert_routing.csv`: `request_id,token_id,layer_id,topk_rank,expert_id`.
  - Keyed by `std::make_pair(token_id, cur_layer)`:
    - If entry exists in `cached_routing`, reuses previously chosen experts.
    - If entry is absent, selects experts via `getRandomExpert` or `getZipfianRandomExpert` and stores in `cached_routing`.
  - Ensures that when the same token is re-evaluated across passes (by resetting `cur_layer = 0`), the identical routing pattern is replayed.
- **Working Tree Status:** Fully present and active.

### Change 8: Telemetry Output Generation and Simulation Isolation
- **Source Files & Locations:**
  - `eval/test.cpp`: lines 42–53 (`TRACE_DIR` directory initialization).
  - `src/scheduler/sequence.cpp`: lines 176–186 (`expert_routing.csv`).
  - `src/hardware/device.cpp`: lines 273–281, 309–314 (`expert_cache.csv`).
  - `src/hardware/linear_impl.cpp`: lines 97–135 (`expert_load.csv`).
- **Implementation Mechanism:**
  - Uses `TRACE_DIR` environment variable to direct CSV output without altering model graph computation.
  - In `eval/test.cpp`, derives `trace_dir` dynamically by replacing `/raw/` in `gantt_directory` with `/traces/`.
  - Flushes file streams on record writes (`.flush()`) to ensure no telemetry loss upon unexpected process termination.
- **Working Tree Status:** Fully present and active.

---

## 3. Source-File and Line-Number Evidence Table

| # | Feature / Change Area | Source File | Function / Scope | Source Line Range | Working Tree Status | Verified Test / Evidence Reference |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **1** | DDR5 10.48M Row Geometry | `src/dram/memory_config.h` | `ddr5_32gb_x16_2ch` | L66–L72 | Modified | `eval/mapping_test_output.txt` (PASS) |
| **1** | GPU Address Decomposition | `src/dram/mmap_controller.h` | `MMapController::addrToVec` | L48–L75 | Unmodified Base | `test_production_ddr5_mapping` (PASS) |
| **2** | LRU Queue & State Structures | `src/hardware/device.h` | `Device` class declaration | L121–L136 | Modified | `eval/lru_test_output.txt` (PASS) |
| **2** | LRU Cache Lookup & Promotion | `src/hardware/device.cpp` | `Device::is_expert_resident` | L269–L298 | Modified | `eval/lru_test_output.txt` (PASS) |
| **2** | LRU Insertion & Eviction Loop | `src/hardware/device.cpp` | `Device::mark_expert_resident` | L300–L332 | Modified | `test_cache_residency_remediation` (PASS) |
| **2** | Cache Reset Method | `src/hardware/device.h` | `Device::reset_expert_cache` | L132–L136 | Modified | `test_cache_residency_remediation` (PASS) |
| **3** | Empty Input Activation Gating | `src/module/linear.cpp` | `Linear::forward` | L48–L56 | Modified | `test_cache_residency_remediation` (PASS) |
| **4** | Post-Load Residency Registration | `src/hardware/linear_impl.cpp` | `LinearExecutionGPU` | L87–L95 | Modified | `test_cache_residency_remediation` (PASS) |
| **5** | Sister Matrix Naming (W1/W2/W3) | `src/module/expert.cpp` | `ExpertFFN::ExpertFFN` | L101–L118 | Modified | `eval/minimal_timing_output.txt` (PASS) |
| **5** | Batch Request Formation | `src/hardware/linear_impl.cpp` | `LinearExecutionGPU` | L66–L80 | Modified | `eval/multi_expert_timing_output.txt` (PASS) |
| **5** | Batched Off-Chip RAM Issue | `src/hardware/layer_impl.cpp` | `issueRamulatorBatch` | L43–L85 | Modified | `eval/minimal_timing_output.txt` (PASS) |
| **5** | Batched PIM Execution & Timing | `src/dram/dram_interface.cpp` | `DRAMInterface::HandleRequest` | L36–L78 | Modified | `eval/minimal_timing_output.txt` (PASS) |
| **5** | Per-Request Duration Vector | `src/module/status.h` | `ExecStatus` struct | L42, L53 | Modified | `eval/minimal_timing_output.txt` (PASS) |
| **5** | Constituent & Aggregate Logging | `src/hardware/linear_impl.cpp` | `LinearExecutionGPU` | L100–L135 | Modified | `raw/expert_load.csv` (2,280 rows) |
| **6** | HBM3 80GB Loading (A100) | `src/hardware/device.cpp` | `Device::Device` | L34–L40 | Modified Base | `logs/full_baseline_run.log` |
| **6** | Dynamic DDR5 YAML Loading | `eval/test.cpp` | `main` | L126–L128 | Modified | `reproduction_A100/results/b1_full/` |
| **7** | Routing Key & Cached Map | `src/scheduler/sequence.h` | `BatchedSequence` | L132 | Modified | `raw/expert_routing.csv` (912 rows) |
| **7** | Deterministic Replay Logic | `src/scheduler/sequence.cpp` | `BatchedSequence::update_expert` | L208–L232 | Modified | `raw/expert_routing.csv` (Pass 1 == Pass 2) |
| **8** | TRACE_DIR Automatic Derivation | `eval/test.cpp` | `main` | L42–L53 | Modified | `eval/test.cpp` |
| **8** | Synchronous Stream Flush | `src/scheduler/sequence.cpp` | `BatchedSequence::update_expert` | L235–L237 | Modified | `raw/expert_routing.csv` |

---

## 4. Existing Test Coverage and Missing Regression Tests

### 4.1 Verified Existing Tests (All Currently Present and Passing)

1. **`eval/test_ddr5_mapping.cpp` & `eval/test_production_ddr5_mapping.cpp`:**
   - *Tested Functionality:* Validates that the expanded $10,485,760$ row geometry maps all $1,285,268,963,328\text{ bytes}$ of 14,592 routed experts injectively.
   - *Status:* PASSED. Output records highest row index $4,902,912 < 10,485,760$ with 0% aliasing.
2. **`eval/test_lru.cpp`:**
   - *Tested Functionality:* Tests LRU eviction mechanics: equal-size eviction, variable-size eviction, multiple evictions per insertion, oversized entries, duplicate insertions, and 908-expert budget limit.
   - *Status:* PASSED. Output records: `All LRU tests passed.`
3. **`eval/test_cache_residency_remediation.cpp`:**
   - *Tested Functionality:* Specifically verifies the DS-5C-9 correctness remediations: unrouted module invariant, cold miss delay, warm hit promotion, capacity-bounded eviction, and cache reset invariant.
   - *Status:* PASSED. Output records: `ALL CACHE RESIDENCY REMEDIATION TESTS PASSED!`
4. **`eval/test_minimal_timing.cpp` & `eval/test_multi_expert_timing.cpp`:**
   - *Tested Functionality:* Tests Ramulator 2 DDR5 batch timing for $W_1, W_2, W_3$ sister weights, timing cache bypass for off-chip transfers, and per-request duration accounting.
   - *Status:* PASSED. Output confirms batch memory duration of $3.90\text{ ms}$ for $88.08\text{ MB}$ load.
5. **`eval/test_staged_module_graph.cpp`:**
   - *Tested Functionality:* Multi-layer model graph execution on 2 MoE layers (5 total layers) with 16 cold expert loads and warm replay.
   - *Status:* PASSED. Runtime: 164.1s wall-clock, $62.42\text{ ms}$ simulated DDR5 time.
6. **`eval/test_deepseekv3_full_baseline.cpp`:**
   - *Tested Functionality:* Full 60-layer baseline (3 dense + 57 MoE layers), 456 cold expert loads in Pass 1, 100% cache hits in Pass 2.
   - *Status:* PASSED. Exit code 0, 4,439.0s wall-clock, $1,778.57\text{ ms}$ simulated DDR5 time, 0.0 ns replay delta.

### 4.2 Missing Regression Tests & Coverage Gaps

1. **Multi-Token Sequential Decoding Regression Test:**
   - *Gap:* Existing tests only evaluated single-token decoding with either a 1-pass execution or a manual reset replay of the *exact same token*. In real sequential decoding, `token_id` advances sequentially ($T_0, T_1, T_2, \dots$), drawing distinct expert combinations.
   - *Risk:* No test currently verifies that cache occupancy expands across multiple sequential decoding tokens until exceeding 80 GB and triggers physical LRU evictions in an active multi-layer model graph.
2. **Generic YAML Driver (`eval/test.cpp`) DeepSeek-V3 Integration Test:**
   - *Gap:* The full baseline was run using `test_deepseekv3_full_baseline.cpp`, which bypassed `Cluster::runIteration`. `eval/test.cpp` invokes `cluster->runIteration` and `scheduler->hittingQueue(10000)`.
   - *Risk:* While `eval/test.cpp` has `deepseekV3` model support, running a full 60-layer model through `runIterationMixed` could encounter issues with sequence queues, warmup routines, or output CSV generation.
3. **Non-Zero Skewness Multi-Token Decoding Test:**
   - *Gap:* The baseline was evaluated with uniform synthetic routing (`skewness: 0.0`). Realistic temporal correlation requires evaluating Zipfian distributions (`skewness > 0`) over sequential tokens.

---

## 5. Mixtral vs. DeepSeek-V3 Workflow Comparison

| Dimension / Aspect | Mixtral Workflow (`b1_full`) | Completed DeepSeek-V3 Baseline (`ds5c11`) | Migration / Reusability Classification |
| :--- | :--- | :--- | :--- |
| **Driver Executable** | Generic `./build/run` (from `eval/test.cpp`) | Dedicated `./build/test_deepseekv3_full_baseline` | **Reusable with Adaptation:** Adapt `eval/test.cpp` to serve as the unified driver for the new experiment. |
| **Configuration Loading** | Single authoritative YAML (`b1_full.yaml`) via CLI argument | C++ hardcoded values in `test_deepseekv3_full_baseline.cpp`; DDR5 YAML via `OFFCHIP_DRAM_CFG` env var | **Reusable with Adaptation:** Create a dedicated authoritative YAML (`deepseekv3_b1_full.yaml`) matching Mixtral schema. |
| **Model Structure** | 32 layers, 8 experts/layer, top-2 (64 expert accesses/token) | 60 layers (3 dense + 57 MoE), 256 experts/layer, top-8 (456 expert accesses/token) | **DeepSeek-V3 Specific:** Must retain 60-layer architecture, 256 routed experts, 1 shared expert, and MLA config. |
| **Expert Footprint** | $11.2\text{ MB}$ per expert ($2.87\text{ GB}$ total for 256 experts) | $88.08\text{ MB}$ per expert ($1.285\text{ TB}$ total for 14,592 routed experts) | **DeepSeek-V3 Specific:** Must retain $88.08\text{ MB}$ footprint and 2.75 TB DDR5 geometry. |
| **Cache Model** | Unbounded `std::set` (infinite residency tracker; 0 evictions) | Finite 80 GB LRU cache (`lru_expert_list` + `resident_experts`) | **DeepSeek-V3 Specific:** Must retain finite 80 GB LRU cache and eviction logging. |
| **Workload Scope** | 5 iterations of 128 decode tokens, batch size 32 | 1 decoding token across 2 passes (Pass 1 cold, Pass 2 warm replay), batch size 1 | **Requires Careful Adaptation:** Scale token count and batch size realistically to prevent intractable runtimes. |
| **Execution Loop** | `Cluster::runIteration` $\rightarrow$ `runIterationMixed` | Manual `cluster->run(metadata)` per pass | **Reusable with Adaptation:** Use `Cluster::runIteration` if compatible, or calibrate a dedicated runner. |
| **Warmup Behavior** | `hittingQueue(10000)` schedules requests without graph execution | Explicit dummy sequence push; no warmup graph execution | **Reusable without Changes:** `hittingQueue` leaves expert cache untouched. |
| **Gantt / Compute Trace** | `cluster->exportGantt` writes `processor_trace.csv` (subject to TimeBoard trimming) | Skipped Gantt export to avoid memory bloat; analytical GPU compute | **Reusable with Adaptation:** Skip Gantt export or enforce memory bounds to prevent OOM. |
| **Runner Script** | Simple CLI execution or shell wrapper | Robust bash runner with single-run lock, RAM/disk guards, RSS monitoring, 60s heartbeats, completion status | **Reusable with Adaptation:** Adopt the robust DeepSeek-V3 runner structure for the new experiment. |
| **Output Hierarchy** | `csv/`, `raw/`, `traces/`, `logs/`, `analysis/` | `raw/`, `processed/`, `figures/`, `report/`, `logs/`, `manifests/`, `scripts/` | **Reusable with Adaptation:** Unify both hierarchies into a clean, complete layout. |
| **Analysis Scripts** | `parse_csv.py`, `analyze_phase8.py`, `analyze_phase9.py` | `process_metrics.py`, `analyze_full_baseline.py`, `generate_graphs.py` | **Reusable with Adaptation:** Adapt DeepSeek-V3 parser and analysis scripts for multi-token decode traces. |

---

## 6. Compatibility Matrix for Intended Single-GPU Experiment

| Requirement / Parameter | Current Implementation Status | Fidelity & Verification Classification | Technical Analysis & Distinction |
| :--- | :---: | :--- | :--- |
| **DeepSeek-V3 Model Shape** | Fully Implemented | Verified Source Fact | 60 total layers (3 dense, 57 MoE), top-8, 256 routed experts, 1 shared expert, hidden dim 7168, intermediate dim 2048. |
| **Single A100-Class GPU Reference** | Implemented & Configured | Verified Source Fact | `num_node=1, num_device=1, gpu_gen="A100"`. Execution path executes on a single device with zero-cost inter-device communication no-ops. |
| **No Multi-GPU Sharding** | Implemented & Configured | Verified Source Fact | `expert_tensor_degree=1, none_expert_tensor_degree=1`. No tensor or pipeline parallel splitting. |
| **No Logic-PIM Execution** | Implemented | Verified Source Fact | All memory operations target `MemoryTarget::OFFCHIP_DRAM` or `HBM`. Logic-PIM processing is completely bypassed. |
| **Off-Chip DDR5 Backing Store** | Fully Implemented | Verified Timing Model | Dual-channel DDR5-3200BN simulated cycle-by-cycle via Ramulator 2 (`dram_config_DDR5_retry.yaml`). Address space sized to 2.75 TB. |
| **Transfers Only for Selected Experts** | Fully Implemented | Verified Source Fact | Backing store maps all 14,592 experts, but memory reads are generated strictly for active routed experts via `linear.cpp:48`. |
| **80 GB HBM Local Cache Reference** | Implemented as Software LRU | Verified Software Model | **Important Distinction:** 80 GB capacity is enforced as an LRU software threshold. Cache hits incur 0 ns memory delay. It is an off-chip access filter, not a cycle-accurate HBM load simulation for hits. |
| **FP16 Accounting** | Implemented | Verified Source Fact | Expert weights are modeled at 2 bytes/element ($88.08\text{ MB}$/expert). Roofline FLOPS evaluated against FP16 A100 peak ($312\text{ TFLOPS}$). |
| **DRAM Timing vs. GPU Compute** | Implemented | Verified Model Distinction | Off-chip DDR5 transfer time is simulated cycle-by-cycle in Ramulator 2; GPU computation time is calculated analytically via roofline ($MAC / FLOPS$). |
| **Sequential Decoding Tokens for Temporal Correlation** | Partially Implemented | Requires Migration | Baseline tested single-token replay (`token_id=1`). Multi-token sequential decoding ($token\_id$ incrementing) is supported by `sequence.cpp` but requires multi-token execution. |

---

## 7. Proposed New Experiment Directory and Artifact Layout

The proposed new experiment will reside in a completely isolated directory tree:

```
reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/
```

### Complete Proposed Directory Structure:

```
deepseekv3_mixtral_style_20261011/
├── configs/
│   ├── deepseekv3_b1_full.yaml              # Authoritative run configuration (Mixtral schema)
│   ├── dram_config_DDR5_retry.yaml          # DDR5 Ramulator 2 configuration snapshot
│   └── dram_config_HBM3_80GB.yaml           # HBM3 Ramulator 2 configuration snapshot
├── scripts/
│   ├── run_mixtral_style_experiment.sh      # Production runner with watchdog, RAM guard, logging
│   ├── process_metrics.py                   # Lightweight CSV transformer for raw telemetry
│   ├── analyze_results.py                   # Multi-token temporal and correlation analysis
│   └── generate_figures.py                  # Publication-ready plots (PNG + PDF)
├── logs/
│   ├── build.log                            # Compilation log
│   ├── run.log                              # Execution stdout/stderr log
│   └── completion_status.txt                # Machine-readable completion status (key=value)
├── csv/
│   └── deepseekV3_synthesis_..._ramul_decode.csv # Standard summary output from Cluster::runIteration
├── raw/
│   └── device_0                             # Raw Gantt trace (if export_gantt enabled)
├── traces/
│   ├── expert_routing.csv                   # Sequence-level routing decisions
│   ├── expert_load.csv                      # W1, W2, W3 constituent & aggregate load events
│   ├── expert_cache.csv                     # HIT, MISS, INSERT, EVICT LRU events
│   └── processor_trace.csv                  # Gantt-derived compute intervals (if enabled)
├── processed/
│   ├── summary_metrics.csv                  # Macro-level summary metrics
│   ├── summary_metrics.json                 # JSON summary for programmatic consumption
│   ├── per_token_metrics.csv                # Metrics broken down per sequential decoding token
│   ├── per_iteration_metrics.csv            # Metrics broken down per iteration
│   ├── per_layer_metrics.csv                # Per-layer load and cache statistics
│   ├── per_expert_metrics.csv               # Expert frequency, hotness, residency times
│   └── integrity_checks.json                # Automated mathematical consistency verification
├── figures/
│   ├── fig1_temporal_routing_correlation.{png,pdf}
│   ├── fig2_multi_token_cache_occupancy.{png,pdf}
│   ├── fig3_lru_eviction_distribution.{png,pdf}
│   ├── fig4_per_token_ddr5_latency.{png,pdf}
│   ├── fig5_compute_vs_memory_overlap.{png,pdf}
│   └── README.md
├── manifests/
│   ├── experiment_manifest.md               # Purpose, parameters, environment, and inputs
│   ├── input_checksums.sha256               # Cryptographic hashes of all configs and code
│   └── reproducibility_guide.md             # Exact instructions to replicate results
└── report/
    └── results.md                           # Comprehensive scientific assessment report
```

---

## 8. Risks and Unresolved Questions

1. **Simulation Runtime Scaling (The 74-Minute Factor):**
   - In the DS-5C-11 baseline, simulating 456 cold expert loads ($40.16\text{ GB}$) in Ramulator 2 required **4,411 seconds (~73.5 minutes)** of host CPU wall-clock time.
   - If Mixtral's full parameters are replicated directly (`output_len: 128`, `iter: 5`, `max_batch_size: 32`), the simulator would attempt to simulate thousands of gigabytes of DDR5 traffic, requiring hundreds of hours of wall-clock time.
   - *Recommendation:* Calibrate the workload to a bounded sequential decoding problem (e.g., `input_len: 2048`, `output_len: 10`, `iter: 5`, batch size 1). Over 5 iterations, this generates enough sequential decoding tokens to exceed the 80 GB cache limit ($> 908$ unique expert selections) and actively exercise LRU eviction, while keeping total host runtime tractable (~3–5 hours).
2. **`TimeBoard` Memory Overhead & Gantt Export:**
   - In Mixtral, `processor_trace.csv` suffered from TimeBoard memory trimming because earlier events were discarded to prevent memory exhaustion.
   - For DeepSeek-V3, instantiating Gantt timestamps across 60 layers could cause substantial host RAM usage.
   - *Recommendation:* Disable Gantt export (`export_gantt: false`) for initial full-model multi-token runs, relying on `expert_load.csv`, `expert_routing.csv`, and `expert_cache.csv`, which provide complete timing information without TimeBoard memory pressure.
3. **Cache Capacity Exposure in YAML:**
   - `eval/test.cpp` does not currently read an `expert_cache_capacity_bytes` field from YAML; it defaults to `80000000000LL` or `DEBUG_CACHE_CAPACITY`.
   - *Recommendation:* Support `DEBUG_CACHE_CAPACITY` in the runner script to ensure the 80 GB budget is explicitly set.
4. **Warmup Queue Execution:**
   - `Cluster::runIteration` executes `scheduler->hittingQueue(10000)`. Our audit confirmed that `hittingQueue` schedules requests but does not execute the model graph, so it will not pollute the cache. However, this must be verified during smoke testing.

---

## 9. Phased Recommendation

To ensure safe, reproducible execution without risking data corruption, system exhaustion, or wasted runtime:

- **Phase 2 — Configuration & Runner Preparation (No Large Runs):**
  1. Author `configs/deepseekv3_b1_full.yaml` adhering to the Mixtral schema.
  2. Author `scripts/run_mixtral_style_experiment.sh` incorporating lock guarding, RAM/disk safety thresholds, RSS monitoring, and 60-second heartbeats.
  3. Ensure `TRACE_DIR` points cleanly to `traces/` and output CSV points to `csv/`.
- **Phase 3 — Staged Smoke Verification:**
  1. Run a 2-layer MoE smoke test (5 total layers, 2 tokens) using the YAML driver to confirm end-to-end pipeline execution, CSV output generation, and trace logging.
  2. Verify that `hittingQueue` executes cleanly without crashing or corrupting cache state.
- **Phase 4 — Calibrated Full-Model Production Execution:**
  1. Launch the calibrated full-model run under watchdog supervision.
  2. Monitor progress via heartbeat logs and memory usage.
  3. Verify exit code 0 and generate `completion_status.txt`.
- **Phase 5 — Multi-Token Telemetry Analysis & Reporting:**
  1. Run `process_metrics.py` to produce summary datasets.
  2. Run `analyze_results.py` to compute temporal routing correlation and eviction rates.
  3. Generate publication-ready figures (`generate_figures.py`) in both PNG and PDF formats.
  4. Author the final scientific results report (`report/results.md`).

---

## 10. Exact Commands Proposed for Future Phases (Non-Executed)

The following commands are proposed for subsequent phases. **In accordance with Phase 1 instructions, none of these commands have been executed.**

### Phase 2: Build & Configuration Verification
```bash
# 1. Verify build environment and compile simulator
cd /home/mohman/ca_project/LLMSimulator
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) run
cd /home/mohman/ca_project/LLMSimulator

# 2. Check binary existence
ls -la ./build/run
```

### Phase 3: Smoke Test Execution (Lightweight 2-Layer Test)
```bash
# Execute lightweight smoke test with bounded configuration
./build/run reproduction_A100/with_off_chip_dram/deepseekv3/configs/ds5c3_smoke_test.yaml \
    > reproduction_A100/with_off_chip_dram/deepseekv3/results/ds5c3_smoke_test/smoke.log 2>&1
```

### Phase 4: Production Run Execution
```bash
# Launch calibrated production experiment via dedicated runner
bash reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/scripts/run_mixtral_style_experiment.sh
```

### Phase 5: Post-Run Analysis & Plot Generation
```bash
# 1. Process raw telemetry
python3 reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/scripts/process_metrics.py \
    --trace-dir reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/traces \
    --output-dir reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/processed

# 2. Perform temporal routing and memory analysis
python3 reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/scripts/analyze_results.py \
    --exp-dir reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011

# 3. Generate publication-ready figures (PNG & PDF)
python3 reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/scripts/generate_figures.py \
    --processed-dir reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/processed \
    --output-dir reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/figures
```

---

*Report authored strictly in accordance with Phase 1 constraints. No existing simulation outputs, configurations, or source files have been modified or deleted.*
