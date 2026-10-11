# DS-5C-14: HBM, Off-Chip DRAM, and Main Log Source Audit

**Experiment Directory:** `reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/`  
**Target Executable:** `build/test_deepseekv3_full_baseline`  
**Executable Source:** `eval/test_deepseekv3_full_baseline.cpp`  
**Runner Script:** `reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/scripts/run_full_baseline.sh`  
**Date:** October 10, 2026  
**Auditor:** Antigravity Research Engineering Suite (Strict Read-Only Source Trace)  
**Status:** COMPLETE — 100% VERIFIED FROM SOURCE CODE AND RUNTIME LOGS  

---

## 1. Executive Summary

This audit establishes the concrete mechanisms governing HBM memory configuration, off-chip DDR5 memory configuration, expert weight dataflow, and logging for the completed DeepSeek-V3 60-layer full baseline simulation.

### Key Conclusions:
1. **HBM Configuration:** HBM is represented by **two distinct abstractions**:
   - A cycle-accurate Ramulator 2 timing model (`DRAMInterface`), configured via `./dram_config_HBM3_80GB.yaml` in the repository root.
   - A software LRU residency tracker (`Device::resident_experts` / `lru_expert_list`), configured with an 80 GB capacity limit hardcoded in C++ at `eval/test_deepseekv3_full_baseline.cpp:100`.
   - In this baseline, on-chip HBM expert hits are modeled with **0 ns memory delay** (`linear_impl.cpp:77`) to specifically isolate the latency contribution of the off-chip DDR5 bus.
2. **Off-Chip DDR5 Configuration:** DDR5 is configured dynamically via the environment variable `OFFCHIP_DRAM_CFG` exported in `run_full_baseline.sh:11,66`. It points to `config_snapshot/dram_config_DDR5_retry.yaml`. Ramulator 2 simulates dual-channel DDR5-3200BN with an address mapping defined in C++ by `ddr5_32gb_x16_2ch` in `src/dram/memory_config.h`.
3. **Weight Dataflow:** When an un-cached expert is probed by its first gate projection ($W_1$), a miss triggers a batched 88.08 MB DDR5 read encompassing all three sister matrices ($W_1\text{_gate}, W_3\text{_up}, W_2\text{_down}$). Upon load completion, the expert is inserted into the HBM LRU cache. Subsequent sister queries ($W_3, W_2$) and Pass 2 replay queries hit the cache, incurring zero DDR5 traffic.
4. **Main Runtime Log:** The definitive runtime log is located at:
   ```
   /home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/logs/full_baseline_run.log
   ```
   It captures both `stdout` and `stderr` via `>> "$LOG_FILE" 2>&1`, alongside 60-second watchdog heartbeats written via `tee -a`.

---

## 2. Detailed Findings

### Question 1: How Does the Simulator Configure HBM?

#### 1.1 Exact Configuration File Opened at Runtime
The HBM configuration file opened at runtime is:
```
/home/mohman/ca_project/LLMSimulator/dram_config_HBM3_80GB.yaml
```
- **Source Mechanism:** In `src/hardware/device.cpp:34-35`:
  ```cpp
  if (config.gpu_gen == "H100" || config.gpu_gen == "A100") {
    dram_cfg_path = "./dram_config_HBM3_80GB.yaml";
  }
  ```
- Because `run_full_baseline.sh` executes `cd "$REPO_DIR"` (`/home/mohman/ca_project/LLMSimulator`) prior to invoking the binary, the relative path resolves directly to the repository root's `dram_config_HBM3_80GB.yaml`.
- The file `config_snapshot/dram_config_HBM3_80GB.yaml` inside the experiment directory has an identical SHA-256 hash (`e7ae1fb6...`), but was preserved as a post-run snapshot; the binary itself opened the repository root file.

#### 1.2 Loader Function and Representing Objects
- **File Loader:** `YAML::LoadFile(dram_cfg_path)` in `src/hardware/device.cpp:40`.
- **Primary DRAM Model:** Instantiated in `src/hardware/device.cpp:59`:
  ```cpp
  dram_interface = DRAMInterface::Create(dram_cfg_path, memory_scale_factor);
  ```
  with `memory_scale_factor = 0.76923` (representing HBM3 80GB at 5.2 Gbps pin rate, line 46).
- **Address Mapping Controller:** Instantiated in `src/hardware/device.cpp:60`:
  ```cpp
  mmap_controller = MMapController::Create(memory_config);
  ```
  where `memory_config = hbm3_80GB;` (line 47).
- **Device Object:** `Device::Ptr device` (`src/hardware/device.h:20-155`) holds `dram_interface` as its primary on-chip memory model.

#### 1.3 Definition of the 80 GB Capacity
The 80 GB cache limit is defined in two locations:
1. **Runtime Override in Executable:**
   In `eval/test_deepseekv3_full_baseline.cpp:100`:
   ```cpp
   device->cache_capacity_bytes = 80000000000LL; // 80 GB standard HBM budget
   ```
2. **Default Member Initialization:**
   In `src/hardware/device.h:127`:
   ```cpp
   long long cache_capacity_bytes = std::getenv("DEBUG_CACHE_CAPACITY") ? 
       std::stoll(std::getenv("DEBUG_CACHE_CAPACITY")) : 80000000000LL;
   ```
   Both explicitly set the capacity to $80 \times 10^9$ bytes ($80\text{ GB}$).

#### 1.4 Ramulator Memory Model vs. Expert-Cache Abstraction
HBM is implemented through **two complementary components**:
1. **Ramulator 2 Memory Model (`device->dram_interface`):**
   - Implements cycle-accurate DRAM channel, rank, and bank-level timing for activation tensors (input and output tensors in `issueRamulator` at `linear_impl.cpp:44,138`).
   - Represents the physical HBM3 memory controller and PHY.
2. **Software Expert-Cache Abstraction (`device->resident_experts` / `device->lru_expert_list`):**
   - Implements a software-level LRU cache tracking expert weight residency on the GPU.
   - Stores tuples of `{layer_id, expert_id, expert_size}` in `std::list` (`src/hardware/device.h:125`) and indexes them using `std::unordered_map` keyed by `{layer_id, expert_id}` (`src/hardware/device.h:126`).
   - Tracks total resident memory in `device->resident_bytes`.

#### 1.5 Residency Tracking and Cache-Hit Mechanics
- When `Linear::execution` runs in `src/module/linear.cpp:48-56`:
  ```cpp
  if (A->is_expert_weight && input->getSize() > 0) {
    int layer_id = sequences_metadata->cur_layer;
    A->expert_layer_id = layer_id;
    if (!device->config.offchip_dram_cfg_path.empty()) {
      if (!device->is_expert_resident(layer_id, A->expert_id)) {
        A->weight_target = MemoryTarget::OFFCHIP_DRAM;
        A->load_sisters = true;
      }
    }
  }
  ```
- If `device->is_expert_resident(layer_id, A->expert_id)` returns `true`:
  - `A->weight_target` remains default `MemoryTarget::HBM`.
  - `A->load_sisters` remains `false`.
  - In `src/hardware/linear_impl.cpp:77`:
    ```cpp
    // It was a hit, no OFFCHIP_DRAM delay
    memory_cost = 0;
    ```
  - The off-chip DRAM subsystem is completely bypassed, eliminating DDR5 bus traffic.

#### 1.6 Is HBM Actually Used by the Expert-Loading Path?
- **For Expert Weights:** In this baseline memory-system study, when an expert hits HBM, its transfer latency is modeled as **0 ns** (`memory_cost = 0`). The Ramulator HBM interface is not stepped forward for resident expert hits. This deliberate design isolates the secondary DDR5 DRAM bus latency and queuing overheads.
- **For Non-Expert Tensors:** The HBM Ramulator model **is actively used** for activation reads and writes:
  - Input activation read: `linear_impl.cpp:44` (`issueRamulator(..., MemoryTarget::HBM)`).
  - Output activation write: `linear_impl.cpp:138` (`issueRamulator(..., MemoryTarget::HBM)`).

---

### Question 2: How Does the Simulator Configure Off-Chip DDR5?

#### 2.1 Exact YAML Configuration File Used
The active DDR5 configuration file is:
```
/home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/config_snapshot/dram_config_DDR5_retry.yaml
```
- SHA-256 Checksum: `7ef030f82d7f654339aa0ab1ff31abce311322c59aee693eb57774e071db9d97`.

#### 2.2 Where the Environment Variable is Set and Read
1. **Set:** In `scripts/run_full_baseline.sh`:
   - Line 11: `CONFIG_DDR5="${EXP_DIR}/config_snapshot/dram_config_DDR5_retry.yaml"`
   - Line 66: `export OFFCHIP_DRAM_CFG="$CONFIG_DDR5"`
2. **Read:** In `eval/test_deepseekv3_full_baseline.cpp`:
   - Lines 55–56:
     ```cpp
     const char* ddr5_cfg_env = std::getenv("OFFCHIP_DRAM_CFG");
     system_config.offchip_dram_cfg_path = ddr5_cfg_env ? ddr5_cfg_env : 
         "reproduction_A100/with_off_chip_dram/configs/dram_config_DDR5_retry.yaml";
     ```
3. **Consumed:** In `src/hardware/device.cpp`:
   - Lines 62, 72:
     ```cpp
     std::string offchip_cfg = config.offchip_dram_cfg_path;
     ...
     offchip_dram_interface = DRAMInterface::Create(offchip_cfg, offchip_scale_factor);
     ```

#### 2.3 YAML Parameters Configuring Timing and Geometry
From `config_snapshot/dram_config_DDR5_retry.yaml`:
- **DRAM Model:** `impl: DDR5` (line 11)
- **Organization:**
  - `preset: DDR5_32Gb_x16` (line 13)
  - `channel: 2` (line 14)
  - `rank: 2` (line 15)
- **Timing:** `preset: DDR5_3200BN` (line 17)
- **Row Hammer Mitigation:** `RFM.BRC: 2` (line 19)
- **Controller Implementation:** `PIM_DRAM_controller` with `PIM_Scheduler` and `AllBank` refresh (lines 22–26)
- **Address Mapping Scheme:** `AddrMapper.impl: RoBaRaCoCh` (Row-Bank-Rank-Column-Channel, line 29)

#### 2.4 Where Model Addresses are Generated and Mapped to DDR5
- **Logical Address Generation:**
  In `src/dram/mmap_controller.cpp:178-189` (`MMapController::setNormal`), tensors are allocated sequential start addresses:
  ```cpp
  addr start_address = start_addr_normal;
  start_addr_normal += tensor->getSize();
  ```
- **DDR5 Physical Address Mapping:**
  In `src/dram/mmap_controller.cpp:21-37` (`MMapController::getAddrVec`), Ramulator converts the logical address to DDR5 coordinate tuples:
  ```cpp
  Ramulator::AddrVec_t ramul_addr_vec = {
      channel, addr_vec.rank, addr_vec.bankgroup,
      addr_vec.bank, addr_vec.row, addr_vec.col
  };
  ```
  Governed by `ddr5_32gb_x16_2ch` in `src/dram/memory_config.h` (2 channels, 2 ranks, 4 bankgroups, 4 banks, 64 columns, 10,485,760 rows, 64-byte burst granularity).

#### 2.5 How W1, W2, and W3 Expert Weights Generate Memory Requests
- In `src/module/expert.cpp:104-118`, when `FeedForward3Way` is instantiated, the three matrices are assigned names and mutual sister links:
  ```cpp
  w1->sister_weights = {w3, w2};
  w3->sister_weights = {w1, w2};
  w2->sister_weights = {w1, w3};
  ```
- When $W_1$ misses the cache in `src/module/linear.cpp:54`, `A->load_sisters = true;`.
- In `src/hardware/linear_impl.cpp:50-56`, `LinearImpl` aggregates $W_1$, $W_3$, and $W_2$ into a single `batch_tensors` vector:
  ```cpp
  if (weight->load_sisters) {
    for (auto sister : weight->sister_weights) {
      sister->expert_layer_id = weight->expert_layer_id;
      batch_tensors.push_back(sister);
    }
    weight->load_sisters = false;
  }
  ```
- In `src/hardware/layer_impl.cpp:58-70`, `issueRamulatorBatch` constructs an individual `DRAMRequest` for each tensor in `batch_tensors` and dispatches the batch to Ramulator via `device->run_ramulator_batch(requests, MemoryTarget::OFFCHIP_DRAM)`.

#### 2.6 Backing Addresses vs. Selected Expert Transfers
- **Backing Addresses:** **All 14,592 routed experts** across all 57 layers have valid, non-overlapping backing addresses allocated in the 2.75 TB DDR5 logical address space during model instantiation (`eval/test_deepseekv3_full_baseline.cpp:236-248`).
- **Simulated Transfers:** The simulator **only generates and issues memory requests for activated experts**. Only the 8 experts selected by the router per layer per token are dispatched to Ramulator 2. Unrouted experts remain unaccessed and generate zero bus transactions.

#### 2.7 Timing/Accounting vs. Physical Data Movement
> [!IMPORTANT]
> **Simulation Methodology Distinction:**  
> LLMSimulator is an **architectural performance timing simulator**, not a functional emulator or physical GPU system:
> - It models the **cycle-accurate timing, queuing, and bus contention** of data movement using Ramulator 2's command queues and bank state machines.
> - It **does not** physically transfer 40 GB or 1.28 TB of real weight floats across physical DRAM or PCIe buses on the host machine.
> - Tensors are lightweight C++ metadata descriptors (`Tensor`, `MemoryObject`) tracking dimensions and logical address offsets, which is why peak process RSS was only **1.08 GB** on the host.

---

### Question 3: How Do HBM and DDR5 Work Together?

The following trace details the exact step-by-step lifecycle of one expert request ($E_{id}=15$, Layer 3 / MoE Layer 0):

```
Token reaches MoE Layer 0 -> Router selects Expert 15 (Top-1)
  │
  ├─ 1. Cache Probe (src/module/linear.cpp:52)
  │      Device::is_expert_resident(layer=0, expert=15) -> MISS
  │      Logs: "0,15,MISS,0" to expert_cache.csv
  │      Sets: W1->weight_target = OFFCHIP_DRAM, W1->load_sisters = true
  │
  ├─ 2. Sister Batch Formation (src/hardware/linear_impl.cpp:50-56)
  │      batch_tensors = [W1_gate, W3_up, W2_down]
  │      Total Batch Size = 29,360,128 B * 3 = 88,080,384 B
  │
  ├─ 3. Off-Chip DRAM Issue (src/hardware/layer_impl.cpp:61-72)
  │      Translates addresses via offchip_mmap_controller (RoBaRaCoCh)
  │      Dispatches to Ramulator 2 via offchip_dram_interface->HandleRequest(...)
  │
  ├─ 4. Transfer Timing Completion (src/hardware/linear_impl.cpp:62-79)
  │      Off-chip duration returned: ~3,901,290 ns (3.901 ms)
  │      Logs: W1, W3, W2 constituent rows + all_weights row to expert_load.csv
  │
  ├─ 5. Cache Insertion (src/hardware/device.cpp:300-332)
  │      Device::mark_expert_resident(layer=0, expert=15, size=88080384)
  │      resident_bytes += 88,080,384 B
  │      Appends {0, 15, 88080384} to lru_expert_list (MRU position)
  │      Logs: "0,15,INSERT,88080384" to expert_cache.csv
  │
  ├─ 6. Sister Weight Access: W3_up (src/module/linear.cpp:52)
  │      Device::is_expert_resident(0, 15) -> HIT
  │      Moves node to MRU; logs: "0,15,HIT,88080384" to expert_cache.csv
  │      W3->weight_target remains HBM; memory_cost = 0 ns
  │      Logs: W3 hit row to expert_load.csv (duration = 0)
  │
  ├─ 7. Sister Weight Access: W2_down (src/module/linear.cpp:52)
  │      Device::is_expert_resident(0, 15) -> HIT
  │      Moves node to MRU; logs: "0,15,HIT,88080384" to expert_cache.csv
  │      W2->weight_target remains HBM; memory_cost = 0 ns
  │      Logs: W2 hit row to expert_load.csv (duration = 0)
  │
  └─ 8. Eviction Check (src/hardware/device.cpp:317)
         Checked during mark_expert_resident:
         while (resident_bytes + expert_size > cache_capacity_bytes) { pop_front(); ... }
         Result: resident_bytes (40.16 GB) < 80 GB -> ZERO EVICTIONS
```

#### Tensor Uniformity:
The current baseline uses **the identical batched path for all routed expert weights**: all three constituent matrices ($W_1, W_3, W_2$) are always batched into a single 88.08 MB DDR5 burst upon the initial $W_1$ miss, ensuring that $W_3$ and $W_2$ always hit HBM.

---

### Question 4: Where is the Main `.log`?

#### 4.1 Verified Absolute Path of the Main Runtime Log
```
/home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/logs/full_baseline_run.log
```
- **Size:** 10,407 bytes
- **Line Count:** 122 lines
- **Status:** Verified present on disk.

#### 4.2 Exact Creation Command
In `scripts/run_full_baseline.sh`:
- Banners, PID, and 60-second progress monitoring are logged via:
  ```bash
  echo "..." | tee -a "$LOG_FILE"
  ```
- The simulation executable's execution output is captured at line 78 via:
  ```bash
  "$EXECUTABLE" >> "$LOG_FILE" 2>&1 &
  ```

#### 4.3 Output Stream Redirection
`>> "$LOG_FILE" 2>&1` redirects **both standard output (`stdout`) and standard error (`stderr`)** into the log file in append mode.

#### 4.4 Taxonomy of Experiment Artifacts

| Category | Relative Path | Size | Description |
| :--- | :--- | :---: | :--- |
| **Main Runtime Log** | `logs/full_baseline_run.log` | 10,407 B | Complete console stdout/stderr and 60-second progress heartbeats |
| **Build Log** | `logs/build.log` | 563 B | Compilation and linking output of `test_deepseekv3_full_baseline` |
| **Runner Status** | `logs/ds5c11_completion_status.txt` | 1,708 B | Machine-readable runner metadata, timestamps, peak RSS, and file checksums |
| **Simulator Status** | `raw/ds5c10_completion_status.txt` | 555 B | Hardware simulated timers, line counts, and C++ assertion verifications |
| **Raw Telemetry: Routing** | `raw/expert_routing.csv` | 11,355 B | 912 routing choices (456 Pass 1 + 456 Pass 2 replay) |
| **Raw Telemetry: Load** | `raw/expert_load.csv` | 464,179 B | 6,384 per-tensor and batch DDR5/HBM transfer records |
| **Raw Telemetry: Cache** | `raw/expert_cache.csv` | 72,902 B | 3,192 cache events (456 MISS, 456 INSERT, 2280 HIT, 0 EVICT) |
| **Processed Results** | `processed/summary_metrics.json` | 1,833 B | Validated aggregate metrics across architecture, host, and DRAM timers |
| **Processed Results** | `processed/summary_metrics.csv` | 2,118 B | Tabular key-value summary metrics |
| **Processed Results** | `processed/per_layer_metrics.csv` | 2,868 B | Per-layer routing selections, off-chip bytes, and DDR5 durations |

#### 4.5 Best Starting Point for Diagnostic Checking
1. **For Runtime Errors, Heartbeats, and Output:**  
   [`logs/full_baseline_run.log`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/logs/full_baseline_run.log)  
   Commands to inspect:
   ```bash
   head -n 25 reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/logs/full_baseline_run.log
   tail -n 25 reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/logs/full_baseline_run.log
   ```
2. **For Instant Pass/Fail, Timestamps, and Peak Memory:**  
   [`logs/ds5c11_completion_status.txt`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/logs/ds5c11_completion_status.txt)
3. **For Cycle-Accurate Simulated DRAM Hardware Timers:**  
   [`raw/ds5c10_completion_status.txt`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/raw/ds5c10_completion_status.txt)

---

### Question 5: Reconcile the Evidence

#### 5.1 Why `build/config.yaml` Does Not Control This Experiment
`build/config.yaml` is copied by CMake (`CMakeLists.txt:30`) to provide configuration for the general test executable `./build/run` (`eval/test.cpp`). The full baseline ran `./build/test_deepseekv3_full_baseline` (`eval/test_deepseekv3_full_baseline.cpp`), which contains no YAML parser code and never accesses `config.yaml`.

#### 5.2 Actual Sources of Effective Parameters

| Setting | Actual Effective Source | Configuration Format | Hardcoded vs. Configured |
| :--- | :--- | :--- | :--- |
| **Model Parameters** | `eval/test_deepseekv3_full_baseline.cpp:66-81` | C++ Struct Assignment | **Hardcoded in C++** |
| **FP16 Precision** | `eval/test_deepseekv3_full_baseline.cpp:77` (`precision_byte = 2`) | C++ Assignment | **Hardcoded in C++** |
| **HBM Timing Configuration** | `dram_config_HBM3_80GB.yaml` (repo root) | Ramulator 2 YAML | Hardcoded path (`device.cpp:35`) loading YAML |
| **Off-Chip DDR5 Configuration**| `config_snapshot/dram_config_DDR5_retry.yaml` | Ramulator 2 YAML | **Configured via YAML** (passed by `OFFCHIP_DRAM_CFG`) |
| **DDR5 Address Mapping** | `src/dram/memory_config.h` (`ddr5_32gb_x16_2ch`) | C++ Struct | **Hardcoded in C++** |
| **Cache Capacity** | `eval/test_deepseekv3_full_baseline.cpp:100` (`80 GB`) | C++ Assignment | **Hardcoded in C++** |
| **Cache LRU Behaviour** | `src/hardware/device.cpp:269-332` | C++ Implementation | **Hardcoded in C++** |
| **Workload & Pass Count** | `eval/test_deepseekv3_full_baseline.cpp:114,117,152` | C++ Execution Sequence | **Hardcoded in C++** |

---

## 3. Source-File and Line-Number Evidence Table

| Component | Source File | Exact Line(s) | Function / Context | Description |
| :--- | :--- | :---: | :--- | :--- |
| **HBM Path Selection** | `src/hardware/device.cpp` | 34–40 | `Device::Device` | Hardcodes path `./dram_config_HBM3_80GB.yaml` and executes `YAML::LoadFile` |
| **HBM DRAM Interface** | `src/hardware/device.cpp` | 59 | `Device::Device` | Instantiates `dram_interface = DRAMInterface::Create(dram_cfg_path, 0.76923)` |
| **DDR5 Config Read** | `eval/test_deepseekv3_full_baseline.cpp` | 55–56 | `run_full_baseline` | Reads `std::getenv("OFFCHIP_DRAM_CFG")` |
| **DDR5 DRAM Interface** | `src/hardware/device.cpp` | 72 | `Device::Device` | Instantiates `offchip_dram_interface = DRAMInterface::Create(offchip_cfg, 0.625)` |
| **DDR5 Bus Geometry** | `src/dram/memory_config.h` | 27–42 | Global Struct | Defines `ddr5_32gb_x16_2ch` geometry (2ch, 2rk, 4bg, 4bk, 64col, 10.48M rows) |
| **DDR5 Address Translation**| `src/dram/mmap_controller.cpp` | 21–37 | `MMapController::getAddrVec` | Maps logical address to `Ramulator::AddrVec_t` (`RoBaRaCoCh`) |
| **80 GB Cache Budget** | `eval/test_deepseekv3_full_baseline.cpp` | 100 | `run_full_baseline` | Sets `device->cache_capacity_bytes = 80000000000LL;` |
| **Cache Lookup** | `src/module/linear.cpp` | 52 | `Linear::execution` | Calls `device->is_expert_resident(layer_id, A->expert_id)` |
| **Cache Hit / Miss Logic**| `src/hardware/device.cpp` | 269–298 | `Device::is_expert_resident` | Probes hashmap, moves to MRU on hit, logs to `expert_cache.csv` |
| **Sister Batch Grouping**| `src/hardware/linear_impl.cpp` | 50–56 | `LinearImpl::execution` | Aggregates $W_1, W_3, W_2$ into `batch_tensors` on cold miss |
| **Ramulator Batch Issue** | `src/hardware/layer_impl.cpp` | 58–72 | `issueRamulatorBatch` | Creates requests and calls `device->run_ramulator_batch` |
| **Cache Insertion** | `src/hardware/linear_impl.cpp` | 70 | `LinearImpl::execution` | Calls `device->mark_expert_resident` on DDR5 transfer completion |
| **LRU Eviction Loop** | `src/hardware/device.cpp` | 317–324 | `Device::mark_expert_resident` | Evicts front of `lru_expert_list` if `resident_bytes + expert_size > capacity` |
| **FP16 Override** | `eval/test_deepseekv3_full_baseline.cpp` | 77 | `run_full_baseline` | Hardcodes `model_config.precision_byte = 2; // FP16` |
| **Runtime Log Command** | `scripts/run_full_baseline.sh` | 78 | Script Execution | Executes `"$EXECUTABLE" >> "$LOG_FILE" 2>&1 &` |

---

## 4. Text-Based Runtime Call-Flow Diagram

```
[scripts/run_full_baseline.sh]
   │
   ├─ export OFFCHIP_DRAM_CFG="${EXP_DIR}/config_snapshot/dram_config_DDR5_retry.yaml"
   ├─ export TRACE_DIR="${EXP_DIR}/raw"
   └─ ./build/test_deepseekv3_full_baseline >> logs/full_baseline_run.log 2>&1
         │
         ▼
[eval/test_deepseekv3_full_baseline.cpp]
   │
   ├─ main()
   │    ├─ Audit DDR5 address space: MMapController::Create(ddr5_32gb_x16_2ch)
   │    └─ run_full_baseline(num_layers=60, moe_layers=57, trace_dir="raw")
   │         │
   │         ├─ Construct SystemConfig (gpu_gen="A100", getenv("OFFCHIP_DRAM_CFG"))
   │         ├─ Construct ModelConfig = deepseekV3 (precision_byte=2 [FP16], num_layers=60)
   │         ├─ Instantiate Cluster & Device
   │         │     │
   │         │     ▼
   │         │  [src/hardware/device.cpp]
   │         │     ├─ dram_cfg_path = "./dram_config_HBM3_80GB.yaml"
   │         │     ├─ dram_interface = DRAMInterface::Create(dram_cfg_path, 0.76923)
   │         │     ├─ offchip_dram_interface = DRAMInterface::Create(OFFCHIP_DRAM_CFG, 0.625)
   │         │     └─ cache_capacity_bytes = 80000000000 B (80 GB)
   │         │
   │         ├─ Instantiate Model Graph (FeedForward3Way, w1->sister_weights={w3, w2})
   │         │
   │         ├─ Pass 1: Cold Start Execution
   │         │    └─ cluster->run(metadata_pass1)
   │         │         │
   │         │         ▼
   │         │      [src/module/linear.cpp]
   │         │         ├─ is_expert_resident(layer, expert) -> MISS
   │         │         └─ A->weight_target = OFFCHIP_DRAM, A->load_sisters = true
   │         │              │
   │         │              ▼
   │         │           [src/hardware/linear_impl.cpp]
   │         │              ├─ batch_tensors = [W1, W3, W2] (88.08 MB)
   │         │              ├─ issueRamulatorBatch(..., MemoryTarget::OFFCHIP_DRAM)
   │         │              │    └─ offchip_dram_interface->HandleRequest(...)
   │         │              │         └─ DDR5 latency returned: ~3.901 ms
   │         │              └─ mark_expert_resident(layer, expert, 88080384)
   │         │                   └─ INSERT into resident_experts, resident_bytes += 88MB
   │         │
   │         └─ Pass 2: Warm Replay Execution
   │              └─ cluster->run(metadata_pass1 [cur_layer reset to 0])
   │                   │
   │                   ▼
   │                [src/module/linear.cpp]
   │                   ├─ is_expert_resident(layer, expert) -> HIT
   │                   └─ A->weight_target = HBM, memory_cost = 0 ns
   │                        └─ DDR5 Delta = 0.000 ns (100% Cache Hits)
```

---

## 5. Unresolved Questions and Implementation Gaps

1. **Zero-Latency HBM Hit Assumption:**
   - In `linear_impl.cpp:77`, HBM cache hits are assigned `memory_cost = 0`. While this isolates the secondary DDR5 DRAM bus latency cleanly, it ignores on-chip HBM read latency (~100–200 ns) for hit weights.
2. **Fixed Dual-Channel Bus Assumption:**
   - Ramulator 2's DDR5 configuration is fixed at 2 channels (`channel: 2`), reflecting a standard dual-channel client/workstation DDR5 channel bus. High-end CXL memory expanders or host server sockets frequently support 8 or 16 DDR5 channels; simulating higher channel counts would require generating a new configuration YAML.
3. **Hardcoded Repository Root Path for HBM:**
   - `src/hardware/device.cpp:35` relies on the process working directory being the repository root to find `./dram_config_HBM3_80GB.yaml`. If the binary were executed from a different working directory, `YAML::LoadFile` would throw `YAML::BadFile`.
