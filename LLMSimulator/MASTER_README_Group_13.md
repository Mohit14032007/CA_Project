# Master Reference: Duplex & Off-Chip DRAM Reproduction Study

**Target Path:** `reproduction_A100/with_off_chip_dram/README.md`  
**Parent Repository:** `CA_Project/LLMSimulator/`  
**Host Hardware Target:** NVIDIA A100-SXM4-80GB GPU Baseline & Secondary Off-Chip DDR5 DRAM Memory System  
**Simulator Core:** LLMSimulator (Cycle-accurate GPU & Ramulator 2 DDR5/HBM3 Subsystem)  
**Document Classification:** Authoritative Master Reproduction & Operational Guide  

---

## 1. Project Overview and Experiment Boundaries

This repository investigates memory tiering, disaggregated off-chip DRAM architectures, and Logic-PIM optimizations for Large Language Model (LLM) Mixture-of-Experts (MoE) inference. Across modern MoE architectures (such as Mixtral 8x7B and DeepSeek-V3), parameter scale drastically exceeds on-package GPU memory (HBM), requiring expert weights to be offloaded to higher-capacity external memory tiers (e.g., host-attached DDR5 or LPDDR5X) or accelerated via near-memory processing.

To maintain strict scientific isolation, this repository separates experiments into **two primary experiment groups**:

```
                              ┌─────────────────────────────────────────────────────────┐
                              │            LLMSimulator Master Repository               │
                              └─────────────────────────────────────────────────────────┘
                                                           │
                        ┌──────────────────────────────────┴──────────────────────────────────┐
                        │                                                                     │
                        ▼                                                                     ▼
┌─────────────────────────────────────────────────┐   ┌─────────────────────────────────────────────────┐
│       Group 1: Duplex A100 Reproduction         │   │   Group 2: DeepSeek-V3 Off-Chip DRAM Workloads  │
│             (reproduction_A100/)                │   │            (deepseekv3_mixtral_style)           │
├─────────────────────────────────────────────────┤   ├─────────────────────────────────────────────────┤
│ • Model: Mixtral 8x7B (47B MoE)                 │   │ • Model: DeepSeek-V3 (60L: 3 dense, 57 MoE)     │
│ • Hardware: 4x A100-80GB HBM (NVLink 3)         │   │ • Hardware: 1x A100-80GB HBM + 2.75TB DDR5-3200 │
│ • Focus: Logic-PIM co-processing (Duplex),      │   │ • Focus: Multi-token sequential decode loop     │
│   Parallel Execution (PE) load balancing,       │   │   (Profile B), HBM LRU cache residency,        │
│   and Expert Tensor Parallelism (ET=4)          │   │   Pilot (1-Step) & Full (5-Step) runs,          │
│ • Status: Verified / Complete                   │   │   30 publication figures & 41 research datasets │
│                                                 │   │ • Status: Verified / Complete                   │
└─────────────────────────────────────────────────┘   └─────────────────────────────────────────────────┘
```

### Strict Non-Interference Guarantees
1. **Source Independence:** Group 1 relies strictly on the native GPU/Logic-PIM cluster simulation path (`build/run`). Group 2 activates the dual-memory subsystem (`offchip_dram_interface` backed by Ramulator 2 DDR5) via the specialized sequential decoding driver (`run_deepseekv3_mixtral_style`).
2. **Artifact Isolation:** Each experiment writes exclusively to its dedicated directory tree (`reproduction_A100/results/` or `deepseekv3_mixtral_style_20261011/`). No experiment modifies, renames, or overwrites another experiment's outputs.
3. **Telemetry Isolation:** Output paths are dynamically controlled via YAML configurations and the `TRACE_DIR` / `OFFCHIP_DRAM_CFG` environment variables, preventing trace cross-contamination.

---

## 2. Repository and Directory Map

```text
LLMSimulator/
├── CMakeLists.txt                                  # Root CMake build definition
├── Makefile                                        # Wrapper Makefile (if present)
├── dram_config_HBM3_80GB.yaml                      # On-chip 80GB HBM3 Ramulator 2 configuration
├── dram_config_DDR5_retry.yaml                     # Off-chip DDR5 Ramulator 2 configuration (2.75 TB backing store)
├── build/                                          # Compiled binaries and libraries
│   ├── run                                         # Main simulator executable (Group 1)
│   ├── run_deepseekv3_mixtral_style                # DeepSeek-V3 sequential decoding driver (Group 2)
│   ├── test_deepseekv3_full_baseline               # DeepSeek-V3 full-model 2-pass test driver
│   └── libllm_system.so                            # Core simulator shared runtime library
├── eval/                                           # Evaluation drivers and unit tests
│   ├── CMakeLists.txt                              # Executable build targets
│   ├── test.cpp                                    # Source for build/run (Group 1)
│   ├── test_deepseekv3_mixtral_style.cpp           # Source for Group 2 driver
│   ├── test_deepseekv3_full_baseline.cpp           # Source for two-pass baseline test
│   ├── test_ddr5_mapping.cpp                       # DDR5 channel/rank/bank address mapping test
│   └── test_lru.cpp                                # HBM LRU cache residency and eviction test
├── src/                                            # Simulator source code
│   ├── dram/                                       # Ramulator 2 wrapper, MMapController, DRAM interfaces
│   ├── hardware/                                   # Device, Cluster, Linear, Attention, Layer execution
│   ├── model/                                      # LLM definitions, model configurations (Mixtral, DeepSeek-V3)
│   ├── module/                                     # Module graph, Expert FFN, Attention, Tensor tracking
│   └── scheduler/                                  # Sequence scheduler, Zipfian/Uniform routing generator
├── reproduction_A100/                              # GROUP 1: Original Duplex A100 Reproduction
│   ├── README.md                                   # A100 Duplex project specification
│   ├── configs/                                    # Immutable execution configurations
│   │   ├── a100_gpu_baseline.yaml                  # GPU Baseline (TP=4, DP=1, PE=off, ET=1)
│   │   ├── a100_duplex.yaml                        # Plain Duplex (GPU+LOGIC, PE=off, ET=1)
│   │   ├── a100_duplex_pe.yaml                     # Duplex + PE (GPU+LOGIC, PE=on, ET=1)
│   │   └── a100_duplex_pe_et.yaml                  # Duplex + PE + ET (GPU+LOGIC, PE=on, ET=4)
│   └── results/                                    # Group 1 simulation outputs
│       ├── a100_gpu_baseline/                      # Baseline CSVs, device logs, and traces
│       ├── a100_duplex/                            # Plain Duplex CSVs, device logs, and traces
│       ├── a100_duplex_pe/                         # Duplex+PE CSVs, device logs, and traces
│       └── a100_duplex_pe_et/                      # Duplex+PE+ET CSVs, device logs, and traces
└── reproduction_A100/with_off_chip_dram/           # GROUP 2: Off-Chip DRAM Extensions
    ├── README.md                                   # This master reference document
    ├── configs/                                    # Early exploration configs (b0_full.yaml, b1_full.yaml)
    ├── results/                                    # Early Mixtral off-chip results (b0_phase5, b1_full, etc.)
    └── deepseekv3/                                 # DeepSeek-V3 Sub-tree
        ├── configs/                                # Diagnostic configs (debug_ds5.yaml, ds5c3_smoke_test.yaml)
        ├── manifests/                              # Engineering audit logs (DS-5C-0 through DS-5C-10)
        └── experiments/
            └── deepseekv3_mixtral_style_20261011/  # GROUP 2: Pilot & 5-Step Sequential Decoding Workload
                ├── README.md                       # Experiment overview and configuration reference
                ├── configs/                        # deepseekv3_b1_pilot_1step.yaml, deepseekv3_b1_full_run_5step.yaml
                ├── pilot_1step/                    # 1-step validation run artifacts
                ├── full_run_5step_20261011/        # Completed authoritative 5-step simulation
                │   ├── logs/                       # run.log (21,159 host seconds), preflight_summary.txt
                │   ├── traces/                     # Raw telemetry (expert_routing, expert_load, expert_cache)
                │   ├── csv/                        # Summary decode CSV
                │   ├── analysis_ds6c5/             # Audit, routing analysis, and research validity report
                │   └── analysis_ds6c6/             # 41 CSV datasets, 30 publication figures, detailed guide
                └── scripts/                        # run_mixtral_style_experiment.sh, generate_figures.py
```

---

## 3. Prerequisites and Build Setup

### System & Environment Requirements
- **OS:** Linux (Ubuntu 20.04/22.04 LTS or WSL2 on Linux kernel 5.15+)
- **Compiler:** `g++` or `clang++` supporting C++17
- **Build System:** CMake $\ge 3.14$, GNU Make
- **Libraries:** `yaml-cpp`, `spdlog` (bundled in `src/dram/ramulator2/ext/`), POSIX threads
- **Python Environment:** Python $\ge 3.8$ with `pandas`, `numpy`, `matplotlib`, `scipy`
- **Minimum Free Host RAM:** 2.5 GB (watchdog aborts if available RAM drops below 2,560 MB)
- **Minimum Free Disk Space:** 5.0 GB

### Authoritative Build Commands
Execute from the repository root (`CA_Project/LLMSimulator`):

```bash
# 1. Create build directory and generate Makefiles
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..

# 2. Compile simulator core library and all evaluation binaries
make -j$(nproc)
cd ..
```

### Verified Build Targets
- `build/libllm_system.so`: Core simulation engine containing `Device`, `Cluster`, `Model`, `Scheduler`, and memory interfaces.
- `build/run`: Standard multi-GPU/Logic-PIM execution harness (`eval/test.cpp`).
- `build/run_deepseekv3_mixtral_style`: Group 2 multi-token sequential decoding harness (`eval/test_deepseekv3_mixtral_style.cpp`).
- `build/test_deepseekv3_full_baseline`: DeepSeek-V3 two-pass memory baseline verification driver.
- `build/test_lru`: Unit test validating HBM LRU eviction and residency invariants.
- `build/test_ddr5_mapping`: Unit test verifying channel, rank, bankgroup, bank, row, and column mapping logic.

---

## 4. Quick Reference: Consolidated Run-Command Table

| Experiment | Configuration File | Build Target | Authoritative Run Command | Output Directory | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A100 GPU Baseline** | `reproduction_A100/configs/a100_gpu_baseline.yaml` | `build/run` | `./build/run reproduction_A100/configs/a100_gpu_baseline.yaml` | `reproduction_A100/results/a100_gpu_baseline/` | **VERIFIED** (Completed) |
| **A100 Plain Duplex** | `reproduction_A100/configs/a100_duplex.yaml` | `build/run` | `./build/run reproduction_A100/configs/a100_duplex.yaml` | `reproduction_A100/results/a100_duplex/` | **VERIFIED** (Completed) |
| **A100 Duplex + PE** | `reproduction_A100/configs/a100_duplex_pe.yaml` | `build/run` | `./build/run reproduction_A100/configs/a100_duplex_pe.yaml` | `reproduction_A100/results/a100_duplex_pe/` | **VERIFIED** (Completed) |
| **A100 Duplex + PE + ET** | `reproduction_A100/configs/a100_duplex_pe_et.yaml` | `build/run` | `./build/run reproduction_A100/configs/a100_duplex_pe_et.yaml` | `reproduction_A100/results/a100_duplex_pe_et/` | **VERIFIED** (Completed) |
| **DeepSeek-V3 Pilot (1-Step)** | `.../deepseekv3_mixtral_style_20261011/configs/deepseekv3_b1_pilot_1step.yaml` | `build/run_deepseekv3_mixtral_style` | `./reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/scripts/run_mixtral_style_experiment.sh configs/deepseekv3_b1_pilot_1step.yaml` | `.../deepseekv3_mixtral_style_20261011/pilot_1step/` | **VERIFIED** (Completed) |
| **DeepSeek-V3 Full Run (5-Step)** | `.../deepseekv3_mixtral_style_20261011/configs/deepseekv3_b1_full_run_5step.yaml` | `build/run_deepseekv3_mixtral_style` | `./reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/scripts/run_mixtral_style_experiment.sh configs/deepseekv3_b1_full_run_5step.yaml` | `.../deepseekv3_mixtral_style_20261011/full_run_5step_20261011/` | **VERIFIED** (Completed) |

---

## 5. Section A — Original Duplex Reproduction (A100 Target)

### Objective & Architectural Principles
The Duplex architecture evaluates offloading memory-bound MoE operations to Logic-PIM (Processing-in-Memory) co-processors embedded directly within 3D-stacked DRAM while compute-bound operations remain on the host GPU. The A100 reproduction study isolates this mechanism under the architectural constraints of an NVIDIA A100-SXM4-80GB system (312 TFLOP/s FP16 compute, 2.039 TB/s HBM bandwidth, NVLink 3 at 150 GB/s bidirectional bandwidth and 3.0 $\mu$s latency).

### Architectural Differences Across Variants

```
1. GPU Baseline:
   [Host GPU] <====== NVLink 3 ======> [Host GPU]
     ├── Attention: GPU
     └── MoE FFN:   GPU (serialized across Top-2 routed experts)

2. Plain Duplex (GPU + LOGIC, PE=off, ET=1):
   [Host GPU] <====== NVLink 3 ======> [Host GPU]
     ├── Attention: GPU
     └── MoE FFN:   Static partitioning (Expert computation assigned to Logic-PIM)

3. Duplex + PE (Parallel Execution, ET=1):
   [Host GPU] <====== NVLink 3 ======> [Host GPU]
     ├── Attention: GPU
     └── MoE FFN:   Dynamic Workload Balancer splits sub-batches across GPU and Logic-PIM
                    concurrently to prevent Logic-PIM queue saturation.

4. Duplex + PE + ET (Expert Tensor Parallelism, ET=4):
   [Host GPU] <====== NVLink 3 ======> [Host GPU]
     ├── Attention: GPU (Tensor Parallelism TP=4)
     └── MoE FFN:   Expert weights sharded across all 4 devices (ET=4), reducing per-device
                    expert footprint by 4x and overlapping inter-device all-reduce.
```

### Workload & Simulation Parameters
- **Model:** Mixtral 8x7B (`model_name: mixtral`), 32 Transformer layers, 8 routed experts per layer, Top-2 selection.
- **Workload Mode:** Pure autoregressive decode mode (`prefill_mode: off`, `decode_mode: on`).
- **Input / Output Context:** Input length = 2048 (`input_len: 2048`), Output length = 128 (`output_len: 128`).
- **Batch Size:** 32 concurrent sequences (`max_batch_size: 32`).
- **Precision:** FP16 (2 Bytes/weight, `precision_byte: 2`).
- **Execution Timing:** 5 measured decode steps (`iter: 5`) following the standard 10,000-request scheduler warmup queue.

### Execution Commands (Copy-Paste)
```bash
# Ensure simulator is built
cd /home/mohman/ca_project/LLMSimulator
make -C build -j$(nproc)

# 1. Run GPU Baseline
./build/run reproduction_A100/configs/a100_gpu_baseline.yaml > reproduction_A100/results/a100_gpu_baseline/logs/concise_run.log 2>&1

# 2. Run Plain Duplex
./build/run reproduction_A100/configs/a100_duplex.yaml > reproduction_A100/results/a100_duplex/logs/concise_run.log 2>&1

# 3. Run Duplex + PE
./build/run reproduction_A100/configs/a100_duplex_pe.yaml > reproduction_A100/results/a100_duplex_pe/logs/concise_run.log 2>&1

# 4. Run Duplex + PE + ET
./build/run reproduction_A100/configs/a100_duplex_pe_et.yaml > reproduction_A100/results/a100_duplex_pe_et/logs/concise_run.log 2>&1
```

### Generated Artifacts & Schemas
Each run populates its respective directory under `reproduction_A100/results/<variant>/`:
1. `csv/mixtral_synthesis_2048_128_..._decode.csv`: Cycle-accurate timing and energy breakdown per decode step (see Section 9 for header schema).
2. `raw/device_0` ... `device_3`: Raw timeline device trees capturing sub-operation start/end timestamps.
3. `traces/expert_routing.csv`: Chronological record of all 10,240 routing decisions ($32\text{ batch} \times 2\text{ top-k} \times 32\text{ layers} \times 5\text{ steps}$).
   - Schema: `request_id,token_id,layer_id,topk_rank,expert_id`
4. `traces/processor_trace.csv`: Chronological operator execution logs on GPU vs Logic-PIM.
   - Schema: `device_id,layer_id,operation,processor,start_time,end_time,duration`

---

## 6. Section B — DeepSeek Off-Chip DRAM Pilot & 5-Step Workload (`deepseekv3_mixtral_style_20261011`)

### Objective & Architecture
The `deepseekv3_mixtral_style_20261011` directory evaluates sequential multi-token autoregressive generation using the controlled **Profile B** sequential decoding loop (`use_hitting_queue: false`), which bypasses the 10,000-request warmup queue to evaluate true token-by-token state transitions under a disaggregated memory hierarchy.

The GPU retains an 80 GB HBM LRU cache for active working sets, while the complete 14,592 routed expert inventory (1.285 TB in FP16) resides in a 2.75 TB host-attached DDR5 memory subsystem.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           A100 Accelerator Node                                 │
│                                                                                 │
│   ┌───────────────────────────┐         ┌───────────────────────────────────┐   │
│   │   80 GB HBM3 LRU Cache    │         │       Compute Subsystem           │   │
│   │                           │         │                                   │   │
│   │ • 1 Shared Expert (Perm)  │ ◄─────► │ • Attention Engines (Flash-MLA)   │   │
│   │ • Active Routed Experts   │         │ • Tensor Cores (312 TFLOP/s FP16) │   │
│   │ • KV Cache & Activations  │         │                                   │   │
│   └─────────────▲─────────────┘         └─────────────────▲─────────────────┘   │
└─────────────────┼─────────────────────────────────────────┼─────────────────────┘
                  │                                         │
                  │ Cache Miss Transfer (Synchronous)       │
                  │                                         │
┌─────────────────▼─────────────────────────────────────────▼─────────────────────┐
│                 Secondary Off-Chip Memory Controller (DDR5)                     │
│                                                                                 │
│ • Ramulator 2 DDR5-3200 Subsystem (2 Channels, 32 Banks, 10.48M Rows)           │
│ • Production GPU Address Mapper (Linear -> Channel/Rank/BG/Bank/Row/Col)        │
│ • Backing Store: 2.75 TB Physical Capacity (Holds 14,592 Routed Experts)        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Complete Source-Code Memory Call Chain
The actual memory path implemented in LLMSimulator executes through the following verified call sequence:

```text
1. Target Selection & Cache Query
   File: src/module/linear.cpp (lines 46–57)
   Function: Linear::calculate(...)
   Detail: A->weight_target defaults to MemoryTarget::HBM.
           If A->is_expert_weight and input->getSize() > 0:
             Query device->is_expert_resident(layer_id, A->expert_id).

2. HBM LRU Hit / Miss Evaluation
   File: src/hardware/device.cpp (lines 272–298)
   Function: Device::is_expert_resident(int layer_id, int expert_id)
   Detail: Searches resident_experts hash map for key {layer_id, expert_id}.
           • HIT:  Promotes node to MRU (tail of lru_expert_list).
                   Appends "layer,expert,HIT,resident_bytes" to expert_cache.csv.
                   Returns true -> A->weight_target remains HBM.
           • MISS: Appends "layer,expert,MISS,resident_bytes" to expert_cache.csv.
                   Returns false -> sets A->weight_target = MemoryTarget::OFFCHIP_DRAM
                   and sets A->load_sisters = true.

3. Concurrent Sister-Weight Batching
   File: src/hardware/linear_impl.cpp (lines 48–60)
   Function: LinearLayerImpl::calculateImpl(...)
   Detail: When load_sisters is true, packages Gate (W1), Up (W3), and Down (W2)
           tensors into batch_tensors (total size = 88.08 MB).
           Dispatches via issueRamulatorBatch(...) targeting MemoryTarget::OFFCHIP_DRAM.

4. DDR5 Address Generation
   File: src/hardware/device.cpp (lines 121–138) & src/dram/mmap_controller.cpp
   Function: Device::get_addr_from_tensor(...) & MMapController::get_address(...)
   Detail: Maps linear tensor byte offset to DDR5 hardware coordinates:
           channel, rank, bankgroup, bank, row, column.
           DDR5-3200 geometry: 2 channels, 10,485,760 rows, 262,144 bytes/row step.

5. Ramulator 2 Memory Simulation
   File: src/dram/dram_interface.cpp & src/dram/ramulator2/
   Function: DRAMInterface::HandleRequest(...)
   Detail: Injects batch DRAM read requests into Ramulator 2 DDR5 system.
           Models bank conflicts, tRCD, tRP, tCAS, and bus contention.
           Updates offchip_dram_interface->time.

6. Residency Tracking & Eviction
   File: src/hardware/device.cpp (lines 303–332)
   Function: Device::mark_expert_resident(int layer_id, int expert_id, long long expert_size)
   Detail: While (resident_bytes + expert_size > cache_capacity_bytes):
             Evicts LRU expert from front of lru_expert_list.
             Appends "layer,expert,EVICT,resident_bytes" to expert_cache.csv.
           Appends newly loaded expert to MRU position.
           Appends "layer,expert,INSERT,resident_bytes" to expert_cache.csv.

7. Telemetry & Hardware Clock Advancement
   File: src/hardware/linear_impl.cpp (lines 61–79)
   Function: LinearLayerImpl::calculateImpl(...)
   Detail: exec_status += weight_status.
           If OFFCHIP_DRAM: adds weight_memory_cost to simulated clock and records
           transfer duration in expert_load.csv.
           If HBM HIT: records memory_cost = 0 ns in expert_load.csv.
```

### Workload Profiles: Pilot vs Full 5-Step Run
1. **Pilot Run (`pilot_1step/`):** Exactly 1 decoding step (`iter: 1`, input len 2048, output len 4). Used for rapid end-to-end telemetry verification (~70 minutes host runtime).
2. **Full Run (`full_run_5step_20261011/`):** Exactly 5 sequential decoding steps (`iter: 5`, input len 2048, output len 10). Generates 5 consecutive tokens (tokens 2048, 2049, 2050, 2051, 2052) under steady-state LRU cache pressure (~5.88 hours host runtime).

### Architectural Parameters & Reconciliation
- **Total Layers:** 60 layers (3 dense layers 0–2, 57 MoE layers 3–59).
- **Experts per MoE Layer:** 256 routed experts + 1 shared expert.
- **Routing Strategy:** Top-8 routed experts activated per token.
- **Routing Generator:** Synthetic pseudorandom generation using `std::mt19937` with seed `777` (`skewness: 0.0`).
- **Precision:** FP16 (2 Bytes/weight).
- **HBM Cache Capacity:** 80 GB ($80,000,000,000$ bytes).
- **Backing Store:** 2.75 TB DDR5-3200 (10,485,760 rows $\times$ 262,144 bytes/row).

### Confirmed Limitation: `cur_layer` Modulo-60 Indexing Drift
> [!WARNING]
> **Source Code Audit Finding (`src/scheduler/sequence.cpp:239-240`):**  
> In `BatchedSequence::route_expert()`, the internal layer cursor advances after every MoE routing invocation:
> ```cpp
> cur_layer++;
> cur_layer %= num_layer;
> ```
> In DeepSeek-V3, `num_layer = 60`, but only 57 layers invoke `route_expert()` because the first 3 layers are dense. Consequently, after processing 1 token, `cur_layer` advances by $57 \pmod{60} = -3$.
> - Token 2048 routes layers `0, ..., 56`
> - Token 2049 routes layers `57, 58, 59, 0, ..., 53`
> - Token 2050 routes layers `54, ..., 50`
> 
> Across all 5 tokens, all 60 physical layer IDs appear in `expert_routing.csv` despite each token evaluating exactly 57 MoE stages. When analyzing sequential routing, grouping by **execution stage order (stages 1–57)** rather than raw `layer_id` preserves true topological alignment.

### Authoritative Execution Commands
```bash
cd /home/mohman/ca_project/LLMSimulator

# Fast syntax and topology preflight check (0.1s, no simulation)
./build/run_deepseekv3_mixtral_style --validate-config reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/configs/deepseekv3_b1_full_run_5step.yaml

# Run Pilot (1 Step, ~70 minutes host runtime)
./reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/scripts/run_mixtral_style_experiment.sh configs/deepseekv3_b1_pilot_1step.yaml

# Run Full 5-Step Workload (~5.88 host hours)
# NOTE: Do NOT rerun if full_run_5step_20261011/ already contains valid results!
./reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/scripts/run_mixtral_style_experiment.sh configs/deepseekv3_b1_full_run_5step.yaml
```

---

## 7. Execution Files & Deliverables Guide ("What, Where, and How You Get It")

This section provides an exhaustive reference explaining how the simulator binary executes, what files are produced upon completion, where each file is stored on the filesystem, and how the simulation pipeline produces them.

### 7.0 Master Summary Matrix: Execution Commands vs Deliverables

The table below provides a unified cross-reference of every command, the underlying execution file invoked, what outputs are generated, where they are placed, and how the simulation pipeline produces them:

| Command / Workflow | Execution File / Binary | What Will I Get? | Where Will I Get It? | How Will I Get It? |
| :--- | :--- | :--- | :--- | :--- |
| **Group 1 Reproduction**<br>`./build/run configs/a100_gpu_baseline.yaml` | `build/run`<br>(Compiled from [`eval/test.cpp`](file:///home/mohman/ca_project/LLMSimulator/eval/test.cpp)) | • 5-step decode metrics CSV (51 cols)<br>• Raw Gantt traces (`device_0..3`)<br>• Expert routing trace (10,240 rows)<br>• Processor timeline trace<br>• Console log | `reproduction_A100/results/<variant>/`<br>├── `csv/*.csv`<br>├── `raw/device_*`<br>├── `traces/expert_routing.csv`<br>├── `traces/processor_trace.csv`<br>└── `logs/concise_run.log` | C++ simulation loop warms up with 10,000 requests, executes 5 decode iterations on 4 simulated A100s, streams operator events to `std::ofstream`, and flushes summary CSV on completion. |
| **Group 2 Preflight Check**<br>`./build/run_deepseekv3_mixtral_style --validate-config <config>` | `build/run_deepseekv3_mixtral_style`<br>(Compiled from [`eval/test_deepseekv3_mixtral_style.cpp`](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp)) | • Terminal geometry audit report<br>• DRAM capacity validation (2.75 TB vs 1.285 TB)<br>• Exit code `0` | Standard output (`stdout`) / console | Reads YAML, computes Ramulator 2 DDR5 geometry formula without running cycle simulation; returns in <0.1 seconds. |
| **Group 2 Pilot Run (1-Step)**<br>`./scripts/run_mixtral_style_experiment.sh configs/pilot_1step.yaml` | `run_mixtral_style_experiment.sh`<br>wraps `build/run_deepseekv3_mixtral_style` | • 1-step decode metrics CSV<br>• 456 routing decisions trace<br>• 1,756 DDR5 weight transfer records<br>• 2,068 HBM cache transitions<br>• SHA-256 verifiable completion token | `.../deepseekv3_mixtral_style_20261011/pilot_1step/`<br>├── `csv/*_iter1_*decode.csv`<br>├── `traces/expert_routing.csv`<br>├── `traces/expert_load.csv`<br>├── `traces/expert_cache.csv`<br>└── `logs/completion_status.txt` | Watchdog shell script acquires `run.lock`, monitors RAM (>2.5 GB free), launches Profile B binary, and writes completion status with cryptographic hashes upon normal exit. |
| **Group 2 Full Run (5-Step)**<br>`./scripts/run_mixtral_style_experiment.sh configs/full_run_5step.yaml` | `run_mixtral_style_experiment.sh`<br>wraps `build/run_deepseekv3_mixtral_style` | • 5-step decode metrics CSV<br>• 2,280 routing decisions trace<br>• 8,780 DDR5 weight transfer records<br>• 10,341 HBM cache transitions<br>• 21,159s wall-clock log (`run.log`)<br>• Verifiable completion token | `.../deepseekv3_mixtral_style_20261011/full_run_5step_20261011/`<br>├── `csv/*_iter5_*decode.csv`<br>├── `traces/expert_routing.csv`<br>├── `traces/expert_load.csv`<br>├── `traces/expert_cache.csv`<br>└── `logs/completion_status.txt` | Executes 5 sequential autoregressive tokens (Profile B), queries 80 GB LRU cache per MoE layer, dispatches misses to DDR5 Ramulator 2, records real-time telemetry, and verifies via watchdog. |
| **Group 2 Offline Analysis**<br>`python3 scripts/generate_csv_datasets.py`<br>`python3 scripts/generate_figures.py`<br>`python3 scripts/generate_additional_correlation_figures.py` | Standalone Python 3 scripts in `analysis_ds6c6/scripts/` | • 41 curated research CSV datasets<br>• 30 publication PNG figures (300 DPI)<br>• 30 vector PDF figures<br>• 105-item manifest (`results_package_manifest.csv`)<br>• Figure explanation guide (`.md`) | `.../analysis_ds6c6/`<br>├── `csv/*.csv` (41 datasets)<br>├── `figures/png/*.png` (30 figures)<br>├── `figures/pdf/*.pdf` (30 figures)<br>├── `manifests/results_package_manifest.csv`<br>└── `figures/png/FIGURES_EXPLANATION_AND_METHODOLOGY.md` | Pure offline Python transformations: ingests raw simulation traces (`expert_routing.csv`, `expert_load.csv`, `expert_cache.csv`), computes statistical models, and renders matplotlib charts in ~5 seconds. |

### 7.1 Deep Dive: Execution Files and Internal Architecture

The simulator relies on two primary compiled binaries and one production watchdog runner script:

#### 1. Binary: `build/run`
- **Source Code:** [`eval/test.cpp`](file:///home/mohman/ca_project/LLMSimulator/eval/test.cpp) (compiled via `eval/CMakeLists.txt:1-6`)
- **Primary Role:** Standard multi-GPU and Logic-PIM cluster evaluation harness for **Group 1 (Mixtral Duplex)** experiments.
- **Internal Execution Flow:**
  1. **Configuration Ingestion:** Loads specified YAML configuration (e.g., `reproduction_A100/configs/a100_gpu_baseline.yaml`) via `YAML::LoadFile()`.
  2. **System Setup:** Sets GPU generation (`A100`), NVLink interconnect (Gen 3: 150 GB/s, 3.0 $\mu$s latency), and processor modes (`GPU` or `GPU+LOGIC`).
  3. **Trace Directory Resolution:** Automatically maps `log.gantt_directory` to `traces/` and sets the `TRACE_DIR` process environment variable.
  4. **Scheduler Warmup:** Populates the sequence scheduler and executes the 10,000-request `hittingQueue` warmup to simulate steady-state KV caching.
  5. **Iterative Simulation:** Calls `cluster->runIteration(total_iter, summary_file)`. For each step, it advances simulated device clocks across all 4 devices in parallel, evaluates Tensor Parallelism communication, and flushes output metrics to CSV.

#### 2. Binary: `build/run_deepseekv3_mixtral_style`
- **Source Code:** [`eval/test_deepseekv3_mixtral_style.cpp`](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp) (compiled via `eval/CMakeLists.txt:42-45`)
- **Primary Role:** High-precision, dual-memory sequential decoding simulator for **Group 2 (DeepSeek-V3)** experiments.
- **Internal Execution Flow:**
  1. **Preflight Geometry Check (`--validate-config`):**
     Calculates DDR5 physical geometry from `ddr5_32gb_x16_2ch` (2 channels, 10,485,760 rows, 262,144 bytes/row = 2.75 TB address space). Confirms that all 14,592 routed experts (1.285 TB in FP16) fit within 46.76% of DDR5 rows without overflow.
  2. **Subsystem Initialization:**
     - Creates the on-chip HBM3 Ramulator 2 instance (`dram_config_HBM3_80GB.yaml`).
     - Creates the off-chip DDR5 Ramulator 2 instance (`dram_config_DDR5_retry.yaml`).
     - Initializes the 80 GB HBM LRU Cache Manager inside `Device` (`cache_capacity_bytes = 80000000000LL`).
  3. **Sequential Autoregressive Loop (Profile B):**
     Bypasses the artificial 10,000-request warmup queue (`use_hitting_queue: false`). For each step $t \in [1, \text{iter}]$:
     - Scheduler pushes a dummy token sequence representing context length $2048 + t - 1$.
     - Executes 60 Transformer layers (Layers 0–2 dense, Layers 3–59 MoE).
     - At each MoE layer: queries `is_expert_resident()`. On miss: dispatches 88.08 MB batch read to DDR5 Ramulator 2, records latency, evicts LRU experts if cache $>80\text{ GB}$, and inserts newly loaded expert.
     - Advances simulated clocks: accumulates DDR5 stall latency and device execution time.
     - Flushes step metrics to summary CSV and records status in `completion_status.txt`.

#### 3. Runner Script: `scripts/run_mixtral_style_experiment.sh`
- **Source Code:** [`reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/scripts/run_mixtral_style_experiment.sh`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/scripts/run_mixtral_style_experiment.sh)
- **Primary Role:** Production safety harness with automated health monitoring and post-processing:
  - **Single-Run Lock Guard:** Enforces `run.lock` to prevent accidental concurrent simulation runs.
  - **Memory Safety Guard:** Continuously polls available host RAM every 5 seconds. If free RAM drops below **2,560 MB (2.5 GB)**, it cleanly terminates the simulator with `SIGTERM` to protect host stability.
  - **Execution Watchdog:** Monitors a 4-hour timeout window (`WATCHDOG_TIMEOUT=14400`).
  - **Progress Heartbeat:** Emits memory RSS and trace line counts every 30 seconds into `logs/run.log`.
  - **Automated Post-Processing:** Upon exit code 0, automatically triggers telemetry post-processing (`process_metrics.py`), sequential routing analysis (`analyze_results.py`), and publication plotting (`generate_figures.py`).

---

### 7.2 What Will I Get? (Deliverables & Output Artifacts)

Running these commands generates four distinct classes of scientific deliverables:

#### Class 1: Summary Performance Metrics (Decode CSV)
- **Group 1:** `mixtral_synthesis_2048_128_..._decode.csv` (1 row per decode step).
- **Group 2:** `deepseekV3_synthesis_2048_10_..._decode.csv` (1 row per decode step).
- **Contents:** Exact cycle-accurate breakdown across 51 hardware metrics:
  - `latency`: Total simulated step latency (ns).
  - `time`: Cumulative simulated timeline position (ns).
  - `act_energy`, `read_energy`, `write_energy`: On-chip HBM electrical power metrics (nJ).
  - `mac_energy`, `total_energy`: Total computational and memory energy.
  - Sub-operation timing: `qkv_gen`, `atten_gen`, `expert_ffn`, `communication`, `layernorm`.

#### Class 2: Chronological Hardware Traces (CSV)
- **`expert_routing.csv`:** Complete routing log recording which token accessed which expert at which layer.
  - Group 1: Exactly **10,240 records** ($32\text{ batch} \times 2\text{ top-k} \times 32\text{ layers} \times 5\text{ steps}$).
  - Group 2 (5-Step): Exactly **2,280 records** ($1\text{ batch} \times 8\text{ top-k} \times 57\text{ layers} \times 5\text{ steps}$).
- **`expert_load.csv` (Group 2):** Exact log of all 8,780 weight loading operations from off-chip memory:
  - Records tensor name (`W1_gate`, `W3_up`, `W2_down`, `all_weights`), transfer byte size, start/end time, load latency (ns), memory source (`OFFCHIP_DRAM` vs `HBM`), and status (`miss` vs `hit`).
- **`expert_cache.csv` (Group 2):** Complete log of all 10,341 HBM LRU cache transitions:
  - Records layer ID, expert ID, action (`MISS`, `INSERT`, `HIT`, `EVICT`), and resultant cache occupancy (`resident_bytes`).
- **`processor_trace.csv` (Group 1):** Detailed chronological timeline of every compute kernel executed on GPU vs Logic-PIM.

#### Class 3: System Logs & Verifiable Status Indicators
- **`logs/run.log` (or `concise_run.log`):** Formatted execution log containing stdout/stderr, startup preflight diagnostics, and per-step simulated DDR5 latencies.
- **`logs/completion_status.txt`:** Cryptographically verifiable machine-readable manifest recording:
  - `STATUS=SUCCESS`
  - `EXIT_CODE=0`
  - Exact host wall-clock runtime in seconds.
  - Checksums (SHA-256) of generated trace files.
  - Exact row counts for routing, load, and cache traces.

#### Class 4: Research Analysis Package (`analysis_ds6c6/`)
- **41 Processed CSV Datasets:** Curated datasets spanning memory wall disparity, per-token timing, cache occupancy, hypergeometric routing null models, multi-step temporal lag decay, and prefetch timing bounds.
- **30 Publication Figures (PNG & PDF):** 300 DPI publication charts and vector PDFs covering architecture diagrams, memory wall disparity ratios, cache timelines, and ST-MoE / APEX comparative bounds.

---

### 7.3 Where Will I Get It? (Directory & Filesystem Mapping)

The directory map below illustrates the exact paths where every output artifact is written:

```text
CA_Project/LLMSimulator/
│
├── reproduction_A100/results/
│   │
│   ├── a100_gpu_baseline/                               # Group 1: GPU Baseline Outputs
│   │   ├── csv/mixtral_synthesis_2048_128_..._decode.csv # Per-step performance metrics
│   │   ├── raw/device_0 ... device_3                    # Raw Gantt device execution trees
│   │   ├── traces/expert_routing.csv                    # 10,240 routing decisions
│   │   ├── traces/processor_trace.csv                   # Chronological kernel timeline
│   │   └── logs/concise_run.log                         # Formatted console execution log
│   │
│   ├── a100_duplex/                                     # Group 1: Plain Duplex Outputs (Same structure)
│   ├── a100_duplex_pe/                                  # Group 1: Duplex + PE Outputs (Same structure)
│   └── a100_duplex_pe_et/                               # Group 1: Duplex + PE + ET Outputs (Same structure)
│
└── reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/
    │
    ├── pilot_1step/                                     # Group 2: Pilot (1-Step) Outputs
    │   ├── csv/deepseekV3_synthesis_..._iter1_...decode.csv
    │   ├── traces/expert_routing.csv (456 records)
    │   ├── traces/expert_load.csv (1,756 records)
    │   ├── traces/expert_cache.csv (2,068 records)
    │   └── logs/completion_status.txt
    │
    └── full_run_5step_20261011/                         # Group 2: Full 5-Step Workload Outputs
        │
        ├── csv/                                         # Step summary metrics
        │   └── deepseekV3_synthesis_2048_10_GPU_..._iter5_...decode.csv
        │
        ├── traces/                                      # Authoritative raw simulation traces
        │   ├── expert_routing.csv                       # 2,280 routing records (57 layers x 8 top-k x 5 tokens)
        │   ├── expert_load.csv                          # 8,780 memory transfer records
        │   └── expert_cache.csv                         # 10,341 HBM cache state transitions
        │
        ├── logs/                                        # Execution monitoring & verification
        │   ├── run.log                                  # 21,159-second console execution log
        │   ├── completion_status.txt                    # SHA-256 verifiable success token
        │   └── preflight_summary.txt                    # System RAM, disk, and topology audit
        │
        ├── analysis_ds6c5/                              # Initial engineering audit & report
        │   └── report/DS-6C-5_final_run_and_routing_analysis.md
        │
        └── analysis_ds6c6/                              # Comprehensive Research Analysis Package
            ├── csv/                                     # 41 Processed CSV Datasets
            │   ├── per_token_timing.csv                 # Step-by-step DDR5 vs Compute times
            │   ├── load_latency_distribution.csv        # DDR5 batch transfer latencies
            │   ├── cache_occupancy_timeline.csv         # Step-by-step MB cache usage
            │   ├── multi_step_temporal_lag.csv          # Autocorrelation decay (lags 1-4)
            │   ├── prefetch_overlap_timing_bounds.csv   # Interconnect hiding bounds
            │   └── ... (36 additional research datasets)
            │
            ├── figures/png/                             # 30 High-Resolution PNG Figures (300 DPI)
            │   ├── fig1_system_architecture.png
            │   ├── fig3_latency_breakdown_log.png
            │   ├── fig5_disparity_ratio.png             # 12,696x memory-to-compute ratio
            │   ├── fig25_spatiotemporal_coactivation_heatmaps.png
            │   ├── fig30_attention_window_vs_transfer_bounds.png
            │   ├── FIGURES_EXPLANATION_AND_METHODOLOGY.md # Comprehensive 30-figure explanation
            │   └── ... (25 additional figures)
            │
            ├── figures/pdf/                             # 30 Vector PDF Figures (Publication-ready)
            │   └── fig1_system_architecture.pdf ... fig30_attention_window_vs_transfer_bounds.pdf
            │
            └── manifests/
                └── results_package_manifest.csv         # Cryptographic inventory of all 105 assets
```

---

### 7.4 How Will I Get It? (Generation Pipeline & Verification)

The deliverables are generated through a deterministic three-stage pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: C++ Simulation Telemetry (Real-Time Generation)                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • During execution, C++ objects append rows directly to open std::ofstream handles:             │
│   - Scheduler (sequence.cpp)    ──> appends each routed top-k expert to expert_routing.csv      │
│   - Device (device.cpp)         ──> appends MISS/INSERT/HIT transitions to expert_cache.csv     │
│   - LinearImpl (linear_impl.cpp)──> appends DDR5 latency & tensor size to expert_load.csv        │
│ • After each decode token, cluster->run() flushes per-token latency/energy rows to summary CSV. │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Process Termination & Watchdog Finalization                                            │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • The simulator exits with code 0 upon completing all iterations.                               │
│ • The runner script (run_mixtral_style_experiment.sh) captures exit code, computes SHA-256      │
│   hashes of all three raw trace files, counts line totals, and writes completion_status.txt.    │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Offline Data Transformation & Figure Generation (Instantaneous)                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • Independent Python analysis scripts process the raw traces into published assets:             │
│   1. generate_csv_datasets.py parses traces into 41 clean research CSVs under analysis_ds6c6/  │
│   2. generate_figures.py renders Figures 1-24 using matplotlib and exports PNG (300 DPI) & PDF. │
│   3. generate_additional_correlation_figures.py computes ST-MoE / APEX metrics (Figs 25-30).   │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### How to Verify Your Generated Deliverables
After executing any experiment command, verify your results with these quick terminal checks:

```bash
# 1. Verify exit code and completion status
cat reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/logs/completion_status.txt
# Expected: STATUS=SUCCESS, EXIT_CODE=0

# 2. Verify raw trace row counts
wc -l reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/traces/*.csv
# Expected:
#   2281 traces/expert_routing.csv (2,280 data rows + 1 header)
#   8781 traces/expert_load.csv    (8,780 data rows + 1 header)
#  10342 traces/expert_cache.csv   (10,341 data rows + 1 header)

# 3. Inspect top-level summary decode metrics
head -n 2 reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/csv/*.csv

# 4. View generated publication figures
ls -lh reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/figures/png/
# Expected: 30 PNG files (~5.4 MB total)
```

---

## 8. Figure and Analysis Documentation

All analytical figures and datasets can be inspected or regenerated **without rerunning the cycle-accurate simulator** by executing the offline analysis scripts on the preserved raw CSV traces.

### Regeneration Commands
```bash
cd /home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6

# 1. Regenerate all 41 CSV datasets from raw traces
python3 scripts/generate_csv_datasets.py

# 2. Regenerate Figures 1-24 (PNG and PDF)
python3 scripts/generate_figures.py

# 3. Regenerate Figures 25-30 (Advanced Spatio-Temporal and Prefetching Bounds)
python3 scripts/generate_additional_correlation_figures.py
```

### Complete 30-Figure Catalog

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Category 1: System & Pipeline Architecture                                                      │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Fig 1: A100 + Off-Chip DDR5 Architecture Diagram                                               │
│ Fig 2: Sequence-Step Execution Timeline & Phase Decomposition                                  │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Category 2: Execution Timing & Memory Wall Disparity                                           │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Fig 3: System Timing Breakdown vs Analytical Peak Compute (Log Scale)                          │
│ Fig 4: Per-Token Simulated Latency Breakdown (Stacked Bar)                                     │
│ Fig 5: Memory-to-Compute Disparity Ratio per Decoding Step (Bar Chart)                         │
│ Fig 6: Off-Chip Load Volume and Cold Batch Count per Token                                     │
│ Fig 7: Batch Transfer Duration Distribution (Histogram & KDE)                                  │
│ Fig 8: Per-Stage DDR5 Transfer Latency Profile                                                 │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Category 3: Routing Selection & Expert Frequency Uniformity                                    │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Fig 9: Expert Selection Heatmap across Decoding Steps (61 x 5 Grid)                            │
│ Fig 10: Expert Selection Frequency Distribution vs Uniform Null (Chi-Square)                   │
│ Fig 11: Top-K Rank Distribution across Stages                                                  │
│ Fig 12: Unique Experts Loaded per Execution Stage                                              │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Category 4: Spatial & Temporal Routing Dynamics                                                │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Fig 13: Consecutive-Token Routing Overlap (Jaccard Similarity)                                 │
│ Fig 14: Observed Routing Overlap vs Hypergeometric Null Distribution                           │
│ Fig 15: Per-Stage Temporal Routing Overlap across Consecutive Tokens                           │
│ Fig 16: Spatial Adjacent-Stage Routing Overlap                                                 │
│ Fig 17: Spatial Transition Probability Matrix (61 x 61)                                        │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Category 5: On-Chip Cache Dynamics & Cache Bypass Analysis                                     │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Fig 18: Cache Occupancy Timeline across Simulation Steps                                       │
│ Fig 19: Cache Action Breakdown (Misses, Inserts, Hits, Evictions)                              │
│ Fig 20: Cache Events by Layer and Token                                                        │
│ Fig 21: Expert Eviction Distribution across Expert IDs                                         │
│ Fig 22: Simulated Cache Capacity vs Active Working Set Occupancy                               │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Category 6: Execution Invariants & Runtime Parity                                              │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Fig 23: Invariant Compliance Scorecard (6 Critical Verification Checks)                        │
│ Fig 24: Planned vs Actual Wall-Clock Runtime Parity                                            │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Category 7: Advanced Spatio-Temporal Correlation & Prefetching Bounds (Literature Comparison)   │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Fig 25: Spatio-Temporal Expert Co-activation Heatmaps (ST-MoE Fig 2 Style)                     │
│ Fig 26: Multi-Step Temporal Lag Autocorrelation Decay (Lags 1-4)                                │
│ Fig 27: Multi-Hop Cross-Layer Spatial Distance Decay (Delta L in [1, 50])                      │
│ Fig 28: Oracle Extra-Prefetch Budget Delta* Distribution (APEX Fig 6a Style)                   │
│ Fig 29: Cumulative Prefetch Coverage Recall Curves (k + Delta Expansion)                       │
│ Fig 30: Attention Window vs DDR5 Transfer Timing Feasibility (APEX Eq 12 Bounds)               │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Disparity Ratio: Mathematical Formulation & Verification
The **Disparity Ratio** measures the severe divergence between physical off-chip memory transfer latency and theoretical compute speed:

$$\text{Ratio}_t = \frac{T_{\text{DDR5},t}}{T_{\text{Roofline},t}}$$

#### Timing Metrics Distinction
1. **Simulated DDR5 Time ($T_{\text{DDR5},t}$):** Cycle-accurate DDR5 transfer latency modeled by Ramulator 2 ($\approx 1,720.39\text{ ms}$ per token, $8,601.93\text{ ms}$ total).
2. **Simulated Device Latency ($T_{\text{device},t}$):** Total simulated hardware clock advance ($\approx 1,742.70\text{ ms}$ per token, $8,713.52\text{ ms}$ total). DDR5 stalls contribute **98.7%** of total execution latency!
3. **Internal Model Compute Time ($T_{\text{comp},t}$):** Simulated tensor execution time reported by the device status ($\approx 22.32\text{ ms}$ per token, $111.59\text{ ms}$ total).
4. **Analytical Peak FP16 Compute Time ($T_{\text{Roofline},t}$):** Theoretical minimum compute time assuming 100% saturation of all 312 TFLOP/s Tensor Cores:
   $$T_{\text{Roofline},t} = \frac{60\text{ layers} \times 8\text{ experts} \times 88,080,384\text{ FLOPs}}{312.0 \times 10^{12}\text{ FLOP/s}} = 0.1355\text{ ms}$$
   Over 5 tokens: $T_{\text{Roofline},\text{total}} = 5 \times 0.1355\text{ ms} = 0.6775\text{ ms}$.

#### Historical vs Verified Disparity Values
- **Aggregate Disparity Ratio:**
  $$\text{Ratio}_{\text{agg}} = \frac{8,601.93\text{ ms}}{0.6775\text{ ms}} = \mathbf{12,696.58\times} \approx \mathbf{12,696\times}$$
- **Mean of Per-Token Ratios:**
  $$\overline{\text{Ratio}} = \frac{13,128.5 + 12,725.3 + 12,639.0 + 12,351.1 + 12,639.0}{5} = \mathbf{12,696.58\times}$$
Both formulations reconcile to **12,696×**, proving that memory bandwidth stalls outpace raw silicon compute capacity by over four orders of magnitude.

---

## 9. Trace and CSV Schemas

### 1. `expert_routing.csv`
- **Location:** `<traces_dir>/expert_routing.csv`
- **Generated by:** `BatchedSequence::route_expert()` in `src/scheduler/sequence.cpp`
- **Header:** `request_id,token_id,layer_id,topk_rank,expert_id`
  - `request_id` (int): Unique sequence identifier (e.g., 2048).
  - `token_id` (int): Current decoding token index (e.g., 2048–2052).
  - `layer_id` (int): Modulo layer index (0–59).
  - `topk_rank` (int): Rank among Top-K selections (1–8).
  - `expert_id` (int): Selected routed expert ID (0–255 for DeepSeek-V3, 0–7 for Mixtral).

### 2. `expert_load.csv`
- **Location:** `<traces_dir>/expert_load.csv`
- **Generated by:** `LinearLayerImpl::calculateImpl()` in `src/hardware/linear_impl.cpp`
- **Header:** `request_id,layer_id,expert_id,weight_name,weight_size,start_time,end_time,duration,memory_source,hit_miss`
  - `weight_name` (string): `W1_gate`, `W3_up`, `W2_down`, or `all_weights`.
  - `weight_size` (int): Tensor byte size (e.g., 29,360,128 Bytes for individual weights; 88,080,384 for batch).
  - `duration` (double): Measured load duration in nanoseconds.
  - `memory_source` (string): `OFFCHIP_DRAM` or `HBM`.
  - `hit_miss` (string): `miss` or `hit`.

### 3. `expert_cache.csv`
- **Location:** `<traces_dir>/expert_cache.csv`
- **Generated by:** `Device::is_expert_resident()` & `Device::mark_expert_resident()` in `src/hardware/device.cpp`
- **Header (Format):** `layer_id,expert_id,action,resident_bytes`
  - `action` (string): `MISS`, `INSERT`, `HIT`, or `EVICT`.
  - `resident_bytes` (long long): Total HBM cache occupancy after action.

### 4. Summary Decode CSV
- **Location:** `<csv_dir>/<model>_<data>_..._decode.csv`
- **Generated by:** `Cluster::runIteration()` or `test_deepseekv3_mixtral_style.cpp`
- **Columns (51 Fields):**
  `iter_info`, `split`, `type`, `time`, `latency`, `queueing_delay`, `arrival_time`, `seq_queue_size`, `input_len`, `output_len`, `num_sum_iter`, `mixed`, `batchsize`, `numtoken`, `num_sum_seq`, `num_gen_seq`, `seqlen`, `sum_attention_opb`, `qkvgen`, `q_down_proj`, `kv_down_proj`, `kr_proj`, `q_up_proj`, `qr_proj`, `kv_up_proj`, `tr_k_up_proj`, `v_up_proj`, `atten_sum`, `atten_gen`, `o_proj`, `ffn`, `expert_ffn`, `communication`, `rope`, `layernorm`, `residual`, `act_energy`, `read_energy`, `write_energy`, `all_act_energy`, `all_read_energy`, `all_write_energy`, `mac_energy`, `total_energy`, `fc_dram`, `fc_comp`, `attn_dram`, `attn_comp`, `moe_dram`, `moe_comp`, `OOM`.

---

## 10. Validation and Reproducibility Checklist

Before declaring any experiment reproduced or deploying changes, execute the following validation checklist:

- [x] **1. Compilation Check:** `make -C build -j$(nproc)` completes with 0 errors and generates all required binaries (`run`, `run_deepseekv3_mixtral_style`, `test_lru`, `test_ddr5_mapping`).
- [x] **2. Configuration Integrity:** `./build/run_deepseekv3_mixtral_style --validate-config <config_path>` returns exit code `0` and confirms row utilization $< 100\%$.
- [x] **3. Trace File Creation:** Execution generates non-empty `expert_routing.csv`, `expert_load.csv`, and `expert_cache.csv` in the configured `traces_directory`.
- [x] **4. Routing Record Parity:** Total routing rows exactly equal:
  $$\text{Expected Rows} = \text{Batch Size} \times \text{Tokens} \times \text{MoE Layers} \times \text{Top-K}$$
  - For Group 1 (Mixtral): $32 \times 5 \times 32 \times 2 = 10,240\text{ records}$.
  - For Group 2 (DeepSeek 5-Step): $1 \times 5 \times 57 \times 8 = 2,280\text{ records}$.
- [x] **5. HBM Residency Bound:** Verified that `resident_bytes` in `expert_cache.csv` never exceeds `cache_capacity_bytes` ($80\text{ GB}$).
- [x] **6. Completion Status Verification:** Ensure `completion_status.txt` exists and reads `STATUS=SUCCESS` and `EXIT_CODE=0`.
- [x] **7. Plotting Integrity:** Scripts generate both vector `.pdf` and 300 DPI `.png` formats without missing data series.

---

## 11. Runtime Expectations, Warnings, and Troubleshooting

### Runtime Expectations
| Workload | Executable | Steps | Host Wall-Clock Time | Primary Bottleneck |
| :--- | :--- | :--- | :--- | :--- |
| **Mixtral Duplex (A100)** | `build/run` | 5 | ~45–60 seconds | GPU Compute & Interconnect |
| **DeepSeek-V3 Pilot (1-Step)** | `build/run_deepseekv3_mixtral_style` | 1 | ~70 minutes | Ramulator 2 DDR5 Simulation |
| **DeepSeek-V3 Full Run (5-Step)** | `build/run_deepseekv3_mixtral_style` | 5 | ~5.88 hours | Ramulator 2 DDR5 Simulation |

### Critical Troubleshooting Guidelines
1. **Never Rerun 5-Step Run for Trivial Reasons:** The 5-step DeepSeek-V3 simulation takes ~5.88 host hours. All raw CSV traces and processed datasets are safely preserved in `full_run_5step_20261011/`. Use offline analysis scripts for research investigations.
2. **Watchdog OOM Abort:** The runner scripts monitor free memory every 5 seconds. If available host RAM drops below 2,560 MB, the simulator is safely terminated to prevent host lockup. Ensure at least 4 GB of free RAM before launching simulations.
3. **Trace Path Collisions:** Always set `export TRACE_DIR="..."` when running drivers manually, or ensure `traces_directory` is defined in the configuration YAML.

---

## 12. Known Limitations and Research Insights

1. **Synthetic Routing Independence:**
   - The current DeepSeek-V3 simulation uses a synthetic pseudo-random routing generator (`std::mt19937` with seed 777).
   - As documented in Figures 25–29, this generator produces statistically independent draws matching a hypergeometric null distribution ($J \approx 0.0163$ vs $E[J] = 0.0169$).
   - Real LLMs (as demonstrated in the attached **ST-MoE** and **APEX** papers) exhibit strong semantic spatial and temporal correlations ($p < 0.01$). Predictive prefetchers evaluated on synthetic traces will observe near-zero hit rates ($\delta^* = 8$ for 76.8% of tokens) until real token routing traces are ingested.
2. **Modulo-60 Layer Index Drift:**
   - As audited in Section 6, `cur_layer` advances by $57 \pmod{60} = -3$ per token across sequential decode steps. All analysis must group by execution stage (1–57) rather than raw layer ID.
3. **Synchronous Memory Stalling:**
   - In the current implementation, expert loading occurs synchronously on a cache miss before matrix multiplication begins. Overlapping expert transfers with attention computation (as proposed in APEX and ST-MoE) requires high-bandwidth chiplet interconnects (NVLink-C2C $\ge 750\text{ GB/s}$) or multi-layer pipelined lookahead to hide the 31.2 ms DDR5 expert transfer window behind the 1.38 ms single-layer attention compute window.
