# DeepSeek-V3 671B MoE Memory-System Simulation: Final Technical Documentation and Results

**Repository**: `/home/mohman/ca_project/LLMSimulator`  
**Experiment Root**: `reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/`  
**Execution Profile**: Full-Depth 60-Layer Model (3 Dense + 57 MoE), Top-8 Routing, 256 Routed Experts + 1 Shared Expert, Single A100 Reference Device, 80 GB Expert LRU Cache, Off-Chip 2-Channel DDR5-3200 via Ramulator2  
**Original Pre-Analysis README Snapshot**: [`manifests/README_preanalysis_snapshot.md`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/manifests/README_preanalysis_snapshot.md) (SHA-256: `dfd35af1e3ce4aeff070973c9024e8748fbb3b0b50deac3518918de011135ca7`)

---

## 1. Experiment Objective

Mixture-of-Experts (MoE) architectures such as DeepSeek-V3 decouple parameter capacity from active computational overhead by activating only a sparse subset of expert feed-forward networks (FFNs) per token. However, when serving trillion-parameter scale MoEs on memory-constrained systems (e.g., a single GPU or a small node), the total working set of expert weights vastly exceeds accelerator High-Bandwidth Memory (HBM) capacity. Offloading non-resident expert parameters to off-chip system DRAM (DDR5) creates an extreme bottleneck known as the **MoE Memory Wall**.

This experiment addresses the following research questions:
1. **The MoE Memory Wall Magnitude**: When serving a full-depth DeepSeek-V3 model (60 layers, 256 routed experts per layer) with an 80 GB HBM expert cache and an off-chip DDR5 backing tier, what fraction of total execution latency is consumed by expert parameter loading vs. active tensor computation?
2. **LRU Cache Working-Set Saturation**: How quickly does an 80 GB LRU cache saturate under continuous sequential decoding, what is the steady-state eviction rate, and does the cache manager strictly respect physical capacity bounds?
3. **Routing Locality Diagnostics**: Do consecutive tokens or adjacent layers exhibit temporal or spatial expert correlation that could enable predictive prefetching?
4. **Hardware Modeling Accuracy**: What are the exact architectural assumptions, timings, and simulation fidelity limits of cycle-accurate Ramulator memory modeling compared to physical hardware?

---

## 2. Effective Model Configuration

The experiment simulates the complete architectural shape of **DeepSeek-V3 (671B total parameters, ~37B active parameters per token)**:

| Architectural Parameter | Configured Value | Mathematical Description | Source Verification |
| :--- | :--- | :--- | :--- |
| **Total Layers ($L$)** | 60 layers | 3 dense layers + 57 MoE layers | [`configs/deepseekv3_b1_pilot_1step.yaml`](file:///home/mohman/ca_project/LLMSimulator/configs/deepseekv3_b1_pilot_1step.yaml) |
| **Leading Dense Layers** | 3 layers | Layers 1, 2, 3 (Transformer layers 1..3) use dense multi-head latent attention (MLA) and standard dense FFN | `first_k_dense_replace: 3` |
| **Trailing MoE Layers** | 57 layers | Layers 4 through 60 (Transformer layers 4..60) contain sparse MoE modules | $60 - 3 = 57$ |
| **Routed Experts Pool ($N$)**| 256 experts | Candidate expert pool per MoE layer | `num_routed_expert: 256` |
| **Top-K Routing ($K$)** | 8 experts | Number of routed experts activated per token per MoE stage | `top_k: 8` |
| **Shared Experts ($S$)** | 1 expert | Always-active shared expert executing for every token | `n_shared_experts: 1` |
| **Hidden Dimension ($d_{\text{model}}$)** | 7,168 | Model representation dimension | `hidden_dim: 7168` |
| **MoE Intermediate Dim ($d_{\text{ffn}}$)** | 2,048 | Intermediate expansion dimension per expert | `moe_intermediate_dim: 2048` |
| **Weight Precision** | FP16 (2 bytes) | 16-bit floating point parameters | `precision_byte: 2` |
| **Single Weight Projection**| 29,360,128 B | $7,168 \times 2,048 \times 2\text{ bytes} \approx 29.36\text{ MB}$ | Size of W1 (gate), W2 (down), or W3 (up) |
| **Full Expert Weight Size** | 88,080,384 B | $3 \times 29,360,128\text{ bytes} \approx 88.08\text{ MB}$ | Combined expert parameter footprint |
| **Total Model Expert Footprint**| 1,289,849,141,760 B | $57 \text{ layers} \times 256 \text{ experts} \times 88,080,384\text{ B} \approx \mathbf{1.29\text{ TB}}$ | Offloaded weight store |
| **Batch Size & Decoding Steps**| Batch 1, 5 Steps | Sequential autoregressive token generation: tokens 2048 to 2052 | `batchsize: 1, iter: 5` |

---

## 3. Hardware and Memory Hierarchy

The simulation executes on a single-device GPU architectural profile with a tiered memory hierarchy:

```text
+-----------------------------------------------------------------------------------+
| SIMULATED ACCELERATOR (Single NVIDIA A100 Reference Model)                        |
|                                                                                   |
|  +-------------------------------------+  +------------------------------------+  |
|  | Peak Compute Engine                 |  | Fast On-Chip Memory (HBM3 Timing)  |  |
|  | Dense FP16 Peak: 312 TFLOP/s        |  | Capacity Partition: 80 GB          |  |
|  | Zero-latency cache-hit GEMM modeling|  | 8 Channels, 4 TB/s Timed Bus       |  |
|  +-------------------------------------+  +------------------------------------+  |
|                                                                                   |
|  +-----------------------------------------------------------------------------+  |
|  | Software Expert LRU Cache Manager                                           |  |
|  | Capacity Bound: 80,000,000,000 Bytes (80 GB decimal / 74.51 GiB binary)      |  |
|  | Resident Tensors: 908 (302 complete experts + 2 partial weight tensors)     |  |
|  | Peak Residency: 79,976,988,672 Bytes (99.97% capacity utilization)         |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
                                         |
                                         | Cache Miss: Exposed Stall
                                         v
+-----------------------------------------------------------------------------------+
| OFF-CHIP BACKING TIER (Host System DRAM via Ramulator2)                           |
|                                                                                   |
|  - Backing Bus: Dual-Channel DDR5-3200 (2 Channels x 32-bit, 51.2 GB/s Peak)      |
|  - Address Controller: offchip_mmap_controller (5 TB expanded physical geometry)  |
|  - Simulation Engine: Cycle-accurate bank conflict, row buffer, refresh timing   |
|  - Average Batch Load Latency: 3.901 ms per 88.08 MB expert                      |
+-----------------------------------------------------------------------------------+
```

### Architectural Model vs Physical GPU Distinction
> [!IMPORTANT]
> The measurements reported herein represent **cycle-accurate architectural simulator outputs**, not empirical physical-GPU benchmarks.
> 1. **Zero-Latency Cache-Hit Abstraction**: Expert lookups hitting in the 80 GB HBM cache are simulated with zero extra memory access latency.
> 2. **HBM3 Timing vs Physical A100 HBM2e**: The on-chip memory model utilizes an 8-channel HBM3 timing profile (~4 TB/s bandwidth), which exceeds physical A100 SXM4 HBM2e memory (1.55–2.0 TB/s).
> 3. **Synchronous Stall Execution**: Off-chip DDR5 requests stall the execution pipeline synchronously; no background DMA prefetching or compute-communication overlap was configured.

---

## 4. How Off-Chip DRAM was Integrated

The off-chip DDR5 path was integrated into LLMSimulator across the following source modules:

1. **System & Device Configuration** ([`src/hardware/device.cpp:60-95`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L60-L95)):
   - `Device::initializeDRAM` instantiates `dram_interface` for on-chip memory and `offchip_dram_interface` for off-chip DDR5 when off-chip simulation is enabled.
   - `offchip_mmap_controller` is allocated with an expanded 5 TB row-buffer address geometry (`num_row = 1,048,576`), resolving the 64-bit logical address overflow bug identified in Phase DS-5.
2. **Request Interception and Batching** ([`src/hardware/layer_impl.cpp:43-86`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/layer_impl.cpp#L43-L86)):
   - `issueRamulatorBatch` intercepts expert weight fetches. If an expert is flagged with `MemoryTarget::OFFCHIP_DRAM`, it constructs a batch of DRAM requests for `W1_gate`, `W2_down`, and `W3_up`.
   - Each tensor is mapped through `offchip_mmap_controller`:
     ```cpp
     auto offchip_mem_obj = MemoryObject::Create(
         tensor->getMMap(), tensor->getMemoryObject()->getLogicAddr(),
         tensor->getSize(), device->offchip_mmap_controller);
     dram_request->AddOperand(offchip_mem_obj, pim_operand_type);
     ```
   - Sets `bypass_cache = true` to ensure off-chip requests actually traverse the Ramulator cycle-accurate engine rather than being short-circuited by the legacy operation memoizer.
3. **Execution Dispatch** ([`src/hardware/device.cpp:173-195`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L173-L195)):
   - `Device::run_ramulator_batch` passes the request list to `device->offchip_dram_interface->run(requests)`, advancing the simulated DDR5 clock and recording latency.

---

## 5. LRU Expert Cache Architecture vs Legacy Execution Cache

Inspection of the source code clarifies the distinction between the two caching systems in LLMSimulator:

```text
+------------------------------------------------------------------------------------+
| 1. SOFTWARE EXPERT LRU CACHE (MoE Working Set Manager)                              |
|    Location: src/hardware/device.h:125-136, device.cpp:272-335                     |
|                                                                                    |
|    - Data Structures:                                                              |
|        std::list<std::tuple<int, int, long long>> lru_expert_list;                  |
|        std::unordered_map<std::pair<int, int>, ...> resident_experts;               |
|    - Capacity Accounting:                                                          |
|        cache_capacity_bytes = 80,000,000,000 Bytes (80 GB)                          |
|        resident_bytes tracks current allocated expert weight bytes                  |
|    - Cache Invariants:                                                             |
|        is_expert_resident(layer_id, expert_id) -> Promotes entry to MRU on HIT     |
|        mark_expert_resident(...) -> Evicts LRU front entries upon capacity exceed  |
+------------------------------------------------------------------------------------+
| 2. LEGACY EXECUTION CACHE (Operation Memoizer)                                      |
|    Location: src/hardware/device.h:87-95, device.cpp:115-145                       |
|                                                                                    |
|    - Purpose: Caches simulated execution times for repetitive tensor dimensions.   |
|    - Key: std::tuple<LayerType, ProcessorType, DRAMRequestType, long, MemoryTarget>|
|    - Behavior: Bypassed for OFFCHIP_DRAM so all cold batches execute in Ramulator. |
+------------------------------------------------------------------------------------+
```

### Expert Cache Interaction with Sister Weights (W1, W2, W3)
DeepSeek-V3 experts comprise three linear projection matrices:
- `W1_gate`: Gate projection ($7,168 \times 2,048$)
- `W3_up`: Up projection ($7,168 \times 2,048$)
- `W2_down`: Down projection ($2,048 \times 7,168$)

During execution, the model checks residency for each sub-weight:
1. When `W1_gate` is accessed, if the expert is absent, it triggers a **MISS**. The entire 3-weight batch (88.08 MB) is loaded from DDR5, and `mark_expert_resident` is called, inserting the expert.
2. When `W3_up` and `W2_down` are subsequently evaluated in the same layer pass, `is_expert_resident` returns **HIT** immediately.
3. Consequently, every cold expert accounts for **1 MISS, 1 INSERT, and 2 HITs**; warm experts account for **3 HITs**.

---

## 6. Complete Detailed MoE Execution Flow

```mermaid
flowchart TD
    A["1. Token Generation Step Advances (Token t in {2048..2052})"] --> B["2. Hidden State Enters Layer L (MoE Stages 1..57)"]
    B --> C["3. GateUpdate::aggregate_expert() Evaluates Router"]
    C --> D["4. Scheduler::getRandomExpert(8) Draws Top-8 Experts"]
    D --> E["5. Loop Over Selected Experts e in {e_1..e_8}"]
    E --> F{"6. is_expert_resident(L, e) in Cache?"}
    
    F -- "YES (HIT)" --> G["7. Promote Entry to MRU in lru_expert_list<br/>Log HIT to expert_cache.csv<br/>Latency = 0.0 ms"]
    F -- "NO (MISS)" --> H["8. Mark MemoryTarget::OFFCHIP_DRAM<br/>Log MISS to expert_cache.csv"]
    
    H --> I["9. Construct W1/W2/W3 Batch (88.08 MB)<br/>Translate Addresses via offchip_mmap_controller"]
    I --> J["10. issueRamulatorBatch() Dispatches to DDR5<br/>Simulate Transfer (~3.90 ms Stall)<br/>Log to expert_load.csv"]
    J --> K{"11. resident_bytes + 88MB > 80 GB?"}
    
    K -- "YES (Over Capacity)" --> L["12. Evict Victims from Front of lru_expert_list<br/>resident_bytes -= size<br/>Log EVICT to expert_cache.csv"]
    L --> K
    K -- "NO (Fits)" --> M["13. Insert Loaded Expert at MRU Position<br/>resident_bytes += size<br/>Log INSERT to expert_cache.csv"]
    
    G --> N["14. Execute Expert FFN Projections on GPU Engine"]
    M --> N
    N --> O["15. Accumulate Outputs with Shared Expert Output"]
    O --> P{"16. Final Layer in Model?"}
    
    P -- "NO (Advance Stage)" --> B
    P -- "YES (Step Complete)" --> Q["17. Advance Autoregressive Decoding Loop<br/>Record Step Latency in decode.csv"]
```

---

## 7. Function Inventory and Call Graph

The following table catalogs all primary source functions governing the execution pipeline:

| Function Signature | Source File & Line Range | Subsystem | Caller | Primary Callees | Responsibility & Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `int main(...)` | [`eval/test_deepseekv3_mixtral_style.cpp:38-457`](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L38-L457) | Harness | OS Process | `Scheduler::Create`, `Cluster::Create`, `cluster->run` | Program entry point; sets up YAML configs, memory interfaces, decoding loop, and metrics logging |
| `Scheduler::Create(...)` | [`src/scheduler/scheduler.cpp:15-68`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L15-L68) | Scheduler | `main` | `Scheduler` constructor, `initRunningQueue` | Initializes request scheduling; parses expert file path (`"none"` activates synthetic PRNG) |
| `Scheduler::getRandomExpert(...)` | [`src/scheduler/scheduler.cpp:141-153`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L141-L153) | Router PRNG | `BatchedSequence::update_expert` | `std::mt19937::operator()` | Samples top-8 expert IDs uniformly without replacement from $[0, 255]$ using fixed seed `777` |
| `BatchedSequence::update_expert(...)` | [`src/scheduler/sequence.cpp:152-241`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp#L152-L241) | Scheduler / Sequence | `GateUpdate::aggregate_expert` | `Scheduler::getRandomExpert` | Advances `cur_layer`, requests routing choices, logs to `expert_routing.csv` |
| `GateUpdate::aggregate_expert(...)` | [`src/module/route.cpp:149-185`](file:///home/mohman/ca_project/LLMSimulator/src/module/route.cpp#L149-L185) | MoE Module | `GateUpdate::run` | `BatchedSequence::update_expert` | Coordinates expert activation counts across sequences during MoE layer execution |
| `Device::is_expert_resident(...)` | [`src/hardware/device.cpp:272-301`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L272-L301) | LRU Cache | `LayerImpl::execution` | `resident_experts.find`, `lru_expert_list` MRU update | Checks HBM residency; promotes to MRU on HIT; logs HIT/MISS to `expert_cache.csv` |
| `Device::mark_expert_resident(...)` | [`src/hardware/device.cpp:303-335`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L303-L335) | LRU Cache | `LayerImpl::execution` | `lru_expert_list.pop_front`, `resident_experts.insert` | Enforces 80 GB limit via LRU eviction; inserts expert at MRU; logs EVICT/INSERT |
| `issueRamulatorBatch(...)` | [`src/hardware/layer_impl.cpp:43-86`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/layer_impl.cpp#L43-L86) | DRAM Subsystem | `LayerImpl::execution` | `Device::run_ramulator_batch`, `MemoryObject::Create` | Creates W1/W2/W3 DRAM requests, applies `offchip_mmap_controller`, dispatches to Ramulator |
| `Device::run_ramulator_batch(...)` | [`src/hardware/device.cpp:173-195`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp#L173-L195) | Hardware Device | `issueRamulatorBatch` | `DRAMInterface::run` | Dispatches DRAM request list to DDR5 interface |
| `Cluster::run(...)` | [`src/hardware/cluster.cpp:85-115`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/cluster.cpp#L85-L115) | Cluster Manager | `main` | `Model::run`, `Device::run` | Executes single token decoding pass across simulated hardware modules |

---

## 8. Artifact, Dataset, and Figure Inventory

All research artifacts are organized under `full_run_5step_20261011/`:

### Raw Input Traces & Execution Logs
- [`traces/expert_routing.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/traces/expert_routing.csv): 2,280 routing selections across 5 tokens.
- [`traces/expert_load.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/traces/expert_load.csv): 18,090 tensor lookup records, including 2,205 cold batches (88.08 MB each).
- [`traces/expert_cache.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/traces/expert_cache.csv): 10,342 cache events (4,635 hits, 2,205 misses, 2,205 inserts, 1,297 evictions).
- [`csv/deepseekV3_..._decode.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/csv/deepseekV3_synthesis_2048_10_GPU_N1_D1_TP1_DP1_maxbatch1_maxprocess524288_iter5_skew0_precision_byte2_parallel_execution0_ramul_decode.csv): Per-step latency and energy breakdown.
- [`logs/run.log`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/logs/run.log): Full execution console log (21,159.21 host wall seconds).

### Research CSV Datasets (41 Files)
Located in [`analysis_ds6c6/csv/`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/):
- **Timing & Performance**: `run_summary.csv`, `timing_breakdown.csv`, `per_token_timing.csv`, `per_layer_timing.csv`, `load_latency_distribution.csv`, `load_volume_by_token.csv`, `load_volume_by_layer.csv`, `load_batch_details.csv`
- **Routing Diagnostics**: `routing_frequency_by_expert.csv`, `routing_frequency_by_layer_expert.csv`, `routing_frequency_by_layer.csv`, `routing_rank_distribution.csv`, `routing_selection_matrix.csv`, `per_token_routing_summary.csv`, `temporal_jaccard_by_stage.csv`, `temporal_jaccard_by_transition.csv`, `temporal_transition_counts.csv`, `spatial_overlap_by_adjacent_stage.csv`, `spatial_transition_counts.csv`, `routing_correlation_summary.csv`, `random_routing_baseline.csv`
- **Advanced Spatio-Temporal Datasets (ST-MoE & APEX Inspired)**: `multi_step_temporal_lag.csv`, `multi_hop_spatial_distance.csv`, `oracle_delta_distribution.csv`, `cumulative_prefetch_coverage.csv`, `prefetch_overlap_timing_bounds.csv`
- **Cache & Memory Hierarchy**: `cache_event_summary.csv`, `cache_event_timeline.csv`, `cache_occupancy_timeline.csv`, `cache_evictions_by_expert.csv`, `cache_evictions_by_layer.csv`, `cache_hits_by_layer.csv`, `cache_misses_by_layer.csv`, `cache_capacity_validation.csv`, `cache_residency_invariants.csv`
- **Provenance & Configurations**: `artifact_inventory.csv`, `integrity_validation.csv`, `metric_provenance.csv`, `model_configuration.csv`, `hardware_configuration.csv`, `source_function_inventory.csv`

### Publication Figures (30 Figures in PNG & PDF)
Detailed construction and analytical guide: [`analysis_ds6c6/figures/png/FIGURES_EXPLANATION_AND_METHODOLOGY.md`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/figures/png/FIGURES_EXPLANATION_AND_METHODOLOGY.md)  
Located in [`analysis_ds6c6/figures/png/`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/figures/png/) and [`analysis_ds6c6/figures/pdf/`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/figures/pdf/):
- **Architecture & Flow**: `fig1_system_architecture`, `fig2_execution_flow`
- **Timing & Memory Wall**: `fig3_latency_breakdown_log`, `fig4_per_token_timing`, `fig5_disparity_ratio`, `fig6_load_volume_per_token`, `fig7_batch_duration_distribution`, `fig8_per_stage_ddr5_latency`
- **Expert Routing**: `fig9_expert_selection_heatmap`, `fig10_expert_frequency_distribution`, `fig11_topk_rank_distribution`, `fig12_unique_experts_per_stage`
- **Correlation & Null Baseline**: `fig13_consecutive_token_jaccard`, `fig14_observed_vs_null_overlap`, `fig15_per_stage_temporal_overlap`, `fig16_spatial_adjacent_stage_overlap`, `fig17_spatial_transition_heatmap`
- **Cache & Working Set**: `fig18_cache_occupancy_timeline`, `fig19_cache_action_composition`, `fig20_cache_events_by_layer`, `fig21_evictions_by_expert_distribution`, `fig22_capacity_vs_occupancy`
- **Validation & Runtime**: `fig23_invariant_checks`, `fig24_runtime_planned_vs_actual`
- **Advanced ST-MoE & APEX Diagnostics**: `fig25_spatiotemporal_coactivation_heatmaps`, `fig26_multistep_temporal_lag_decay`, `fig27_multihop_spatial_distance_decay`, `fig28_oracle_delta_budget_distribution`, `fig29_cumulative_prefetch_coverage`, `fig30_attention_window_vs_transfer_bounds`

---

## 9. Key Results and Quantitative Interpretation

### 1. The MoE Memory Wall Dominance
- Total simulated device latency across 5 tokens: **8,713.53 ms** (~1.74 s per token).
- Simulated off-chip DDR5 transfer stalls: **8,601.93 ms** (**98.72%** of execution time).
- Simulated internal model compute & attention: **111.60 ms** (**1.28%** of execution time).
- Analytical peak FP16 compute (A100 312 TFLOP/s): **0.6775 ms** across all 5 tokens.
- **Memory-to-Compute Disparity Ratio**:
  $$\text{Disparity Ratio} = \frac{8,601.93\text{ ms}}{0.6775\text{ ms}} \approx \mathbf{12,696\times}$$
  Because execution is synchronous, **100% of the 8.60 seconds of DDR5 transfer time is exposed as pipeline stall time**.

### 2. Cache Capacity Saturation and Steady-State
- **Working-Set Growth**: The 80 GB cache begins empty. During Token 2048, 456 cold experts are requested.
- **Capacity Ceiling**: At Event Index 1358 (during Token 2048), occupancy reaches **79,976,988,672 bytes (908 weight tensors = 302 complete experts + 2 partial tensors)**.
- **Eviction Regime**: From that moment onward, every subsequent insertion triggers an eviction.
  - Total Inserts: 2,205
  - Total Evictions: 1,297
  - Net Resident in HBM: $2,205 - 1,297 = 908$ tensors ($\mathbf{79.98\text{ GB}}$).
  - Maximum occupancy strictly respects the 80.00 GB limit ($79.98\text{ GB} \le 80.00\text{ GB}$).

### 3. Routing Provenance and the Layer-Indexing Instrumentation Finding
- **Provenance**: Verified that routing selections were generated by `Scheduler::getRandomExpert()` querying `std::mt19937(777)`. Selections are independent uniform draws without replacement from $\{0 \dots 255\}$.
- **Instrumentation Finding**: DeepSeek-V3 calls `update_expert` 57 times per token. Because `cur_layer` updates modulo 60, recorded `layer_id` shifts by $-3$ per token.
  Aligning analysis by **MoE execution stage order ($1 \dots 57$)** resolves this shift and provides true temporal and spatial alignment.

### 4. Correlation Analysis vs Analytical Null Model
Under independent uniform draws of size $K=8$ from $N=256$:
- Expected intersection size: $E[X] = \frac{8 \times 8}{256} = 0.2500$ experts.
- Theoretical Hypergeometric Null Jaccard: $E[J] = \mathbf{0.0169}$.
- Observed Temporal Jaccard (by stage): $\mathbf{0.0155} \pm 0.034$ (76.8% zero overlap, 23.2% single-expert overlap, 0% multi-expert overlap).
- Observed Spatial Jaccard (adjacent stages): $\mathbf{0.0176} \pm 0.030$ (73.9% zero overlap, 25.7% single-expert overlap, 0.4% two-expert overlap).
- **Finding**: Both temporal and spatial correlations match the independent uniform null model within statistical variance.

---

## 10. Methodological Comparison: APEX and ST-MoE Literature

The findings of this experiment relate directly to the methodologies established in the MoE literature:

| Research Dimension | APEX & Prefetching Literature | ST-MoE (Zoph et al., 2022) | This Experiment (DS-6C Baseline) |
| :--- | :--- | :--- | :--- |
| **Expert Loading Bottleneck** | Identifies memory bus as primary bottleneck in offloaded MoE; transfers overwhelm compute by $10^2$–$10^4\times$. | Focuses on training stability, router entropy, and capacity factors on TPU clusters. | Confirms $12,696\times$ disparity between DDR5 transfer stalls and peak compute in full-depth DeepSeek-V3. |
| **Cache & Capacity Bounds** | Proposes dynamic expert caching in GPU memory; demonstrates hit rates saturate as cache fills. | Examines token drop rates under strict expert capacity limits. | Demonstrates strict 80 GB LRU saturation at Event 1358; 1,297 evictions balance subsequent cold insertions. |
| **Temporal Routing Locality** | Reports that real LLM text tokens exhibit temporal reuse ($J \approx 0.15$–$0.35$), enabling cache hits. | Notes semantic specialization of experts on specific syntactic/domain tokens. | **Absent by design**: Synthetic PRNG yields $J = 0.0155$, matching independent uniform noise ($E[J] = 0.0169$). |
| **Spatial Cross-Layer Correlation** | Leverages hidden-state similarity between adjacent layers for predictive prefetching. | Evaluates routing entropy and router z-loss across deep layers. | **Flat transition matrix**: Synthetic PRNG yields uniform dispersion with zero clustering across adjacent stages. |
| **Random-Routing Baseline** | Uses Hypergeometric independent draws as the theoretical lower bound for cache hit rates. | Evaluates uniform routing as a sanity check for router collapse. | Explicitly validates the exact analytical Hypergeometric baseline ($E[X]=0.25, E[J]=0.0169$) against simulation data. |

> [!WARNING]
> While the memory-system architecture correctly models DDR5 transfer mechanics and LRU eviction, **the current data must not be used to claim that DeepSeek-V3 lacks temporal locality or cannot be prefetched**. The lack of correlation is an artifact of the configured pseudo-random routing generator (`std::mt19937`), not a property of real DeepSeek-V3 inference.

---

## 11. Limitations and Future Work

1. **Synthetic Routing Workload**: As confirmed in Task 2, routing decisions were generated by a pseudo-random number generator without evaluating genuine language tokens or router gate projections.
2. **Exploratory Sample Size**: Only 5 sequential decoding steps were simulated due to the heavy computational cost of cycle-accurate Ramulator simulation (~1.18 hours of CPU time per token).
3. **Synchronous Execution Baseline**: The current harness lacks asynchronous double-buffering or speculative prefetching; all DDR5 transfers occur on the critical execution path.
4. **HBM Timing Abstraction**: Cache hits are modeled with zero latency, and the simulated HBM profile uses fast HBM3 timings rather than physical A100 HBM2e parameters.
5. **Cheapest Path to Real Routing Traces**: To evaluate real DeepSeek-V3 routing correlations without simulating 1,000s of tokens in Ramulator (~49 days of CPU time), future work should extract the 57 router gate projection weight matrices (~209 MB total) and compute top-8 routing traces offline across standard NLP benchmarks. Those traces can then be replayed directly through the existing `expert_file_path` mechanism in LLMSimulator.

---

## 12. Reproduction Guide

To reproduce all 36 CSV datasets, 24 publication figures, and the results manifest from the completed run artifacts without rerunning Ramulator:

```bash
# Ensure current working directory is the repository root
cd /home/mohman/ca_project/LLMSimulator

# 1. Regenerate all 36 CSV datasets
python3 reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/scripts/generate_csv_datasets.py

# 2. Regenerate all 24 publication figures (PNG at 300 DPI + Vector PDF)
python3 reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/scripts/generate_figures.py
```
