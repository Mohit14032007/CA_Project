# DeepSeek-V3 Mixtral-Style MoE Memory-System Reproduction Study

**Experiment Directory:** `reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/`  
**Host Simulation Binary:** `build/run_deepseekv3_mixtral_style`  
**Architecture Profile:** Single-Device A100-SXM4-80GB Reference with Off-Chip DDR5 Backing Store  

---

## 1. Research Objective and Experiment Scope

This experiment quantifies the memory-wall bottleneck in serving modern Mixture-of-Experts (MoE) architectures, specifically modeling a DeepSeek-V3-class architecture on an A100-class accelerator. 

The primary scope is to model the memory hierarchy dynamics:
- Local high-bandwidth memory (HBM3 80 GB) acting as an LRU expert cache.
- Off-chip DDR5 DRAM acting as the backing tier holding the entire routed expert inventory.
- Quantifying the disparity between simulated memory transfer duration and analytical/simulated GPU computation time.
- Investigating spatial and temporal expert routing overlap during sequential autoregressive decoding.

---

## 2. Architecture of the Memory-System Model

The memory system models a two-tier hierarchy:
```
+-------------------------------------------------------------------+
|                        NVIDIA A100 Device                         |
|                                                                   |
|  +--------------------+             +--------------------------+  |
|  | GPU Compute Engine | <---------> | Local HBM3 (80 GB Pool)  |  |
|  | (SMs / Tensor Core)|             | 2 TB/s Peak Bandwidth    |  |
|  +--------------------+             | LRU Expert Cache Tier    |  |
|                                     +--------------------------+  |
|                                                  ^                |
+--------------------------------------------------|----------------+
                                                   | Off-Chip PCIe /
                                                   | DRAM Interconnect
                                                   v
                                      +--------------------------+
                                      | Off-Chip DDR5 Backing    |
                                      | 2.75 TB / 2.5 TiB Store  |
                                      | 14,592 Routed Experts    |
                                      | (1.285 TB FP16 Footprint)|
                                      +--------------------------+
```

1. **Local HBM Tier:** Modeled via Ramulator 2 with `dram_config_HBM3_80GB.yaml` (80 GB capacity, 32 channels, 16,384 rows). An LRU cache manager manages up to 80,000,000,000 bytes of expert weights.
2. **Backing DDR5 Tier:** Modeled via Ramulator 2 with `dram_config_DDR5_retry.yaml` (2.75 TB addressable capacity across 10,485,760 rows).
3. **Execution Semantics:** For each MoE layer during decoding, top-$k$ routed experts are selected. If an expert is already resident in HBM (Cache HIT), it is moved to MRU with zero off-chip memory transfer latency. If absent (Cache MISS), the expert's weight matrices ($W_1, W_2, W_3$) are batched together and loaded from off-chip DDR5 via Ramulator, displacing LRU resident experts if the 80 GB cache limit is exceeded.

---

## 3. Dedicated Harness and Component Reuse

The execution harness `eval/test_deepseekv3_mixtral_style.cpp` directly instantiates LLMSimulator core components:
- **`Scheduler::Create(...)`**: Creates the LLMSimulator request and sequence scheduler.
- **`Cluster::Create(...)`**: Instantiates the device topology and interconnect.
- **`Model model(...)`**: Generates the complete computation and module graph for 60 layers.
- **`cluster->run(metadata)`**: Directly steps through the hardware simulation path per decoding step.
- **`cluster->setStat(stat)` & `setTimeBreakDown(stat)`**: Corrected in DS-6C-2F to populate genuine simulation statistics and breakdowns rather than empty zeros.
- **Bypass of hittingQueue:** The legacy 10,000-step synthetic pre-warmup queue (`hittingQueue`) is bypassed (`use_hitting_queue: false`), executing deterministic, sequential decoding immediately.

---

## 4. DeepSeek-V3 Model Parameters & FP16 Accounting

The simulated model follows DeepSeek-V3 architectural dimensions:
- **Total Layers:** 60 layers (3 dense attention/MLP layers + 57 MoE layers).
- **Routed Experts per MoE Layer:** 256 routed experts.
- **Top-$k$ Routing:** 8 active experts per token.
- **Shared Experts:** 1 shared expert per MoE layer (always resident/executed).
- **Hidden Dimension ($d_{\text{model}}$):** 7,168.
- **Expert Intermediate Dimension ($d_{\text{expert}}$):** 2,048.
- **Weight Matrices per Expert:** $W_1$ (gate), $W_2$ (up), $W_3$ (down).
  $$\text{Elements per expert} = 3 \times (7,168 \times 2,048) = 44,040,192\text{ parameters}$$
- **Precision:** FP16 (`precision_byte: 2`).
  $$\text{Bytes per expert} = 44,040,192 \times 2 = 88,080,384\text{ bytes} \approx 88.08\text{ MB}$$
- **Total Routed Inventory:** $57 \times 256 = 14,592$ routed experts.
  $$\text{Total expert volume} = 14,592 \times 88,080,384 = 1,285,268,963,328\text{ bytes} \approx 1.285\text{ TB}$$

---

## 5. HBM Capacity, DDR5 Backing Tier, and LRU Cache

- **HBM Cache Capacity:** 80,000,000,000 bytes ($74.51\text{ GiB}$).
  $$\text{Max Resident Experts} = \lfloor 80,000,000,000 / 88,080,384 \rfloor = 908\text{ complete experts}$$
- **Working Set vs Cache Capacity:**
  - 1 token accesses $57 \times 8 = 456$ routed experts ($40.16\text{ GB}$).
  - 2 tokens access up to $912$ routed experts ($80.33\text{ GB}$).
  - Therefore, working set crosses the 80 GB boundary during Token 2, triggering LRU evictions in subsequent steps.

---

## 6. DDR5 Geometry Expansion and Address Mapping Validation

- **Problem in Original Config:** Original DDR5 geometry had 131,072 rows ($2^{17}$), providing only 34.36 GB capacity, causing aliasing when addressing 1.285 TB of experts.
- **Expanded DDR5 Geometry:** Row count was expanded to $10,485,760$ rows ($80 \times 2^{17}$).
  $$\text{Total Addressable DDR5} = 10,485,760 \times 262,144\text{ bytes} = 2,748,779,069,440\text{ bytes} \approx 2.75\text{ TB}$$
- **Validation:** Validated via unit test `build/test_production_ddr5_mapping`. Row utilization for 14,592 experts is $46.76\%$, strictly preventing row index overflow and collisions.

---

## 7. Workload Distinctions

| Characteristic | Smoke Test (`smoke/`) | 5-Step Run (`full_run_5step_20261011/`) | Reference Workload (`deepseekv3_b1_full_reference.yaml`) |
| :--- | :--- | :--- | :--- |
| **Layers** | 2 MoE layers | 60 layers (3 dense + 57 MoE) | 60 layers (3 dense + 57 MoE) |
| **Tokens / Steps** | 2 steps | 5 sequential steps | 128 tokens |
| **Batch Size** | 1 | 1 | 32 |
| **Cache Budget** | 200 MB | 80 GB ($80,000,000,000$ B) | 80 GB |
| **Host Runtime** | ~1.5 minutes | ~6.25 hours | ~150+ hours (Not launched) |
| **Purpose** | Fast harness verification | Full-depth memory-wall study | Production scaling reference |

---

## 8. Telemetry Layer Numbering Convention

**Crucial Mapping:**
- Transformer architecture contains 3 dense attention layers followed by 57 MoE layers.
- In `expert_routing.csv`: `layer_id` is 0-indexed across MoE layers:
  - `layer_id = 0` $\rightarrow$ First MoE layer (Transformer Layer 4)
  - `layer_id = 56` $\rightarrow$ Last MoE layer (Transformer Layer 60)
- In `expert_load.csv` and `expert_cache.csv`: `layer_id` is 1-indexed across MoE layers:
  - `layer_id = 1` $\rightarrow$ First MoE layer (Transformer Layer 4)
  - `layer_id = 57` $\rightarrow$ Last MoE layer (Transformer Layer 60)
- This numbering is strictly preserved to maintain backward compatibility with analysis scripts.

---

## 9. Build, Execute, Monitor, and Validate

### Build
```bash
cd /home/mohman/ca_project/LLMSimulator
cmake -B build -S .
cmake --build build --target run_deepseekv3_mixtral_style -j$(nproc)
```

### Preflight Config Validation
```bash
./build/run_deepseekv3_mixtral_style configs/deepseekv3_b1_full_run_5step.yaml --validate-config
```

### Execute with 7-Hour Watchdog
```bash
WATCHDOG_TIMEOUT=25200 ./scripts/run_mixtral_style_experiment.sh configs/deepseekv3_b1_full_run_5step.yaml
```

### Monitor
```bash
tail -f full_run_5step_20261011/logs/run.log
```

---

## 10. Output Directory Hierarchy

```
full_run_5step_20261011/
├── configs/          # Run YAML configuration snapshot
├── config_snapshot/  # Exact Ramulator DRAM YAMLs & checksums
├── logs/             # run.log, completion_status.txt
├── raw/              # Raw completion status copy
├── traces/           # expert_routing.csv, expert_load.csv, expert_cache.csv
├── csv/              # Canonical LLMSimulator summary CSV (*_ramul_decode.csv)
├── processed/        # summary_metrics.json, per_layer_loads.csv, etc.
├── figures/          # fig1 - fig5 in PNG (300 DPI) and vector PDF
├── manifests/        # full_run_manifest.md with full provenance
├── report/           # full_run_results.md
└── README.md         # Dedicated run documentation
```

---

## 11. CSV Schemas and Metric Definitions

1. `expert_routing.csv`: `request_id,token_id,layer_id,topk_rank,expert_id`
2. `expert_load.csv`: `request_id,layer_id,expert_id,weight_tensor,size_bytes,duration,memory_source,cache_hit_or_miss`
3. `expert_cache.csv`: `layer_id,expert_id,action,resident_bytes` (`action` $\in$ `{HIT, MISS, INSERT, EVICT}`)
4. Canonical Simulator CSV: 51 columns recording per-token latency, attention/FFN breakdowns, and energy values.

---

## 12. Reproduction and Figure Generation

To regenerate metrics and figures from raw traces:
```bash
python3 scripts/process_metrics.py \
    --trace-dir full_run_5step_20261011/traces \
    --csv-dir full_run_5step_20261011/csv \
    --output-dir full_run_5step_20261011/processed

python3 scripts/analyze_results.py \
    --traces-dir full_run_5step_20261011/traces \
    --processed-dir full_run_5step_20261011/processed

python3 scripts/generate_figures.py \
    --processed-dir full_run_5step_20261011/processed \
    --traces-dir full_run_5step_20261011/traces \
    --output-dir full_run_5step_20261011/figures
```

---

## 13. Limitations of Synthetic Routing and Analytical Compute

- **Synthetic Routing:** Top-$k$ selection utilizes uniform/skewed synthetic distribution, not real token embeddings. Real NLP tasks exhibit higher semantic clustering.
- **Analytical Compute Estimates:** Compute duration for Figure 5 is calculated analytically using NVIDIA A100 dense FP16 peak tensor throughput ($312\text{ TFLOP/s}$). It represents ideal compute saturation and does not account for kernel launch overhead or tensor core fragmentation.

---

## 14. Interpretability Scope of the 5-Token Workload

- **Supported Conclusions:**
  - Demonstrates cache capacity saturation ($80\text{ GB}$) across sequential decoding.
  - Proves LRU eviction mechanisms operate correctly under multi-gigabyte working sets.
  - Quantifies severe order-of-magnitude memory-wall disparity ($>10,000\times$) between off-chip DRAM loading and GPU matrix multiplication.
- **Unsupported Conclusions:**
  - Cannot make long-term routing clustering or multi-turn conversational locality claims (requires $\ge 128$ tokens).
  - Cannot measure multi-batch queueing or preemption dynamics (batch size is fixed to 1).
