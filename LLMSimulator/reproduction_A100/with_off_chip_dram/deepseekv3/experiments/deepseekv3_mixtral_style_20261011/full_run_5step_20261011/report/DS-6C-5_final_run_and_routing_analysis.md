# DS-6C-5: Final Run Audit, Routing-Correlation Analysis, and Research-Validity Assessment

**Date:** 2026-10-11  
**Repository:** `/home/mohman/ca_project/LLMSimulator`  
**Audited Run Root:** `reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/`  
**Harness Executable:** `build/run_deepseekv3_mixtral_style`  
**Source Commit:** `f75d362ae157f81ef2d3bd4055c9b22e34882a9f`  
**Deliverable ID:** DS-6C-5  

---

## 1. Run-Integrity Audit

An independent recalculation and provenance verification of all completed artifacts in [`full_run_5step_20261011/`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/) was performed directly against the raw CSV traces, logs, and simulator outputs.

### A. Provenance & Execution Invariants
- **Process Status:** The simulation exited normally with exit code `0` (`STATUS=SUCCESS`).
- **Host Wall-Clock Runtime:** 21,159.0 seconds (**5.877 hours** / 352.65 minutes), matching the pre-flight expected range of 5.83 to 6.28 hours (21,000–22,600 s) with a variance of -0.8%.
- **Watchdog Setting:** 25,200.0 seconds (7.0 hours). The process completed with **4,041.0 seconds** (**67.35 minutes**) of safety headroom remaining, satisfying the launch gate margin requirement.
- **Host Resource Stability:** Peak RSS reached **1,084.7 MB** (~1.08 GB), and minimum available system RAM remained at **4,741 MB**, exceeding the 2,560 MB safety threshold.
- **Configuration & Binary Checksums:**
  - Binary `build/run_deepseekv3_mixtral_style`: `819a752527b536224b1467a297d72158281da5c7d5e54e7f792a0d521c4e1093`
  - YAML `configs/deepseekv3_b1_full_run_5step.yaml`: `1bdcaee0f058c448a69f94e256998dc44da6acca5fdbd01b74db41b3a3954f6d`
  - DDR5 `config_snapshot/dram_config_DDR5_retry.yaml`: `7ef030f82d7f654339aa0ab1ff31abce311322c59aee693eb57774e071db9d97`
  - HBM3 `config_snapshot/dram_config_HBM3_80GB.yaml`: `e7ae1fb6802016fba28dd8d2cac439a15c01d2615c2063973eacb628903e79c6`

### B. Trace File Schema & Line Count Verification
| Trace File | Header Schema | Expected Lines | Verified Raw Lines | Status |
|:---|:---|:---:|:---:|:---:|
| `expert_routing.csv` | `request_id,token_id,layer_id,topk_rank,expert_id` | 2,281 | 2,281 (2,280 data) | **MATCH** |
| `expert_load.csv` | `request_id,layer_id,expert_id,weight_tensor,size_bytes,start,end,duration,memory_source,memory_destination,cache_hit_or_miss` | 18,091 | 18,091 (18,090 data) | **MATCH** |
| `expert_cache.csv` | `layer_id,expert_id,action,resident_bytes` | 10,343 | 10,343 (10,342 data) | **MATCH** |
| Canonical Decode CSV | 51 canonical simulator columns | 6 | 6 (5 steps) | **MATCH** |

---

## 2. Independently Recalculated Memory & Cache Statistics

Every reported quantity was independently recomputed from raw rows using [`analysis_ds6c5/audit_and_analyze.py`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c5/audit_and_analyze.py):

| Metric Description | Reported Reference | Independently Recalculated | Discrepancy |
|:---|:---:|:---:|:---:|
| **Sequential Decoding Tokens** | 5 (tokens 2048–2052) | **5 (tokens 2048–2052)** | None |
| **Total Routed Selections** | 2,280 | **2,280** | None |
| **Routed Selections Per Token** | 456 ($57\text{ MoE} \times 8$) | **456** across all 5 tokens | None |
| **Cold Expert Batches Loaded** | 2,205 | **2,205** | None |
| **Constituent Expert Matrices** | 6,615 ($2,205 \times 3$) | **6,615** ($W_1, W_2, W_3$) | None |
| **Off-Chip DDR5 Transferred Bytes** | 194,217,246,720 B | **194,217,246,720 B** (~194.22 GB) | None |
| **Cache Hits** | 4,635 | **4,635** | None |
| **Cache Misses** | 2,205 | **2,205** | None |
| **Cache Insertions** | 2,205 | **2,205** | None |
| **LRU Cache Evictions** | 1,297 | **1,297** | None |
| **Peak Resident Footprint** | 79,976,988,672 B | **79,976,988,672 B** (74.48 GiB) | None |
| **Steady-State Resident Experts** | 908 complete experts | **908 complete experts** | None |
| **Simulated DDR5 Transfer Time** | 8,601.93 ms | **8,601.93 ms** | None |
| **Simulated Device Latency** | 8,713.53 ms | **8,713.53 ms** | None |

### Accounting Reconciliation:
- Total cache events ($10,342$) strictly reconcile:
  $$\text{Hits } (4,635) + \text{Misses } (2,205) + \text{Inserts } (2,205) + \text{Evictions } (1,297) = 10,342$$
- Total off-chip load records ($8,820$) strictly reconcile:
  $$2,205\text{ aggregate batches} + 6,615\text{ constituents } (W_1, W_2, W_3) = 8,820$$
- Resident capacity strictly respects the 80 GB limit:
  $$908\text{ experts} \times 88,080,384\text{ bytes} = 79,976,988,672\text{ bytes} \le 80,000,000,000\text{ bytes}$$

---

## 3. Routing Data Generation & Code-Level Provenance

Inspection of the actual execution binary and scheduler source code reveals precisely how routing decisions were generated:

1. **Invocation Path in Harness:**
   - In [`eval/test_deepseekv3_mixtral_style.cpp`](file:///home/mohman/ca_project/LLMSimulator/eval/test_deepseekv3_mixtral_style.cpp#L240):
     ```cpp
     Scheduler::Ptr scheduler = Scheduler::Create(system_config, model_config, "none", max_batch_size, 8192, max_process_token);
     ```
     The expert file path argument is explicitly `"none"`.
2. **Trace Ingestion Bypass in Scheduler:**
   - In [`src/scheduler/scheduler.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L514-L519):
     ```cpp
     if (!expert_file_path.compare("none")) {
       std::cout << "Using synthesis trace" << std::endl;
       real_data = false;
       real_expert_data = false;
       return;
     }
     ```
     `real_expert_data` is explicitly set to `false`.
3. **Synthetic PRNG Sampling in `getRandomExpert`:**
   - In [`src/scheduler/scheduler.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/scheduler.cpp#L141-L153):
     ```cpp
     std::set<int> Scheduler::getRandomExpert(int top_k) {
       static unsigned int seed = 777;
       static std::mt19937 generator(seed);
       static std::uniform_int_distribution<int> distribution(0, model_config.num_routed_expert - 1);
       std::set<int> route;
       while (route.size() < top_k) {
         int seq_id = (distribution(generator) % model_config.num_routed_expert);
         route.insert(seq_id);
       }
       return route;
     }
     ```
   - For every decoding step and MoE layer, 8 distinct integers are sampled uniformly at random without replacement from $\{0, 1, \dots, 255\}$ using a deterministic Mersenne Twister engine (`std::mt19937`) initialized with seed `777`.

### Core Validity Conclusion:
> [!IMPORTANT]
> The routing decisions in this experiment were generated purely by **synthetic pseudo-random uniform sampling**. The experiment used **zero actual DeepSeek-V3 neural network weights, zero router gate activations, and zero real language modeling inputs**.
> 
> Therefore, these memory-system measurements characterize the performance of the memory hierarchy under a **uniform synthetic random-routing workload**. They **do not reflect, demonstrate, or validate routing correlations, clustering, or predictability of actual DeepSeek-V3 inference**.

---

## 4. Temporal Routing Correlation Analysis

To evaluate temporal routing continuity across consecutive decoding tokens, the selected expert sets were reconstructed for all 5 decoding steps.

### A. Theoretical Null Model (Independent Uniform Sampling)
When expert sets $A$ and $B$ are drawn independently and uniformly at random from $N=256$ candidates with top-$k = 8$, the intersection size $X = |A \cap B|$ follows a Hypergeometric distribution:
$$P(X = k) = \frac{\binom{8}{k} \binom{248}{8-k}}{\binom{256}{8}}, \quad k \in \{0, 1, \dots, 8\}$$

The exact analytical probability mass function (PMF) and resulting expected Jaccard similarity $E[J]$ are:
- $P(X = 0) = 77.29\%$ (Jaccard = $0.0000$)
- $P(X = 1) = 20.53\%$ (Jaccard = $1/15 \approx 0.0667$)
- $P(X = 2) = 2.08\%$ (Jaccard = $2/14 \approx 0.1429$)
- $P(X = 3) = 0.10\%$ (Jaccard = $3/13 \approx 0.2308$)
- $P(X \ge 4) < 0.003\%$
- **Analytical Null Model Expected Jaccard:**
  $$E[J_{\text{null}}] = \sum_{k=0}^8 P(X = k) \frac{k}{16 - k} = \mathbf{0.0169}$$

### B. Observed Temporal Routing Statistics
Across the 4 consecutive token transitions ($2048 \to 2049 \to 2050 \to 2051 \to 2052$):
- **Observed Mean Jaccard (by Sequential Execution Stage):** **0.0155**
- **Observed Mean Jaccard (by Telemetry `layer_id`):** **0.0221** (Mean reported in DS-6C-4 was **0.0199**)
- **Empirical Overlap Distribution:**
  - $0$ overlapping experts: $76.75\%$
  - $1$ overlapping expert: $23.25\%$
  - $\ge 2$ overlapping experts: $0.00\%$

```
Temporal Overlap Comparison:
  Observed Mean Jaccard:   0.0155
  Null Model Expectation:  0.0169
  Difference:             -0.0014 (Within expected statistical variance for 228 samples)
```

The observed temporal Jaccard similarity matches the independent uniform null model expectation. There is **zero detectable temporal correlation** between consecutive tokens beyond pure chance.

---

## 5. Spatial Routing Correlation Analysis

Spatial correlation evaluates whether routing choices in adjacent MoE layers within the same token ($L \to L+1$) exhibit co-activation patterns.

### Observed Spatial Statistics:
- **Observed Mean Spatial Jaccard:** **0.0176**
- **Null Model Expectation:** **0.0169**
- **Difference:** $+0.0007$
- **Spatial Overlap Distribution:**
  - $0$ overlapping experts: $75.89\%$ (vs $77.29\%$ null)
  - $1$ overlapping expert: $22.50\%$ (vs $20.53\%$ null)
  - $2$ overlapping experts: $1.61\%$ (vs $2.08\%$ null)
  - $\ge 3$ overlapping experts: $0.00\%$ (vs $0.10\%$ null)

### Spatial Transition Matrix Analysis:
A full $256 \times 256$ transition co-occurrence matrix was computed for all layer transitions across the 5 tokens. The transition counts range between 0 and 4 across the 65,536 matrix cells, with an average of 0.27 co-occurrences per cell. As visualized in [`fig_s3_spatial_transition_heatmap.png`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c5/fig_s3_spatial_transition_heatmap.png), the distribution is completely dispersed across the matrix with zero diagonal concentration or clustering.

---

## 6. Telemetry Layer-ID Wrapping Phenomenon

The audit discovered a structural artifact in the telemetry layer numbering:
- The model architecture comprises 60 total layers: 3 dense layers followed by 57 MoE layers.
- In `src/scheduler/sequence.cpp`, the layer counter `cur_layer` wraps modulo $60$ (`cur_layer %= num_layer`).
- However, `BatchedSequence::update_expert` is called only for the 57 MoE layers.
- Consequently, each sequential token advances the starting layer index by $57 \pmod{60} = -3$:
  - **Token 2048:** Routes through telemetry layers `0` to `56` (57 layers).
  - **Token 2049:** Routes through telemetry layers `57, 58, 59, 0, 1, ..., 53` (57 layers).
  - **Token 2050:** Routes through telemetry layers `54, 55, ..., 59, 0, ..., 50` (57 layers).
  - **Token 2051:** Routes through telemetry layers `51, ..., 47` (57 layers).
  - **Token 2052:** Routes through telemetry layers `48, ..., 44` (57 layers).

This modular shift explains why `df['layer_id'].nunique() == 60` across the entire run despite each token evaluating exactly 57 MoE layers. When analyzing sequential continuity, grouping by **execution stage order** correctly aligns corresponding MoE layers.

---

## 7. Audit of Load-versus-Compute Conclusions

The reported memory-wall disparity of $\approx 12,696\times$ was recalculated:

$$T_{\text{DDR5}} = 8,601.93\text{ ms}$$
$$T_{\text{compute}} (60\text{ layers}) = \frac{5 \times 60 \times 8 \times 88,080,384\text{ FLOPs}}{312.0 \times 10^{12}\text{ FLOP/s}} = 0.6775\text{ ms} \ (677.54\ \mu\text{s})$$
$$\text{Disparity Ratio} = \frac{8,601.93\text{ ms}}{0.6775\text{ ms}} = \mathbf{12,695.8\times} \approx \mathbf{12,696\times}$$

When recalculated strictly for the 57 active MoE layers:
$$T_{\text{compute}} (57\text{ layers}) = \frac{5 \times 57 \times 8 \times 88,080,384\text{ FLOPs}}{312.0 \times 10^{12}\text{ FLOP/s}} = 0.6437\text{ ms} \ (643.66\ \mu\text{s})$$
$$\text{Disparity Ratio (57 MoE)} = \frac{8,601.93\text{ ms}}{0.6437\text{ ms}} = \mathbf{13,364.0\times}$$

### Essential Interpretive Boundaries:
1. **Analytical Roofline vs Physical Silicon:** The denominator assumes $100\%$ tensor core saturation at the theoretical peak of 312 TFLOP/s. On physical GPUs, tensor fragmentation, memory coalescing, and CUDA kernel launch overheads lower effective computational efficiency, making practical speedup ratios lower than the idealized roofline limit.
2. **Synchronous Memory Stalls:** In the modeled execution path, expert weights are loaded synchronously upon a cache miss before matrix multiplication begins. Therefore, simulated DDR5 transfer duration directly constitutes an exposed memory stall.
3. **HBM Timing Abstraction:** Hits in the HBM LRU cache are modeled with zero transfer latency, folding local memory bandwidth into the forward compute stage. Additionally, the Ramulator HBM config (`dram_config_HBM3_80GB.yaml`) models an HBM3-class timing baseline, whereas physical A100 GPUs utilize HBM2e memory.

---

## 8. Real Routing Traces: Inventory & Proposed Methodology

### A. Current Workspace Inventory
A filesystem search confirms that **no real DeepSeek-V3 routing traces exist in the repository or user workspace**. All previous traces under `reproduction_A100/` and `SSD-offloading/` represent synthetic traces (`*_synthesis_*.csv`) or Mixtral 8x7B benchmarks (32 layers, top-2 routing).

### B. Why Running 1,000s of Simulated Tokens is Counterproductive
Simulating 1 token in the cycle-accurate Ramulator 2 DDR5 model takes **~1.18 host hours** (4,230 host seconds). Simulating 1,000 tokens through cycle-accurate Ramulator simply to gather routing samples would require over **1,180 host hours** (49 days) of CPU compute, and would merely produce more synthetic pseudo-random numbers without any insight into real DeepSeek-V3 routing dynamics.

### C. Recommended Next Steps for Real Route Collection
To obtain genuine DeepSeek-V3 routing data without requiring 671B model execution:
1. **Lightweight Router-Only Inference:**
   - DeepSeek-V3 routing decisions depend only on token representations $h$ and router gate weights $W_{\text{gate}}$ ($d_{\text{model}} = 7,168 \to 256$).
   - A single router layer has only $7,168 \times 256 \approx 1.83\text{ million parameters}$ (~3.67 MB per layer).
   - Across all 57 MoE layers, the total router gate weights require only **~209 MB** of storage.
   - An isolated Python script can extract just the router gate matrices and evaluate them on sample text embeddings, saving genuine top-8 expert routing decisions directly to a CSV trace.
2. **Offline Trace-Replay in LLMSimulator:**
   - Once a real routing CSV is generated, update `Scheduler::initExpertList` to ingest top-8 CSV traces directly.
   - Replay the real trace through the LRU cache model without re-running PRNG generators.

---

## 9. Reproduction Instructions

To reproduce the independent audit, statistical calculations, and supplemental figures:
```bash
cd /home/mohman/ca_project/LLMSimulator

# Execute independent audit and routing correlation script
python3 reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c5/audit_and_analyze.py
```

All supplemental artifacts will be generated in:
`reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c5/`
- `audit_summary.json`
- `temporal_correlation_by_stage.csv`
- `spatial_correlation_adjacent_layers.csv`
- `spatial_expert_transition_matrix.csv`
- `fig_s1_temporal_overlap_distribution.png` (and `.pdf`)
- `fig_s2_spatial_overlap_distribution.png` (and `.pdf`)
- `fig_s3_spatial_transition_heatmap.png` (and `.pdf`)
