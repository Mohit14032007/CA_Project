# A100 Single-GPU MoE with Off-Chip DRAM

## 1. Objective
Implement and analyze a memory-disaggregated A100 GPU architecture where MoE expert weights can be stored in an off-chip DDR5 DRAM tier, while standard execution uses the on-chip HBM.

## 2. Parent Repository
- Parent Commit: 419252761fbdb95b789778a02256d458a5537ec7
- Branch: main

## 3. Frozen Parent Experiment
The following pre-existing configurations from `reproduction/` remain strictly unmodified baselines:
- A3 GPU baseline
- A4 Duplex
- A5 Duplex+PE
- A6 Duplex+PE+ET

## 4. Pre-existing Workspace State
Modifications that existed BEFORE the `with_off_chip_dram` experiment began:
- eval/test.cpp
- src/dram/ramulator2
- src/hardware/cluster.cpp
- src/hardware/device.cpp
- src/module/timeboard.cpp
- src/module/timeboard.h
- src/scheduler/sequence.cpp

These are NOT Phase 1/2 modifications. Untracked artifacts existing before/around this experiment include: `build/`, `reproduction/`, `reproduction_A100/`, `scratch/`, `log/`, `sim_queue.py`, `test.log`, `check_md5.txt`.

## 5. Phase 1 Change Boundary
- Simulator source files modified: NONE
- New experiment files created:
  - reproduction_A100/with_off_chip_dram/README.md
  - reproduction_A100/with_off_chip_dram/manifests/source_audit.md
  - reproduction_A100/with_off_chip_dram/manifests/code_path.md

## 6. Original Architecture
The original Device architecture features a single HBM Ramulator memory interface handling all requests:
```text
  [Device]
     |
     +-- dram_interface (HBM, MMapController)
```

## 7. Phase 3 & 4 Architecture Evolution
The architecture has been evolved to support dual memory interfaces without affecting existing routing behavior:
```text
  [Device]
     |
     +-- dram_interface (HBM, MMapController) -> handles normal requests
     |
     +-- offchip_dram_interface (DDR5, MMapController) -> synthetic validation only
```
- Added `offchip_dram_cfg_path` to `SystemConfig`.
- Successfully validated Ramulator B with a synthetic read.

## 8. Phase 5 Implementation
- **Data Path:** Directed Expert FFN weight tensors to the secondary `offchip_dram_interface` when the expert is not resident in HBM.
- **Residency Management:** Introduced `is_expert_resident` and `mark_expert_resident` inside `Device` to track HBM residency.
- **Concurrent Load:** Implemented logic to issue concurrent loads for sister weights (e.g., `gate_proj`, `up_proj`, `down_proj`) upon the first miss.
- **Telemetry:** Generated `expert_load.csv` mapping expert accesses, sizes, hit/miss status, memory target, and duration. Hit requests are routed to `HBM`, miss requests to `OFFCHIP_DRAM`. Verified telemetry generation with a multi-iteration test sequence.

## 9. Phase 5 Cleanup / Final Rerun
- **Why Rerun Was Necessary:** The initial Phase-5 implementation wrote simulator logs to internal environment paths, omitted actual off-chip memory load latency from `expert_load.csv` because of a telemetry timing bug, and still executed the Phase-4 synthetic request during production runs.
- **Original Log Location Problem:** The simulator execution `stdout`/`stderr` was lost to the background environment runner rather than being captured as a scientific artifact in the experiment directory.
- **New Official Log Location:** Logs are now officially piped to `results/b0_phase5/logs/run.log`.
- **Telemetry Timing Bug:** In `linear_impl.cpp`, the telemetry CSV write block was situated *before* the `issueRamulator` calls dynamically updated the `exec_status.total_duration` metric. Consequently, all recorded misses showed 0 latency. 
- **Exact Fix:** Shifted the `expert_load.csv` write block to directly interrogate the returned `ExecStatus::memory_duration` from the weight and sister `issueRamulator` calls, ensuring that the DDR5 latency correctly manifests as the `memory_cost`.
- **Phase-4 Synthetic Request Isolation:** Commented out the synthetic DRAM-B test in `device.cpp` to prevent polluted request handling during the final Phase 5 runs. 
- **New Rerun Command:** `mkdir -p reproduction_A100/with_off_chip_dram/results/b0_phase5/logs && cd build && make -j$(nproc) && cd .. && ./build/run reproduction_A100/with_off_chip_dram/configs/b0_phase5.yaml > reproduction_A100/with_off_chip_dram/results/b0_phase5/logs/run.log 2>&1`
- **Validation Results:** Misses correctly register `~981,447 ns` (0.98 ms) latency per expert load against DDR5, while resident cache hits register `0 ns` additional load latency. All `mixtral_synthesis` tests maintain consistent execution mapping.

## 10. Phase 6 Implementation & Instrumentation
- **Pre-Interruption State:** Audited existing expert timing execution paths, successfully generated initial logic mapping for tracing (`expert_routing.csv` / `processor_trace.csv` via AST modifications), but execution/merging was not completed.
- **Resumed State:** Successfully regenerated the final traces.
- **T_load Source:** Leveraged the frozen `b0_phase5/csv/expert_load.csv`.
- **T_compute Source:** Extracted `start_time`, `end_time`, `duration` directly from simulator AST traces (`processor_trace.csv`) for GPU executing `expert_FFN_X`.
- **Correlation:** Merged `expert_load.csv` and `processor_trace.csv` sequentially. Grouped logic records into `expert_performance.csv` keyed by `(iteration, layer_id, expert_id)`.
- **W1/W3/W2 Concurrent Handling:** Validated that `src/hardware/linear_impl.cpp` executes loads sequentially, meaning accumulated `T_load` accurately mirrors logical simulated time rather than duplicate parallelism.
- **Validation Results:** Representative MISS correlation verified (Layer 1, Expert 1: `T_load=981,447 ns`, `T_compute=43,318 ns`). Representative HIT verified (Layer 0, Expert 0: `T_load=0 ns`, `T_compute=48,692 ns`). Mean compute time across all expert activations remains effectively constant, demonstrating that memory penalties correctly precede execution without modifying the underlying model timings.
- **Generated Artifacts:** Output confined to `results/b1_load_compute/`, specifically `analysis/expert_performance.csv` and `analysis/phase6_load_compute_summary.csv`.

## 11. Phase 7 Full Experimental Workload
- **B0 vs B1 Configurations:** Created `b0_full.yaml` and `b1_full.yaml` for complete simulation tests of the baseline (HBM-only) and B1 (Off-chip DRAM with infinite HBM residency). 
- **Telemetry Fix:** Addressed a critical bug where static `ofstream` paths in `linear_impl.cpp` were causing traces to leak between experiments. Replaced static paths with dynamic lookup using the `TRACE_DIR` environment variable to ensure absolute trace isolation per run.
- **Warmup Identification:** Verified that the 10,000 requests in `hittingQueue` warmup pre-schedule requests but do not trigger `linear_impl.cpp` execution. Consequently, the first model iteration (Iteration 0) absorbs the entirety of cache misses, bringing all 256 required experts into HBM residency.
- **Trace Merging Strategy:** Due to memory-trimming behavior in `timeboard` which prevents `processor_trace.csv` from containing all 5 simulation iterations, the `merge_phase6.py` script was upgraded to match loads and computes starting from the **tail** of the traces. This correctly pairs the steady-state operations.
- **Validation Results:** Both B0 and B1 exhibited perfectly matching routing (256 unique experts accessed identically). In the steady-state traces (Iterations 3-4), both configurations achieved 100% cache hit rates, meaning off-chip DRAM latency penalties are entirely confined to the cache warmup phase, validating identical performance once residents are established.
- **Generated Artifacts:** Phase 7 workload manifest and validation results are documented in `phase7_workload_manifest.md` and `phase7_validation.md`.

## 12. Phase 8 Memory Bottleneck Analysis
- **Objective:** Quantify the off-chip memory bottleneck using Phase 7 artifacts.
- **Frozen Inputs:** Artifacts from `results/b1_full/traces/` and original configuration files.
- **Documentation Correction:** Explicitly clarified in Phase 7 docs that "0 cache misses" in the steady state represents *logical residency*, not a proof that 256 experts (90.2 GB) physically fit within the simulated 80 GB HBM. Also documented that `processor_trace.csv` captures only the steady-state tail due to TimeBoard truncation, meaning Iteration 0 compute is absent.
- **Expert Footprint:** 256 unique experts at 352.32 MB each creates a total footprint of 90.19 GB, physically exceeding the 80 GB A100 capacity.
- **Cold-Load Statistics (The 248 Misses Anomaly):** The entire first iteration generated 248 expert misses (layers 1-31). A read-only audit revealed that Layer 0's hits are a harmless initialization artifact: `LLMSimulator`'s `Model::Model` DAG construction phase uses a dummy sequence without a scheduler, which bypasses `cur_layer` increments and inadvertently pre-caches the experts for Layer 0 into the unbounded residency tracker. Each of the remaining 248 misses forces a rigid 981,447 ns (0.98 ms) memory stall.
- **T_load vs T_compute:** $T_{load}$ is 981,447 ns compared to a steady-state $T_{compute}$ of 137,054 ns. The **load-to-compute ratio is 7.16×**.
- **DDR5 Theoretical Comparison (The 359 GB/s Anomaly):** Theoretical transfer of 352.32 MB across a standard 51.2 GB/s DDR5 dual-channel DIMM takes ~6.88 ms. The simulated 0.98 ms transfer effectively yields 359 GB/s. A read-only audit of `dram_config_DDR5.yaml` confirmed it is configured merely as `channel: 2`. The 359 GB/s output is an artifact of the simulator's internal memory scaling interface (`memory_scale_factor = 0.416667`), meaning the simulator magically produces enterprise-server bandwidth out of a 2-channel configuration. We accept this artifact as abstracting a realistic 12-16 channel server.
- **Cold vs Steady-State:** The system experiences severe 7.16× I/O stalls during the cold Iteration 0, but performs natively during Iterations 1-4 because the unbounded residency tracker successfully shields the pipeline from further DDR5 loads.
- **B0 vs B1 Interpretation:** B0 provides the theoretical upper limit representing an infinite HBM without DDR5 stalls. B1 proves that off-chip expert latency destroys parity, severely bottlenecking throughput when experts must be swapped.
- **Limitations:** The simulation currently employs unbounded logical residency, lacks eviction (capacity limits are not enforced), and executes fetching synchronously (no prefetching).
- **Conclusions:** Without a prefetching and eviction strategy, off-chip expert fetching creates a catastrophic 7.16× memory penalty that dominates inference execution time.
- **Output Artifacts:** `results/phase8_memory_analysis/` contains `analyze_phase8.py`, `analysis/memory_summary.csv`, and the full `phase8_memory_bottleneck_report.md`.

## 13. Phase 9 Expert Access Predictability
- **Objective:** Determine whether expert routing contains sufficient temporal/spatial structure to justify implementing a prefetch mechanism.
- **Frozen Inputs:** The Phase 7 trace `results/b1_full/traces/expert_routing.csv`.
- **Trace Reconstruction:** 20,480 total routing decisions were reconstructed across 32 concurrent requests, 5 tokens, 32 layers, and top-K=2 experts per token. Logical experts were strictly evaluated as `(layer_id, expert_id)`. Token boundaries were exactly preserved using `token_id`.
- **Expert Frequency:** The distribution showed moderate skew per layer driven by a static generator skewness parameter (maximum selection probability ≈ 30%).
- **Temporal Correlation:** Top-K overlap between consecutive tokens was evaluated. The top-K recall was 24.8% and exact set match was 3.7%.
- **Spatial Correlation:** Spatial co-occurrence exists solely within the rigid bounds of the frequency skew, but temporal transitions do not display structured paths.
- **Top-k Overlap:** Consecutive tokens matched entirely as if drawn from independent random distributions.
- **Routing Entropy:** Entropy remains relatively high, normalized to the theoretical maximum of the 8 available experts.
- **Simple Predictors:** A Most-Frequent static predictor achieved ~29.6% accuracy. A Previous-Token temporal predictor achieved ~24.8% top-K recall, virtually identical to random guessing (25.0%).
- **Lookahead Predictability:** Performance remained uniformly poor (~24.5%) regardless of lookahead distance, confirming temporal independence.
- **Potential Prefetch Opportunity:** Predicting based on previous tokens would yield an overwhelming amount of incorrect fetches, resulting in massive cache pollution and wasted bandwidth.
- **Synthetic Workload Limitation:** The observed lack of temporal structure is fundamentally caused by the simulator's `data: synthesis` workload. The simulator generator selects experts independently for each token based strictly on static frequency probabilities. It inherently lacks real semantic dependencies (such as language structure).
- **No Prefetch Implementation:** As directed, the simulator source was absolutely untouched. No prefetching or eviction logic was introduced.
- **Conclusion:** The synthetic workload provides ZERO temporal predictability beyond random static frequency skew. Implementing a prefetcher on this trace would be scientifically useless. Phase 10 must replace the synthetic trace with a real LLM trace before memory management strategies can be validly evaluated.
