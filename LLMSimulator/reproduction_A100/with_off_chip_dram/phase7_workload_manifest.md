# Phase 7 Workload Manifest

## 1. Warmup Behavior Analysis

The `LLMSimulator` initializes the workload using a `hittingQueue` warmup process. In Phase 7, we identified that the `hittingQueue` logic simulates sequence injection and schedules requests but **does not execute the `linear_impl.cpp` model graph**.
- As a result, the warmup phase does not produce any memory or compute events in `expert_load.csv` or `processor_trace.csv`.
- The `expert_load.csv` accurately captures the exact 5 iterations of the `runIterationMixed` loop (producing 3,840 loads for B1, which corresponds to 8 experts * 3 tensors * 32 layers * 5 iterations).
- Because `resident_experts` is implemented as an unbounded set (adhering strictly to the Phase 5 frozen semantics without forced eviction), all 256 unique experts are cached *logically* during their first access (in Iteration 0).
- Consequently, iterations 1 through 4 experience a 100% cache hit rate for expert weights. Note: This means B1 reports zero subsequent logical residency misses after the cold iteration because the current residency tracker is unbounded. This must NOT be interpreted as evidence that the complete set of 256 experts (≈ 90.2 GB) fits physically within the 80 GB HBM.

## 2. B0 vs B1 Configuration Differences

The two full configurations (`configs/b0_full.yaml` and `configs/b1_full.yaml`) use identical baseline parameters representing a standard A100 environment executing a Mixtral-like workload:
- **B0 (Baseline - HBM Only):** The `offchip_dram_cfg_path` is explicitly empty. Expert weights are placed statically in HBM.
- **B1 (Off-Chip DRAM enabled):** The `offchip_dram_cfg_path` points to a DDR5 configuration. Expert weights trigger off-chip loads upon their first access and are subsequently maintained in HBM using the infinite residency tracker.

## 3. Justification for Trace Sizes (Iteration Matching)

During execution, `timeboard` manages a growing tree of `TimeStamp` objects. Because `cluster->exportGantt` is invoked outside the main simulation loop, and because `top_module_graph->reset_status()` partially clears the state, **the final `processor_trace.csv` does NOT retain the full 5 iterations and is NOT a complete execution trace.**
- TimeBoard memory trimming causes earlier events to be discarded before the final export. Iteration 0 compute is entirely absent.
- B0's `processor_trace.csv` contains only the retained tail of the last 3 iterations (768 matched operations).
- B1's `processor_trace.csv` contains only the retained tail of the last 2 iterations (512 matched operations).
- The `processor_trace.csv` trace is suitable for **steady-state compute analysis only**. The cold-load analysis must use `expert_load.csv` as the authoritative source.
- To ensure valid analysis, the trace merging logic (`merge_phase6.py`) was updated to pair the **tail** of `expert_load.csv` with the tail of `processor_trace.csv`. By comparing the last N iterations, we accurately isolate the steady-state execution phase.

Because both B0 and B1 are evaluated in their steady state (Iterations 3-4), the experts are already resident in the logical HBM tracker for B1. As a result, B1 reports zero subsequent logical residency misses (often denoted as 0 cache misses) in the final analysis summary, and exhibits near-identical compute performance. This must not be interpreted as implying zero physical memory overhead in a physically constrained system.
