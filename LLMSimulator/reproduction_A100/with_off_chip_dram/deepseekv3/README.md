# DeepSeek-V3 Single-GPU Off-Chip DRAM Experiment

Status:
DS-1 compatibility audit in progress/completed

Architecture target:
- 1 GPU
- A100-class 80 GB HBM
- Logic-PIM disabled
- multi-GPU disabled
- expert tensor parallelism = 1
- off-chip DRAM for expert weights
- GPU executes experts

Scientific goals:
1. Measure expert loading time
2. Measure expert execution time
3. Compare T_load vs T_compute
4. Measure expert memory traffic
5. Analyze expert frequency
6. Analyze temporal correlation
7. Analyze spatial correlation
8. Measure predictability
9. Eventually use the results to determine whether
   a prefetcher is justified

Do not claim any DS results yet.

Status:
DS-2 FP16 validation completed.
Ready for DS-3 (LRU Eviction Implementation).

## Phase DS-4A: DDR5/Ramulator Audit
- **Status**: Completed
- **Verdict**: MUST RERUN
- **Findings**: The off-chip latency calculation is artificially impacted by the execution cache and a mismatch between the MMapController's 32-channel mapping and Ramulator's 2-channel DDR5 configuration. The execution cache bypassed Ramulator for 99.9% of requests, meaning realistic queuing and contention effects were absent. A fix must be implemented before moving to DS-5.
