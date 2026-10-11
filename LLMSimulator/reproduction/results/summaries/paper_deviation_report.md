# Paper Deviation Report

## Paper Result (Duplex)
The original Duplex paper introduces an architectural paradigm for co-processing sparse MoE blocks on Logic-PIM memory alongside a GPU handling dense compute.

## Repository Result
The evaluated repository provides a deterministic, cycle-approximate timeline simulator that accurately implements the PE (Parallel Execution) and ET (Expert Tensor Parallelism) mechanisms described in the paper. 

## Our Measured Result / Deviations
1. **Workload:** We strictly evaluated a highly constrained purely-decode microbenchmark (32 requests × 5 iterations). The original paper likely profiles extensive end-to-end multi-workload execution.
2. **Telemetry Scope:** Physical hardware metrics (e.g. absolute memory bandwidth saturated in GB/s, wall-clock power measurements) are inherently simulated/modeled values. They demonstrate theoretical behavior rather than empirical physical silicon performance.
3. **Execution Model:** The performance measurements are generated analytically via discrete event scheduling models, which provide relative scalability insights (e.g., the 3.07x speedup) but should not be conflated with raw physical benchmarking.
