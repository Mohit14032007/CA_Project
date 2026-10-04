# Phase 8 Memory Analysis Manifest

## Input Artifacts
This analysis relied exclusively on the following frozen artifacts generated during Phases 5-7:
- `reproduction_A100/with_off_chip_dram/results/b1_full/traces/expert_load.csv`
- `reproduction_A100/with_off_chip_dram/results/b1_full/traces/processor_trace.csv`
- `reproduction_A100/with_off_chip_dram/results/b1_full/traces/expert_routing.csv`
- `reproduction_A100/with_off_chip_dram/configs/b1_full.yaml`
- `reproduction_A100/with_off_chip_dram/configs/dram_config_DDR5.yaml`

## Analysis Scripts Created
- `reproduction_A100/with_off_chip_dram/results/phase8_memory_analysis/analyze_phase8.py`: A Python 3 script using standard libraries (`csv`, `collections`, `statistics`, `os`) to parse the CSV outputs, compute total sizes, aggregate miss durations, derive mean/median statistics for T_load and T_compute, and output a CSV summary.

## Output Files Created
- `reproduction_A100/with_off_chip_dram/results/phase8_memory_analysis/analysis/memory_summary.csv`
- `reproduction_A100/with_off_chip_dram/results/phase8_memory_analysis/phase8_memory_bottleneck_report.md`

## Formulas Used
- **Total Expert Size:** $\sum (size\_bytes)$ for unique tensors. The simulator collapses W1, W3, W2 into a single $A$ tensor of 352,321,536 bytes.
- **Total Footprint (GB):** $(256 \times 352,321,536) / 1,000,000,000 = 90.19$ GB.
- **Ratio T_load / T_compute:** $mean(expert\_load\_durations) / mean(expert\_FFN\_durations)$.
- **Theoretical DDR5 Bandwidth:** $3200$ MT/s $\times 2$ channels $\times 8$ bytes (assuming standard 64-bit DIMMs) = $51.2$ GB/s.
- **Theoretical Transfer Time:** $Expert Size (GB) / Bandwidth (GB/s)$.

## Units
- Latencies and durations are measured and reported natively in **nanoseconds (ns)** unless explicitly converted to milliseconds (ms) or microseconds (µs) for readability.
- Storage and weight capacities are calculated in strict **bytes**, then converted to decimal GB ($10^9$) and binary GiB ($2^{30}$) appropriately.

## Assumptions
- Iteration 0's steady-state compute (absent from `processor_trace.csv`) is assumed to be equivalent in magnitude to the recorded Iterations 3-4 (137,054 ns). 
- DDR5 peak theoretical bandwidth assumes a standard desktop/workstation 64-bit dual-channel memory setup (51.2 GB/s) as a baseline comparison.

## Limitations
- **No Iteration 0 Compute Trace:** Because the TimeBoard drops Iteration 0, end-to-end wall-clock analysis of the cold iteration is impossible; we can only compare the isolated component metrics ($T_{load}$ vs steady-state $T_{compute}$).
- **Unbounded HBM Logical Caching:** HBM is physically 80 GB, but 90.19 GB of experts are resident by Iteration 1. Hit metrics for Iterations 1-4 apply only to this unbounded logical model.
- **Trace Layer-ID Drift:** As documented, the initial Iteration 0 load events for `layer_id=0` occur at the end of the iteration and logically collide with the `resident_experts` tracker initialized by `layer_id=1` through `layer_id=31`. This artifact causes 24 early load records to register as hits instead of misses.

## Reproducibility Commands
To reproduce the numerical output:
```bash
python3 reproduction_A100/with_off_chip_dram/results/phase8_memory_analysis/analyze_phase8.py
```
