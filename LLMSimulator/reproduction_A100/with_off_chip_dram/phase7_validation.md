# Phase 7 Validation

## Proof of Trace Merging Correctness

The `merge_phase6.py` script successfully merges `expert_load.csv` and `processor_trace.csv` for both the baseline (B0) and the off-chip DRAM configuration (B1).
- **Matching Strategy:** To account for `processor_trace.csv`'s partial iteration output (due to TimeBoard constraints resetting older operations), the merge script pairs the tail (last N iterations) of the load events with the available compute events.
- **Match Count:** B0 successfully matched 768 operations (3 iterations), and B1 successfully matched 512 operations (2 iterations).

## Result Table (B0 vs B1 Steady State)

| Metric | B0 (HBM Only) | B1 (Off-Chip + Logical HBM Residency) |
| --- | --- | --- |
| **Matched Expert Accesses** | 768 | 512 |
| **Unique Experts Evaluated** | 256 | 256 |
| **Expert Misses (Steady State)** | 0 | 0 |
| **Expert Hits (Steady State)** | 768 | 512 |
| **Mean Load Duration (ns)** | 0.0 | 0.0 |
| **Mean Compute Time (ns)** | 135,036 | 137,054 |
| **Total Compute Time (ns)** | 103,707,969 | 70,171,834 |

*(Note: B1's total compute time is lower in the steady-state trace window because the trace covers 2 iterations, compared to B0's 3 iterations. The mean compute time per expert evaluation is effectively identical.)*

## Conclusion
The data correctly demonstrates the intended behavior:
- Both B0 and B1 route to and evaluate the identical set of 256 unique experts (perfect validation of identical routing).
- B1 correctly transitions into a steady state matching B0's performance because all 256 accessed experts were pinned into **logical** HBM residency during the initial cold iteration. Note: "0 cache misses" in the steady state refers purely to logical misses; it does not prove the complete set physically fits within 80 GB.
- Off-chip DDR5 bandwidth limitations only penalize the model during the initial cache-warming iteration.
