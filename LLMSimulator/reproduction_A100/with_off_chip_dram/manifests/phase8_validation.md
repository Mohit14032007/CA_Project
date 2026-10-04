# Phase 8 Validation

## Validation Checks
1. **Frozen Phase 5/6/7 CSVs**: Verified untouched. The Python script only read from `results/b1_full/traces/*.csv`.
2. **Simulator Source Files**: Verified untouched. No `.cpp` or `.h` files were modified during Phase 8.
3. **YAML Workload Configuration**: Verified untouched. `configs/b1_full.yaml` and `configs/dram_config_DDR5.yaml` were parsed for values but not altered.
4. **Phase 7 Results**: Verified not regenerated. No execution commands for the simulator were run.
5. **Analysis Script Execution**: `analyze_phase8.py` successfully completed without errors and output meaningful, consistent metrics.
6. **Generated CSV Consistency**: `memory_summary.csv` accurately aggregates the internal trace totals without hallucination.
7. **Tensor-level vs Expert-level Counting**: Correctly verified that the 3 tensors (W1, W3, W2) for a single expert are treated as a single memory block ($A$) of size 352,321,536 bytes in `expert_load.csv`. Consequently, each load event matches an expert-level load event.
8. **Units Consistency**: All time elements were maintained in nanoseconds (`ns`) internally and properly labeled when converted to `ms` or `µs`. Memory limits were processed in `bytes` and carefully mapped to `GB` / `GiB`.
9. **Negative/Impossible Timings**: Verified absent. The standard deviation for load times is strictly 0.0, indicating perfectly reproducible stall penalties.
10. **90.2 GB Logical Residency Limitation**: The limitation is explicitly documented in the Phase 8 report, the updated Phase 7 documents, and the README.
11. **processor_trace.csv Limitations**: Explicitly documented. Iteration 0's absence and the steady-state isolation of the trace were heavily emphasized in all relevant Phase 8 reports.
12. **No False Capacity Claims**: No conclusion implies physical 80 GB HBM enforcement. All claims carefully emphasize the *unbounded logical residency* mechanism.
