# Final Processor Trace Validation

Data source: `processor_trace.csv` from all four target configurations.

## Summary

- **Required columns present:** Yes (`device_id`, `layer_id`, `operation`, `processor`, `start_time`, `end_time`, `duration`).
- **Valid processor names:** Yes (`GPU`, `Logic`).
- **Non-negative durations:** Verified.
- **Valid timestamps:** Start and End times strictly monotonically increasing within device boundaries.
- **No malformed rows:** Verified.

## Hardware Isolation Verification
- **GPU Baseline Trace:** Contains strictly `GPU` records. Logic-PIM records correctly do not exist.
- **Duplex Traces:** Correctly multiplex operations onto `GPU` (Linear, Attention) and `Logic` (MoE Expert FFNs).
- **PE Migration Traces:** Track exactly the configured offloads (e.g., AttentionSum migrating back to GPU).
- **ET Synchronization Traces:** Track `moe_all_reduce_for_e_tp` natively occurring across the cluster.
