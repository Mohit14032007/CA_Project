# Final Artifact Inventory

## Configuration Files (`reproduction/configs/`)
- `gpu_baseline.yaml`: Authoritative configuration for GPU baseline
- `duplex.yaml`: Authoritative configuration for Plain Duplex
- `duplex_pe.yaml`: Authoritative configuration for Duplex + PE
- `duplex_pe_et.yaml`: Authoritative configuration for Duplex + PE + ET
- `debug_instrumentation.yaml`: Intermediate debugging config (obsolete)
- `debug_gpu_baseline.yaml`, etc: Intermediate configs (obsolete)

## Results CSV (`reproduction/results/csv/`)
- `performance_summary.csv`: Core simulator measured performance
- `final_comparison.csv`: Normalized comparison across configs
- `final_configuration_diff.csv`: Evaluated differences in configs
- `final_device_comparison.csv`: Device load balance
- `final_moe_comparison.csv`: MoE-specific isolation metrics
- Routing CSVs (`routing_trace_validation.csv`, `routing_entropy.csv`, etc): Authoritative routing validation files.

## Traces (`reproduction/results/traces/`)
- `gpu_baseline/`
- `duplex/`
- `duplex_pe/`
- `duplex_pe_et/`
(Each contains `processor_trace.csv`, `expert_routing.csv`, and simulator raw logging output).

## Summaries (`reproduction/results/summaries/`)
- All Phase-specific `.md` reports are preserved as the historical audit trail.
- `FINAL_REPORT.md`: The authoritative final summary.

## Missing/Duplicated
No required components missing. Debug configurations (`debug_*`) exist but are appropriately marked as non-authoritative intermediate files.
