# Final Artifact Index

## 1. Configurations
**Path:** `reproduction/configs/`
**Status:** Authoritative (Inputs)
- `gpu_baseline.yaml`
- `duplex.yaml`
- `duplex_pe.yaml`
- `duplex_pe_et.yaml`

## 2. Raw Outputs
**Path:** `reproduction/results/raw/`
**Status:** Authoritative (Raw Simulator Outputs)
Contains device-level Gantt execution logs for each configuration. Note that for non-expert layers, all 4 devices are functionally identical due to TP=4. Expert layers diverge according to device allocation (ET=1 vs ET=4).

## 3. CSV Outputs
**Path:** `reproduction/results/csv/`
**Status:** Authoritative
- `final_results_table.csv`
- `<config>/performance_summary.csv`

## 4. Routing Traces
**Path:** `reproduction/results/traces/<config>/expert_routing.csv`
**Status:** Authoritative
Logs every token's path through the MoE layers.

## 5. Processor Traces
**Path:** `reproduction/results/traces/<config>/processor_trace.csv`
**Status:** Authoritative
Appends full, device-aware operational timing and processor assignments for all layers.

## 6. Analysis CSVs
**Path:** `reproduction/results/csv/`
**Status:** Non-authoritative (Derived from traces)
Includes summaries such as `final_comparison.csv`, `time_breakdown.csv`, `processor_summary.csv`, etc.

## 7. Plots
**Path:** `reproduction/plots/final/presentation/`
**Status:** Non-authoritative (Visualizations)
- `device_execution_comparison.png`
- `energy_comparison.png`
- `moe_vs_attention_time.png`
- `speedup_comparison.png`
- `time_breakdown_by_config.png`

## 8. Reports
**Path:** `reproduction/results/summaries/`
**Status:** Interpretative 
- `FINAL_REPORT.md`: Comprehensive final analysis.
- `FINAL_ARTIFACT_INDEX.md`: This file.
- `presentation.md`: Presentation slide outline.
- `device_trace_validation.md`: Proof of device-aware trace accuracy.
