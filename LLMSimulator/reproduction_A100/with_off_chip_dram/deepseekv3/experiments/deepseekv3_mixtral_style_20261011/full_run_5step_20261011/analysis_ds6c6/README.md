# DS-6C-6 Analysis Artifacts and Dataset Package Directory

`reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/`

This directory contains the complete, self-contained research results package generated for Phase DS-6C-6 from the completed 5-step DeepSeek-V3 memory-system simulation.

---

## Directory Organization

```text
analysis_ds6c6/
├── csv/                            # 41 Research CSV Datasets
│   ├── run_summary.csv
│   ├── timing_breakdown.csv
│   ├── per_token_timing.csv
│   ├── per_layer_timing.csv
│   ├── load_latency_distribution.csv
│   ├── load_volume_by_token.csv
│   ├── load_volume_by_layer.csv
│   ├── load_batch_details.csv
│   ├── routing_frequency_by_expert.csv
│   ├── routing_frequency_by_layer_expert.csv
│   ├── routing_frequency_by_layer.csv
│   ├── routing_rank_distribution.csv
│   ├── routing_selection_matrix.csv
│   ├── per_token_routing_summary.csv
│   ├── temporal_jaccard_by_stage.csv
│   ├── temporal_jaccard_by_transition.csv
│   ├── temporal_transition_counts.csv
│   ├── spatial_overlap_by_adjacent_stage.csv
│   ├── spatial_transition_counts.csv
│   ├── routing_correlation_summary.csv
│   ├── random_routing_baseline.csv
│   ├── multi_step_temporal_lag.csv
│   ├── multi_hop_spatial_distance.csv
│   ├── oracle_delta_distribution.csv
│   ├── cumulative_prefetch_coverage.csv
│   ├── prefetch_overlap_timing_bounds.csv
│   ├── cache_event_summary.csv
│   ├── cache_event_timeline.csv
│   ├── cache_occupancy_timeline.csv
│   ├── cache_evictions_by_expert.csv
│   ├── cache_evictions_by_layer.csv
│   ├── cache_hits_by_layer.csv
│   ├── cache_misses_by_layer.csv
│   ├── cache_capacity_validation.csv
│   ├── cache_residency_invariants.csv
│   ├── artifact_inventory.csv
│   ├── integrity_validation.csv
│   ├── metric_provenance.csv
│   ├── model_configuration.csv
│   ├── hardware_configuration.csv
│   └── source_function_inventory.csv
├── figures/
│   ├── png/                        # 30 High-Resolution (300 DPI) Publication Figures
│   │   ├── fig1_system_architecture.png
│   │   ├── fig2_execution_flow.png
│   │   ├── ... (Figures 1 through 30)
│   │   ├── fig30_attention_window_vs_transfer_bounds.png
│   │   └── FIGURES_EXPLANATION_AND_METHODOLOGY.md (Comprehensive analytical guide)
│   └── pdf/                        # 30 Vector PDF Publication Figures
│       ├── fig1_system_architecture.pdf
│       ├── ... (Figures 1 through 30)
│       └── fig30_attention_window_vs_transfer_bounds.pdf
├── manifests/
│   └── results_package_manifest.csv # Cryptographic SHA-256 and size audit of all files (105 entries)
└── scripts/
    ├── generate_csv_datasets.py    # Generates base 36 CSVs from raw traces
    ├── generate_figures.py         # Generates base 24 PNG/PDF figures from CSVs
    └── generate_additional_correlation_figures.py # Generates Figures 25-30 and 5 new CSVs (ST-MoE & APEX inspired)
```

---

## Dataset Categorization (36 CSV Files)

### 1. Run and Timing Results (8 CSVs)
- [`csv/run_summary.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/run_summary.csv): High-level run metrics (5 steps, 2,280 routing selections, 2,205 cold batches, 194.22 GB DDR5 transfer, 8,601.93 ms DDR5 stalls, 8,713.53 ms device latency, exit code 0).
- [`csv/timing_breakdown.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/timing_breakdown.csv): Percentage breakdown between off-chip DDR5 transfer stalls (98.72%), internal simulated GPU/HBM compute (1.28%), and analytical peak roofline compute (0.0078%).
- [`csv/per_token_timing.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/per_token_timing.csv): Step-by-step breakdown across tokens 2048 to 2052.
- [`csv/per_layer_timing.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/per_layer_timing.csv): Total and average DDR5 stalls per aligned MoE execution stage (Stages 1..57).
- [`csv/load_latency_distribution.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/load_latency_distribution.csv): Min, percentiles (p1..p99), median (3.901 ms), mean (3.901 ms), and std of individual 88 MB batch transfers.
- [`csv/load_volume_by_token.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/load_volume_by_token.csv): Cold batch counts and transfer volumes in decimal GB and binary GiB.
- [`csv/load_volume_by_layer.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/load_volume_by_layer.csv): Transferred MB and average duration per MoE execution stage.
- [`csv/load_batch_details.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/load_batch_details.csv): Exact record for all 2,205 individual cold batch loads.

### 2. Routing and Expert Statistics (18 CSVs)
- [`csv/routing_frequency_by_expert.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/routing_frequency_by_expert.csv): Selection counts, frequencies, and z-scores for all 256 candidate experts.
- [`csv/routing_frequency_by_layer_expert.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/routing_frequency_by_layer_expert.csv): Stage-by-expert frequency counts.
- [`csv/routing_frequency_by_layer.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/routing_frequency_by_layer.csv): Unique expert count per stage compared to uniform expectation (37.07).
- [`csv/routing_rank_distribution.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/routing_rank_distribution.csv): Mean and std of assigned expert IDs across top-k ranks 1..8.
- [`csv/routing_selection_matrix.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/routing_selection_matrix.csv): Binary selection indicator matrix (285 token-stage rows x 256 expert columns).
- [`csv/per_token_routing_summary.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/per_token_routing_summary.csv): Unique experts and lookup hit rates per token.
- [`csv/temporal_jaccard_by_stage.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/temporal_jaccard_by_stage.csv): Mean Jaccard similarity across 4 consecutive token transitions per MoE stage.
- [`csv/temporal_jaccard_by_transition.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/temporal_jaccard_by_transition.csv): Mean Jaccard and overlap fractions for each transition (2048->2049, 2049->2050, etc.).
- [`csv/temporal_transition_counts.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/temporal_transition_counts.csv): Observed intersection size distribution (k=0..8) vs Hypergeometric null.
- [`csv/spatial_overlap_by_adjacent_stage.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/spatial_overlap_by_adjacent_stage.csv): Overlap between adjacent MoE stages (L -> L+1).
- [`csv/spatial_transition_counts.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/spatial_transition_counts.csv): Observed spatial overlap distribution vs Hypergeometric null.
- [`csv/routing_correlation_summary.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/routing_correlation_summary.csv): Comprehensive summary comparing observed correlations against analytical models.
- [`csv/random_routing_baseline.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/random_routing_baseline.csv): Exact mathematical reference distribution for independent top-8 draws from 256 candidates.
- [`csv/multi_step_temporal_lag.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/multi_step_temporal_lag.csv): Multi-step temporal autocorrelation decay across lags 1..4 (ST-MoE Fig 2b diagnostic).
- [`csv/multi_hop_spatial_distance.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/multi_hop_spatial_distance.csv): Cross-layer routing correlation across stage distance $\Delta L \in \{1 \dots 50\}$ (ST-MoE Fig 2a diagnostic).
- [`csv/oracle_delta_distribution.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/oracle_delta_distribution.csv): Distribution of Oracle extra prefetch budget $\delta^*$ required for 100% coverage (APEX Fig 6a diagnostic).
- [`csv/cumulative_prefetch_coverage.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/cumulative_prefetch_coverage.csv): Recall curves under budget expansion $(k+\delta)$ vs Hypergeometric bound (APEX Fig 3 style).
- [`csv/prefetch_overlap_timing_bounds.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/prefetch_overlap_timing_bounds.csv): Attention computation window vs transfer latency bounds across interconnects (APEX Eq 12 diagnostic).

### 3. Cache and Memory-System Behaviour (9 CSVs)
- [`csv/cache_event_summary.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/cache_event_summary.csv): Summary of HIT (4,635), MISS (2,205), INSERT (2,205), and EVICT (1,297) events.
- [`csv/cache_event_timeline.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/cache_event_timeline.csv): Complete chronological sequence of all 10,342 cache events.
- [`csv/cache_occupancy_timeline.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/cache_occupancy_timeline.csv): Sampled milestones tracking cache growth from 0 to 79.98 GB and steady-state saturation.
- [`csv/cache_evictions_by_expert.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/cache_evictions_by_expert.csv): Eviction frequency distribution across 256 experts.
- [`csv/cache_evictions_by_layer.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/cache_evictions_by_layer.csv): Evictions recorded per layer ID.
- [`csv/cache_hits_by_layer.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/cache_hits_by_layer.csv): Hits recorded per layer ID.
- [`csv/cache_misses_by_layer.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/cache_misses_by_layer.csv): Misses recorded per layer ID.
- [`csv/cache_capacity_validation.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/cache_capacity_validation.csv): Proof of cache capacity adherence (Peak 79.98 GB <= 80.00 GB).
- [`csv/cache_residency_invariants.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/cache_residency_invariants.csv): Verification of mathematical invariants governing weights, lookups, inserts, and evictions.

### 4. Provenance and Integrity (6 CSVs)
- [`csv/artifact_inventory.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/artifact_inventory.csv): File sizes, line counts, and SHA-256 hashes of input traces and configs.
- [`csv/integrity_validation.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/integrity_validation.csv): 13 reference checks comparing DS-6C-4 reported totals with independent raw trace recalculations (100% Exact Match).
- [`csv/metric_provenance.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/metric_provenance.csv): Traceability matrix mapping every metric to its source trace file or C++ function.
- [`csv/model_configuration.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/model_configuration.csv): Parameters of the DeepSeek-V3 671B model architecture.
- [`csv/hardware_configuration.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/hardware_configuration.csv): Hardware specifications, clock timings, and architectural assumptions.
- [`csv/source_function_inventory.csv`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/csv/source_function_inventory.csv): C++ functions across the harness, scheduler, cache, and DRAM interface.

---

## Reproduction Commands

To reproduce all 41 CSV datasets, 30 PNG figures, 30 PDF figures, and the cryptographic manifest without rerunning Ramulator:

```bash
cd /home/mohman/ca_project/LLMSimulator

# 1. Regenerate base 36 CSV files
python3 reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/scripts/generate_csv_datasets.py

# 2. Regenerate base 24 publication figures (PNG + PDF)
python3 reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/scripts/generate_figures.py

# 3. Regenerate additional spatio-temporal figures (Figures 25-30) and datasets (ST-MoE & APEX inspired)
python3 reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/analysis_ds6c6/scripts/generate_additional_correlation_figures.py
```
