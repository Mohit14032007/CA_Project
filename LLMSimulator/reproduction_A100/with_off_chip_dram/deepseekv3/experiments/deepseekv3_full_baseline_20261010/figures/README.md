# Figure Index & Reproduction Guide

**Experiment:** DeepSeek-V3 Full Baseline (`deepseekv3_full_baseline_20261010`)  
**Directory:** `reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/figures/`  
**Date:** October 2026  

All figures below are generated programmatically from the processed telemetry datasets using [`scripts/generate_graphs.py`](file:///home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/scripts/generate_graphs.py).

---

## Index of Figures

| Figure | Image File | Vector PDF | Plotted Fields & Description | Input Dataset |
| :--- | :--- | :--- | :--- | :--- |
| **Fig 1** | `fig1_routing_coverage.png` | `fig1_routing_coverage.pdf` | Routed expert count per MoE layer (8 per layer) across Pass 1 (cold start) and Pass 2 (warm replay). | `processed/per_layer_metrics.csv` |
| **Fig 2** | `fig2_expert_selection_heatmap.png` | `fig2_expert_selection_heatmap.pdf` | 2D heatmap of expert selection matrix ($57\text{ layers} \times 256\text{ experts}$) under synthetic PRNG routing. | `raw/expert_routing.csv` |
| **Fig 3** | `fig3_cold_loaded_experts_per_layer.png` | `fig3_cold_loaded_experts_per_layer.pdf` | Number of cold-loaded routed experts per layer (8 per layer, total = 456). | `processed/per_layer_metrics.csv` |
| **Fig 4** | `fig4_offchip_bytes_per_layer.png` | `fig4_offchip_bytes_per_layer.pdf` | Valid off-chip DDR5 batch volume per layer (704.64 MB per layer, total = 40.16 GB). | `processed/per_layer_metrics.csv` |
| **Fig 5** | `fig5_cache_event_composition.png` | `fig5_cache_event_composition.pdf` | Discrete count of cache events: 456 MISS, 456 INSERT, 2280 HIT, 0 EVICT. | `processed/cache_event_counts.csv` |
| **Fig 6** | `fig6_cache_occupancy_evolution.png` | `fig6_cache_occupancy_evolution.pdf` | Resident weight data (GB) vs. sequential cache event index against the 80 GB HBM capacity limit. | `raw/expert_cache.csv` |
| **Fig 7** | `fig7_ddr5_batch_duration_distribution.png` | `fig7_ddr5_batch_duration_distribution.pdf` | Latency distribution histogram and boxplot for 456 off-chip DDR5 batches (mean = 3.9011 ms). | `raw/expert_load.csv` |
| **Fig 8** | `fig8_per_layer_simulated_ddr5_time.png` | `fig8_per_layer_simulated_ddr5_time.pdf` | Simulated hardware DDR5 transfer time per MoE layer (~31.21 ms per layer, total = 1.7789 s). | `processed/per_layer_metrics.csv` |
| **Fig 9** | `fig9_host_vs_simulated_time.png` | `fig9_host_vs_simulated_time.pdf` | Comparison of CPU host runtime (4,439 s) vs. simulated hardware DRAM time (1.7789 s) illustrating ~2,491× simulation slowdown. | `processed/summary_metrics.json` |
| **Fig 10** | `fig10_expert_load_timeline.png` | `fig10_expert_load_timeline.pdf` | Cumulative simulated hardware time progression across the 456 cold batch completions. | `raw/expert_load.csv` |

---

## Reproduction Command

To regenerate all 10 figures:
```bash
python3 reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/scripts/generate_graphs.py
```
