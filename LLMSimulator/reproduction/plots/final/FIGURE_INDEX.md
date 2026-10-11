# Figure Index

This document catalogs the visualizations generated during the final evaluation phase located in `reproduction/plots/final/`.

| Figure | Filename | Purpose | Source CSV | Main Interpretation |
|---|---|---|---|---|
| 1 | `overall/total_execution_time.png` | Show gross latency comparison | `final_comparison.csv` | Duplex massively accelerates execution; ET adds slight further gain. |
| 2 | `overall/speedup_vs_gpu.png` | Normalize performance relative to GPU | `final_comparison.csv` | 2.77x to 3.07x speedup boundaries achieved. |
| 3 | `breakdown/stacked_execution_time_breakdown.png` | Isolate component contributions | `final_comparison.csv` | The memory-bound MoE phase shrinks dramatically, bottleneck shifts to Linear computation. |
| 4 | `processor/gpu_vs_logic_pim_execution_time.png` | Highlight device utilization | `final_comparison.csv` | Logic-PIM handles a tiny fraction of absolute wall-clock time due to high bandwidth, while GPU struggles with compute. |
| 5 | `energy/total_energy.png` | Compare energetic efficiency | `final_comparison.csv` | Logic-PIM adds slight absolute power cost, but ET regains efficiency by shortening execution. |
