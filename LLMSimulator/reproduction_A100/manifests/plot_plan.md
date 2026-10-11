# Plot Plan (A100)

## OVERALL
- **total_execution_time**: Bar chart comparing baseline, Duplex, PE, ET.
- **speedup_vs_gpu**: Relative multipliers over baseline.
- **throughput**: Derived tokens/second (if requested).

## LATENCY
- **latency_distribution**: Boxplot of individual decoding iteration latencies across the 5 measurements.

## BREAKDOWN
- **stacked_execution_time_breakdown**: Stacked bar (Attention, MoE, FFN, Comm) by configuration.
- **communication_overhead**: Highlighting the ET penalty.

## PROCESSOR
- **gpu_vs_logic_pim_execution_time**: Dual stacked bars showing compute share.

## DEVICE
- **per_device_total_work**: Sum of latency grouped by device to highlight spatial imbalance.
- **et_device_distribution**: Showing perfectly balanced loads under ET=4.

## MOE
- **moe_compute_vs_communication**: Stacked bars isolating MoE computation from its specific `moe_all_reduce_for_e_tp` communication penalty.

## EXPERTS
- **expert_frequency_histogram**: Frequency count (out of 10,240 records) per expert.
- **expert_by_layer_heatmap**: Matrix showing routing preferences per layer.

## ROUTING
- **routing_consistency**: Matrix of Top-1 vs Top-2 overlap.

## MEMORY
- **effective_bandwidth_utilization**: Using derived traffic stats from simulator CSV.

## ENERGY
- **total_energy**: `total_energy` vs `baseline`.

## TIMELINE
- **layer_timeline_gantt**: Start/end durations of the forward pass across the MoE network.
