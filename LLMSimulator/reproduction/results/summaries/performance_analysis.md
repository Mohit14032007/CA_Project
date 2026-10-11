# Phase H: Processor and Performance Analysis Summary

This document summarizes the execution behaviors of the four studied hardware/optimization configurations under a pure-decode workload (Mixtral 47B, 32 requests). It strictly isolates execution performance from the previously validated routing semantics (Phase G).

## 1. Overall Performance
Data source: `reproduction/results/csv/performance_summary.csv`

| Configuration | Total Time (us) | Latency (us) | Speedup vs Baseline |
|---|---|---|---|
| GPU Baseline | 32,091,176.09 | 8,022,574.16 | 1.00x |
| Duplex (Plain) | 11,571,463.88 | 2,870,059.67 | 2.77x |
| Duplex + PE | 11,571,463.88 | 2,870,059.67 | 2.77x |
| Duplex + PE + ET | 10,470,314.59 | 2,611,663.02 | 3.07x |

*Measured Observation:* Integrating Logic-PIM (Duplex) provides an immediate 2.77x speedup. Adding Expert Tensor Parallelism (ET) pushes the total speedup to 3.07x.

## 2. Time Breakdown
Data source: `reproduction/results/csv/time_breakdown.csv`

The execution time is largely dominated by MoE layers and non-MoE linear operations.
- **GPU Baseline:** MoE accounts for ~21% of execution time.
- **Duplex:** MoE accounts for ~18% of execution time (absolute time dropped drastically from 6.7M us to 2.1M us).
- **Communication:** Static overhead across GPU/Duplex/Duplex+PE (335K us), but increases to 502K us under ET due to the `all_reduce_for_e_tp` penalty.

## 3. Processor Usage
Data source: `reproduction/results/csv/processor_summary.csv` and `processor_operation_summary.csv`

In Duplex architectures, the processor trace shows distinct specialization:
- **GPU:** Handles exclusively `AttentionGen`, `Linear`, `AttentionSum`, and `activation`.
- **Logic-PIM:** Handles MoE `expert_FFN`, `moe_gather`, `moe_scatter`.
- **Fraction of Processor Time:** Because Logic-PIM is highly efficient for bandwidth-bound FFNs, it executes the entire MoE workload in ~8.2M ns, while the GPU grinds on the compute-bound linear layers for ~44M ns per iteration.

## 4. MoE Behavior
Data source: `reproduction/results/csv/moe_performance.csv`

The migration from GPU to Logic-PIM yields a massive reduction in MoE execution time (from 6.76M us down to 2.13M us), demonstrating the PIM architecture's core advantage for memory-bound sparse operations.

## 5. PE (Parallel Execution) Behavior
Data source: `reproduction/results/csv/pe_analysis.csv`

- PE was designed to dynamically load-balance `AttentionSum` and expert FFNs between GPU and Logic-PIM.
- *Measured Observation:* The trace confirms exactly 256 `AttentionSum` operations shifted from Logic-PIM to the GPU.
- However, zero `expert_FFN` work shifted. The dynamic load balancer correctly calculated that the GPU was already the bottleneck (due to heavy Linear layer processing in this decode workload), so migrating additional compute to the GPU would have worsened performance. Consequently, total execution time remained identical (11.57M us).

## 6. ET (Expert Tensor Parallelism) Behavior
Data source: `reproduction/results/csv/et_analysis.csv`

- **Execution Change:** Under ET=4, all 8 experts are sharded across all 4 devices rather than statically mapped (2 per device). The trace logged a massive increase in parallel operations (processor trace records increased from 7,936 to 14,080).
- **Performance:** Perfect spatial load balancing reduced MoE computation time from 2.13M us to 1.71M us.
- **Overhead:** Exactly 128 `moe_all_reduce_for_e_tp` operations were tracked per step, costing ~167K us in communication overhead.
- **Net Impact:** The compute gain outweighed the communication overhead, yielding a 1.11x speedup over Duplex+PE.

## 7. Device Load Balance
Data source: `reproduction/results/csv/device_load_balance.csv`

The `expert_load_balance.csv` from Phase G proved the *algorithm* was already load balanced. However, the *processor traces* show that local device compute times vary slightly. ET perfectly flattens these variations by ensuring every device executes 1/4th of every expert.

## 8. Memory / Bandwidth
Data source: simulator raw outputs.

The extreme speedup in MoE layers (Duplex vs GPU) is the direct result of shifting from GPU memory limits to the massive internal bandwidth of the Logic-PIM. The heavy linear projections remain on the GPU, bottlenecking the system.

## 9. Communication
Data source: `reproduction/results/csv/communication_analysis.csv`

Communication represents a small fraction of total time (~1-4%) in this configuration.
- Duplex/Duplex+PE: 335,162.03 us
- Duplex+PE+ET: 502,743.04 us
The +167k us delta represents the synchronization required to merge the tensor-parallel expert shards across devices.

## 10. Energy
Data source: `reproduction/results/csv/energy_analysis.csv`

| Configuration | Total Energy (J) |
|---|---|
| GPU Baseline | 3,136M |
| Duplex (Plain) | 3,340M |
| Duplex+PE | 3,340M |
| Duplex+PE+ET | 3,305M |

*Measured Observation:* Duplex consumes slightly more energy than the GPU baseline, trading power efficiency for a massive 2.77x latency reduction. Activating ET slightly reduces energy by accelerating overall execution and eliminating idle static power.

## 11. Cross-Configuration Explanation

1. **GPU → Duplex:** Migrates memory-bound MoE operations to Logic-PIM. Drastically reduces MoE time, providing a 2.77x speedup. The GPU becomes the new bottleneck due to linear projections.
2. **Duplex → Duplex+PE:** The PE dynamic balancer activates. It shifts `AttentionSum` to the GPU but refuses to shift `expert_FFN` because the GPU is already bottlenecked. Net speedup: 1.00x.
3. **Duplex+PE → Duplex+PE+ET:** Expert weights are sharded across all devices. The compute load perfectly balances across the PIMs, reducing MoE execution time. The added all-reduce overhead is eclipsed by the compute gains. Net speedup: 1.11x (3.07x overall).

## 12. Main Measured Observations
- The Logic-PIM effectively eliminates the memory-wall for MoE sparse operations.
- The dynamic load balancer correctly adapts to workload constraints without manual tuning.
- ET provides a robust distributed tradeoff: it purchases perfect compute balance at the cost of `all_reduce` synchronization overhead.

## 13. Limitations
- **Workload specificity:** This decode-only microbenchmark artificially bottlenecks the GPU with linear projections. A prefill workload would likely shift the bottleneck and drastically alter the PE load-balancing decisions.
- **Interconnect Assumptions:** The ET communication overhead (+167K us) assumes high-bandwidth NVLink/PCIe 5 semantics. On weaker interconnects, ET would likely degrade performance.
- **Energy modeling:** The energy numbers are derived from the simulator's theoretical hardware profiles, not direct physical telemetry.
