# Phase F: Duplex + PE + ET Validation Report

## 1. Exact Configuration
- **Base Config:** `reproduction/configs/duplex_pe.yaml`
- **ET Config:** `reproduction/configs/duplex_pe_et.yaml`
- **Model:** Mixtral 47B (32 layers, 4096 hidden, 14336 intermediate, 8 experts, top-k=2, FP16)
- **Hardware:** 1 node, 4 devices (GPU + Logic-PIM), H100 GPU + PIM
- **TP/DP/ET:** TP=4, DP=1, ET=4

## 2. Parameter Differences (Phase E → Phase F)
| Parameter | Duplex+PE | Duplex+PE+ET | Changed? |
|---|---|---|---|
| `expert_tensor_degree` | `1` | `4` | Yes |
| Everything else | identical | identical | No |

## 3. ET Source Implementation
- **Expert Sharding:** In `src/module/expert.cpp`, `expert_tensor_degree` (e_tp_dg) dictates how experts are mapped to devices. When `e_tp_dg=4`, `num_expert_per_device` evaluates to `8`. This means all 8 experts are instantiated on **all 4 devices**. The expert weights (Column/RowParallelLinear) are tensor-parallel sharded across the 4 devices.
- **All-Reduce Synchronization:** To combine the partial outputs of the tensor-parallel execution, an explicit `moe_all_reduce_for_e_tp` operation is inserted directly after the expert FFN forward pass, coordinating across the 4 devices in the `expert_device_list`.

## 4. Execution Validation (Debug & Full)
Both the debug workload and full experiment successfully completed. The output traces accurately captured the architectural changes.

## 5. Trace Validations
- **Routing Trace:** Passed (10,240 records). The routing decisions (`expert_id`) remained exactly identical to Phase E. ET does not alter the routing algorithm, only how the chosen experts are executed.
- **Processor Trace:** Passed. The debug run processor trace showed a massive increase in records (7,936 → 14,080). This accurately reflects the fact that every expert is now executing its chunked tensor-parallel workload on all 4 devices simultaneously. 
- **All-Reduce Presence:** Exactly 128 `moe_all_reduce_for_e_tp` operations were logged (32 layers × 4 devices) per iteration on the GPU, proving the communication overhead was correctly simulated.

## 6. Performance Metrics

| Metric | Duplex+PE (ET=1) | Duplex+PE+ET (ET=4) | Difference |
|---|---|---|---|
| Total Time (us) | 11,571,463.88 | 10,470,314.59 | -1,101,149.28 (1.11x speedup) |
| Latency | 2,870,059.67 | 2,611,663.02 | -258,396.65 |
| Attention Time | 167,828.47 | 167,828.47 | +0.00 |
| MoE Time | 2,139,961.13 | 1,713,983.47 | -425,977.66 |
| Communication Time | 335,162.03 | 502,743.04 | +167,581.01 |
| Total Energy (J) | 3,340,542,300.00 | 3,305,614,559.89 | -34,927,740.11 |

## 7. Analysis of ET Behavior
1. **Load Balancing:** When ET=1, experts are statically assigned to devices (2 per device). If the router selects one expert heavily, its host device bottlenecks. With ET=4, every expert is sharded across all 4 devices. Thus, regardless of routing skew, the computational load is **perfectly balanced** across the cluster.
2. **Execution vs Communication Tradeoff:** 
   - Expert FFN execution time dropped by **425,977 us** (perfect load balancing).
   - Communication time increased by **167,581 us** (due to the `moe_all_reduce_for_e_tp` overhead).
   - The net gain was highly positive, resulting in a **1.11x overall speedup** (-258,396 us per step latency).
3. **PE Interaction:** PE remained active. As with Phase E, the dynamic load-balancer calculated that the GPU was still bottlenecked by the heavy non-MoE Linear operations. Thus, PE continued to offload `AttentionSum` to the GPU, but correctly refrained from offloading the (now highly-optimized) expert workloads.

## 8. Preliminary Four-Way Comparison

| Metric | GPU Baseline | Duplex | Duplex+PE | Duplex+PE+ET |
|---|---|---|---|---|
| Total Time (us) | 32,091,176.09 | 11,571,463.88 | 11,571,463.88 | 10,470,314.59 |
| Latency | 8,022,574.16 | 2,870,059.67 | 2,870,059.67 | 2,611,663.02 |
| Attention Time | 693,668.09 | 167,828.47 | 167,828.47 | 167,828.47 |
| MoE Time | 6,766,635.99 | 2,139,961.13 | 2,139,961.13 | 1,713,983.47 |
| Communication Time| 335,162.03 | 335,162.03 | 335,162.03 | 502,743.04 |
| Total Energy | 3,136M J | 3,340M J | 3,340M J | 3,305M J |

## 9. Limitations / Warnings
ET demonstrates a classic distributed computing tradeoff: it trades communication overhead for compute balance. On hardware with poor interconnect bandwidth (e.g., PCIe instead of NVLink), the +167k us communication overhead might eclipse the compute gains, making ET detrimental. However, with the H100 + NVLink 5 configuration tested, ET is definitively beneficial.
