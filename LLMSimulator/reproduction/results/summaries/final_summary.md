# Final Executive Summary

## 1. Experiment Objective
To evaluate the performance of sparse Mixture-of-Experts (MoE) routing and execution on the Duplex hardware architecture (GPU + Logic-PIM) running a Mixtral 47B inference decode workload.

## 2. Four Configurations
1. **GPU Baseline**: Standard GPU-only execution.
2. **Plain Duplex**: Offloads MoE FFNs to Logic-PIM.
3. **Duplex + PE**: Enables dynamic runtime load balancing between GPU and Logic-PIM.
4. **Duplex + PE + ET**: Enables Expert Tensor Parallelism to perfectly shard experts across devices.

## 3. Hardware/Model Configuration
- **Model:** Mixtral 47B (32 layers, 8 routed experts, top-2 routing).
- **Workload:** 32 requests, 5 token generation steps (decode-only).
- **Hardware Modeled:** 1-node, 4-device H100-equivalent GPUs with optional Logic-PIM memory modules.

## 4. Experimental Fairness
A strict configuration difference audit (`final_configuration_diff.csv`) confirmed the tests maintained perfect parity on model size, batch size, topology scale, and mathematical routing. Only the intended execution hardware flags varied.

## 5. Overall Performance
- **GPU Baseline:** 32.09M us
- **Duplex:** 11.57M us (2.77x speedup)
- **Duplex+PE:** 11.57M us (2.77x speedup)
- **Duplex+PE+ET:** 10.47M us (3.07x speedup)

## 6. Duplex Effect
**Measured:** Logic-PIM processed the MoE layers ~3x faster than the GPU.
**Interpreted:** Logic-PIM effectively bypasses the memory-bandwidth wall for sparse operations.

## 7. PE Effect
**Measured:** PE migrated `AttentionSum` to the GPU but left all `expert_FFN` work on the Logic-PIM.
**Interpreted:** The GPU was heavily bottlenecked computing dense Linear projections; shifting sparse work back to it would have degraded performance. The dynamic balancer functioned correctly.

## 8. ET Effect
**Measured:** ET induced 128 `all_reduce` synchronization events per step (~167K us overhead), but lowered absolute MoE compute time from 2.13M us to 1.71M us.
**Interpreted:** For the evaluated NVLink/PCIe5 topology, perfect spatial load balancing yields greater gains than the interconnect penalties.

## 9. Routing Findings
**Measured:** Traces across all configs were 100% identical. Predictability based on local temporal/spatial locality (17.8%) is worse than a static frequency baseline (25.9%).
**Interpreted:** Prefetching based solely on historical locality is unviable for this workload due to near-uniform routing entropy.

## 10. Memory/Bandwidth Findings
**Measured:** GPU performance was memory-bound on FFN layers but compute-bound on Linear layers.

## 11. Limitations
The evaluation relies on simulator models executing a decode-only microbenchmark. Prefill-heavy workloads will exhibit vastly different compute vs. memory constraints, thereby likely altering the effectiveness of PE and ET.

## 12. Reproducibility
Execute `reproduction/scripts/run_reproduction.sh` to fully rebuild the simulator and generate all datasets sequentially.

## 13. Workload and Token Semantics
The simulation executes a highly controlled decode microbenchmark. The nominal configuration states `input_len = 2048` and `output_len = 128`. 

In the simulator:
- **`input_len`**: Defines the pre-existing prompt context. Because `decode_mode = on`, sequences initialize their active length directly to 2048.
- **`output_len`**: Defines the lifespan of the sequence (generating 128 tokens) before it is popped and replaced.
- **`batchsize` = 32**: 32 concurrent requests are processed. Since each generates exactly 1 token per decode step, `numtoken = 32` and `num_gen_seq = 32`. 
- **Trace Starting Index (2142)**: The simulator runs a 10,000-iteration `hittingQueue` warm-up phase to reach a steady state before the measured iterations. Because each sequence's decode lifespan is exactly 127 steps (from 2048 up to 2174), the active sequences turn over multiple times. The remainder of `10,000 % 127` is `94`. Thus, the measured trace predictably begins at token index `2048 + 94 = 2142`.

The 5-iteration measured run precisely evaluates:
- **32 concurrent requests** generating 1 token per step over **5 iterations** = **160 processed tokens**.
- **160 tokens** traversing **32 MoE layers** at **Top-2 routing** = **10,240 routing records**.
