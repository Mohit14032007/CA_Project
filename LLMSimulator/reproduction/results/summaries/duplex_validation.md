# Phase C: Plain Duplex Validation Report

## Configuration
- **Path:** `reproduction/configs/duplex.yaml`
- **Model:** Mixtral 47B, 32 layers, hidden dimension 4096, intermediate 14336, 32 attention heads, 8 KV heads, 8 routed experts, top-k = 2, FP16
- **Hardware:** 1 node, 4 devices, H100 GPU equivalent
- **Processor Type:** `GPU+LOGIC` (High Processor: GPU, Low Processor: LOGIC)
- **Parallelism (TP/DP):** `none_expert_tensor_degree: 4` (TP=4), DP=1
- **Expert Tensor Degree:** 1
- **Parallel Execution (PE):** `off` (Disabled)
- **Workload:** Synthesis, input_len 2048, output_len 128, decode_mode on, maxbatch=32, iter=5.

## Deviations from Intended Configuration
None. All baseline Phase B configurations (model, precision, workload, hardware count) were preserved. The only parameter changed is `processor_type` to `GPU+LOGIC`.

## Configuration Differences
```
Parameter                | GPU Baseline         | Plain Duplex         | Changed?
-------------------------|----------------------|----------------------|---------
processor_type           | GPU                  | GPU+LOGIC            | Yes
parallel_execution       | off                  | off                  | No
expert_tensor_degree     | 1                    | 1                    | No
model_name               | mixtral              | mixtral              | No
```

## Validation Checks

### Debug Run Result
A debug run using `iter: 1` successfully loaded 4 devices with `GPU+LOGIC` enabled, without triggering parallel execution (PE). The simulation successfully completed without any fallback.

### Traces Verified
1. **Routing Trace:** `reproduction/results/traces/duplex/expert_routing.csv` generated correctly with `request_id`, token information, and expert IDs. 
2. **Processor Trace:** `reproduction/results/traces/duplex/processor_trace.csv` verified.
   - **GPU execution evidence:** Found in qkv_proj and o_proj operations.
   - **Logic-PIM execution evidence:** Found managing AttentionGen, self-attention, up_proj, and down_proj linear layers.

## Main Metrics and Phase B Comparison

Metrics represent iteration 1 measurements (decode latency per token):

| Metric | GPU Baseline | Plain Duplex | Difference |
| :--- | :--- | :--- | :--- |
| **Total Decode Latency (per step)** | ~8.02 ms | ~2.88 ms | -5.14 ms |
| **Attention Gen Time** | 0.69 ms | 0.17 ms | -0.52 ms |
| **MoE Expert FFN Time** | 6.77 ms | 2.15 ms | -4.62 ms |
| **Communication Time** | 0.335 ms | 0.335 ms | No change |
| **Total Energy** | ~3.136 x 10^9 | ~3.340 x 10^9 | +6.5% |

**Calculated Metric:** 
- **Speedup:** `8.02 ms / 2.88 ms = 2.78x` (Calculated ratio)

## Conclusion
Phase C PASS. Plain Duplex successfully executes memory-bound logic operations on Logic-PIM, driving end-to-end latency down significantly as the baseline GPU operations were heavily bottlenecked. The architecture functions as intended.
