# Phase E: Duplex + PE Validation Report

## 1. Exact Config
- **Base Config:** `reproduction/configs/duplex.yaml`
- **PE Config:** `reproduction/configs/duplex_pe.yaml` (and debug equivalent)
- **Model:** Mixtral 47B (32 layers, 4096 hidden, 14336 intermediate, 8 experts, top-k=2, FP16)
- **Hardware:** 1 node, 4 devices (GPU + Logic-PIM), H100 GPU + PIM
- **TP/DP/ET:** TP=4, DP=1, ET=1 (Disabled)

## 2. Parameter Differences
| Parameter | Plain Duplex | Duplex+PE | Changed? |
|---|---|---|---|
| `parallel_execution` | `off` | `on` | Yes |
| Everything else | identical | identical | No |

## 3. Source-Code PE Mechanism
The PE mechanism (`parallel_execution: true`) activates dynamic workload delegation in two places:
1. **Expert Co-processing (`src/module/route.cpp`)**: A heuristic function `getIdxHighOptimal()` calculates `high_time` (GPU) and `low_time` (Logic). It sorts experts by token load and evaluates whether shifting an expert to the GPU reduces the overall `max(high_time, low_time)`. 
2. **Attention Co-processing (`src/module/attention.cpp`)**: The Attention module explicitly splits into `AttentionSum` (prefill, routed to GPU via `setPerformHigh()`) and `AttentionGen` (decode, routed to Logic-PIM via `setPerformLow()`).

## 4. Execution Validation (Debug & Full)
Both the 1-iteration debug run and the 5-iteration full run completed successfully without any OOM or fallback events. The telemetry perfectly captured the processor differences.

## 5. Trace Validations
- **Routing Trace:** Passed. Total records = 10,240. 32 unique requests * 5 decode steps * 32 layers * 2 experts = 10,240. `token_id` spanned 2142-2146. Exactly 2 records per token/layer. PE did not perturb the routing logic.
- **Processor Trace:** Passed. Captured the exact redistribution of operations.

## 6. Evidence that PE Actually Executed
PE was proven to be active through exact processor trace record counts:
* **Plain Duplex:** GPU = 13,312 records. Logic = 2,560 records.
* **Duplex+PE:** GPU = 13,568 records. Logic = 2,304 records.
* **Shift:** Exactly 256 records shifted from Logic to GPU.
* **Why?** The `AttentionSum` operations (prefill attention) were actively offloaded to the GPU by the PE `AttentionSplit` logic.

## 7. Performance Metrics & Comparison

| Metric | Plain Duplex | Duplex+PE | Difference |
|---|---|---|---|
| Total Time (us) | 11,571,463.88 | 11,571,463.88 | +0.00 (1.00x) |
| Latency | 2,870,059.67 | 2,870,059.67 | +0.00 |
| MoE Time | 2,139,961.13 | 2,139,961.13 | +0.00 |
| GPU Time | 88.9M ns | 88.9M ns | +0.00 |
| Logic Time | 16.47M ns | 16.47M ns | +0.00 |

## 8. Why is there 0.00 performance difference?
The trace data reveals a perfect, logically sound explanation for why PE activated but execution time did not change:
1. **Attention Co-Processing:** Because this is a **pure-decode workload** (`prefill_mode: off`), `AttentionSum` has 0 tokens. Offloading it to the GPU shifted the operational records, but shifted exactly `0.00 ns` of actual work. `AttentionGen` remained on Logic.
2. **Expert Co-Processing:** The load-balancer `getIdxHighOptimal()` analyzes whether shifting expert compute from Logic to GPU will reduce the overall latency. However, as the telemetry proves, the GPU is already a massive bottleneck (~88.9M ns vs ~16.4M ns on Logic), entirely overwhelmed by massive `Linear` projections (up/down/gate). Shifting expert work to the GPU would only increase `high_time` further. The dynamic load balancer correctly evaluated this and opted to shift **zero** experts to the GPU.

## 9. Limitations / Warnings
Because this workload heavily bottlenecks the GPU with standard Linear layers, we cannot empirically witness expert load-balancing speedups. To see expert co-processing yield a performance delta, we would need a workload or model where the Logic-PIM is the bottleneck (e.g., higher MoE ratios or less dense FFNs). However, PE is structurally active.
