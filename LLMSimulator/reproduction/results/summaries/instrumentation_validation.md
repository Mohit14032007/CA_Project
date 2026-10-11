# Phase B.5: Instrumentation Validation Report

## Overview
As part of Phase B.5, the LLMSimulator was successfully instrumented to capture both token-level expert routing decisions and processor-level execution timing without altering the core simulation timing/behavior.

## 1. Trace Formats and Correctness
A debug workload (`debug_instrumentation.yaml`) was executed to validate the structure of the outputs. Both traces are generated at the end of the simulation and stored under `reproduction/results/traces/<config_name>/`.

### A. Expert Routing Trace (`expert_routing.csv`)
Successfully captures routing decisions for each token prior to aggregation in `BatchedSequence::update_expert`.

**Format Output Example:**
```csv
request_id,token_id,layer_id,topk_rank,expert_id
40016,128,0,1,1
40016,128,0,2,5
40017,128,0,1,1
40017,128,0,2,2
```
* **Correctness & Semantics:** 
  * `token_id` is the absolute token position within the request sequence.
  * In this decode-only workload, exactly one token is processed per request per iteration.
  * `request_id` + `token_id` uniquely identifies the token within the sequence.
  * `layer_id` identifies the MoE layer through which that token passes.
  * `topk_rank` identifies rank 1 or 2.
  * `expert_id` identifies the selected expert.
  * *Note: An earlier version of this trace included a `decode_step` column. It was explicitly removed because the codebase populates both `decode_step` and `token_id` from the exact same `token_id` index variable, making them completely identical.*

### B. Processor-Level Execution Trace (`processor_trace.csv`)
Captured directly from the `TimeBoard` logic via `exportGantt` to ensure exact fidelity with the internal representation of time.

**Format Output Example:**
```csv
device_id,layer_id,operation,processor,start_time,end_time,duration
0,0,MoE_decoder_0,GPU,0,226726,226726
0,0,input_layer_norm,GPU,0,21.9952,21.9952
0,0,attention,GPU,21.9952,11326.2,11304.2
0,0,attn_qkv_proj,GPU,21.9952,3789.29,3767.29
0,0,expertFFN,GPU,11370.2,226704,215334
```
* **Correctness:** Captures hierarchical device-layer operations, maps to processor type (GPU vs Logic vs PIM), and exports exact start and end times in simulated ns. `layer_id` is automatically carried down the recursion tree from the top-level Decoder module.

## 2. Non-Interference Check
The implementation guarantees zero interference with the simulation performance model because:
1. **Expert Routing** (`expert_routing.csv`): Only uses a file stream appended to `BatchedSequence::update_expert` in an identical flow. The actual counters `num_token_in_expert` are updated exactly as before. 
2. **Processor Trace** (`processor_trace.csv`): Sits entirely within the `exportGantt` pass over the `TimeStamp` tree (inside `src/module/timeboard.cpp`). This pass only iterates over historical timing nodes created *after* the simulated iteration completes, inherently having zero impact on execution timing calculations.

## Next Steps
Proceeding to Phase C (Plain Duplex configuration) confident that all required telemetry is correctly operational.
