# Device Trace Validation Report

## 1. Previous trace problem (Investigation)
The user suspected that `raw/device_*` logs and `processor_trace.csv` were blindly duplicating the global execution tree across all devices because operations like `AttentionGen` appeared identically for devices 0, 1, 2, and 3.

## 2. Actual source location of device identity
Investigation into `src/hardware/cluster.cpp` and `src/module/expert.cpp` reveals that **the simulator already constructs fully independent, device-specific computational graphs**.
- During model creation (`Model::model_distribute`), `LLM::Create` builds a distinct neural network graph for each device.
- `src/hardware/device.cpp`: Each device maintains its own `top_module_graph` containing its assigned operations.
- `Cluster::exportGantt` correctly loops over devices and exports each device's specific `top_module_graph`.

## 3. New instrumentation location
**No new instrumentation is required.** The current implementation of `processor_trace.csv` (from Phase B.5) is already perfectly device-aware. The perceived "duplication" is the mathematical reality of Tensor Parallelism, not a logging bug.

## 4. CSV schema
The existing schema is retained as it correctly models the hardware topology:
`device_id,layer_id,operation,processor,start_time,end_time,duration`

## 5. Validation method
- Compared MD5 checksums of raw device logs across all configurations.
- Searched `processor_trace.csv` for Non-Expert layers (TP=4).
- Searched `processor_trace.csv` for Expert layers (ET=1 vs ET=4).

## 6. GPU test result
In `gpu_baseline` (TP=4, ET=1), the raw device files are explicitly **different**:
- `device_0`: 531c8810...
- `device_1`: 7d0536af...
- `device_2`: ded50fb6...
- `device_3`: b893304b...
(The user likely only inspected the first 500 lines, which contain the TP=4 Attention layers that are identically dispatched across all devices).

## 7. Duplex test result
In `duplex` (TP=4, ET=1), `AttentionGen` appears for `device_id` 0, 1, 2, and 3 because `ne_tp_dg = 4` shards attention across all devices. However, expert execution is correctly isolated.

## 8. ET test result
In `duplex_pe_et` (ET=4), the raw device logs *are* perfectly identical. This is precisely correct because `e_tp_dg = 4` shards all 8 experts across all 4 devices, meaning every device has the exact same topological workload.

## 9. Example expert → device mapping
Calculated from `src/module/expert.cpp` for `duplex` (ET=1, num_device=4):
- `num_expert_per_device = 8 / 4 = 2`
- `device_2` handles `expert_offset = 2 * (2 / 1) = 4` (Experts 4 and 5)

Grep verification on `reproduction/results/traces/duplex/processor_trace.csv` confirms:
`2,0,expert_FFN_4,GPU...`
**Zero** records exist for `0,0,expert_FFN_4` or `1,0,expert_FFN_4`. 
The trace flawlessly maps Expert 4 exclusively to Device 2.

## 10. Confirmation that timing was unchanged
No code changes were made to the simulator. Execution semantics, routing, and timing remain perfectly identical.

## 11. Limitations
Because the simulator is a bulk-synchronous parallel (BSP) model, TP layers (like Attention) will show exact identical start/end times across all participating devices. This represents ideal network synchronization rather than a logging artifact.
