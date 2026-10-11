# PHASE 2 — GROUP 0: CREATE FILE CHANGES AND EXPLANATION DOCUMENT

## 1. PROJECT BASELINE

**Frozen Validated Baseline:**
- **Config**: `reproduction/configs/gpu_baseline.yaml`

**Existing Results (Protected):**
- `reproduction/results/csv/gpu_baseline/`
- `reproduction/results/raw/gpu_baseline/`
- `reproduction/results/gpu_baseline_run.log`

**Effective Baseline Parameters:**
- **Model**: Mixtral
- **Hardware**: H100
- **Scale**: 1 node, 4 devices
- **Parallelism**: TP4 / DP1 (Expert Tensor parallel degree `e_tp_dg` typically tied to baseline config; baseline assumes ET=1 unless configured otherwise)
- **Lengths**: Input length 2048, output length 128
- **Precision**: 2 bytes (FP16/BF16)
- **Batching**: Max batch 32
- **Process Token**: Max process token 524288
- **Simulation**: Iterations 5
- **Workload**: Skew 0, decode mode
- **Architecture**: Parallel execution OFF, Duplex Logic-PIM OFF (Baseline config specifies standard GPU).

**THE BASELINE CONFIGURATION MUST NOT BE MODIFIED.**

---

## 2. SOURCE FILES THAT WILL BE MODIFIED

| File | Class / Function | Planned Modification | Why |
|------|------------------|----------------------|-----|
| `src/scheduler/sequence.cpp` | `BatchedSequence::update_expert()` | Add observer-only CSV export logic | To safely capture the exact `request_id`, `token_id`, `layer_id`, `topk_rank`, and `expert_id` data flow before tokens are aggregated into flat expert chunks. |
| `src/module/timeboard.cpp` | `TimeStamp::writeProcessorTrace()` | Augment output schema for processor trace | To serialize `device_id`, `layer_id`, `operation`, `processor` (from `status.processor_type`), `start_time`, `end_time`, and `duration` exactly as required. |
| `src/utils/trace_writer.h` (Optional) | `TraceWriter` | Introduce minimal static file handler | If required, a minimal utility cleanly handles singleton file opening, truncating new runs, header generation, and thread-safe flushing to prevent trace corruption across devices. |

---

## 3. EXPERT ROUTING TRACE

**Verified Data Path:**
`precomputed/synthetic expert assignment` → `BatchedSequence::update_expert()` → `expert aggregation` → `Route::forward()` → `ExpertFFN::forward()`

**Explicit Statement:**
`Route::forward()` does **NOT** dynamically calculate top-k from logits. 

The current workload uses either:
- Precomputed expert traces through `Scheduler::initExpertList()` OR
- Synthetic expert assignments through `Scheduler::getRandomExpert()`

The token-level expert information is exclusively available in `src/scheduler/sequence.cpp` inside `BatchedSequence::update_expert()`.

**Exact Capture Fields:**
- `request_id`
- `token_id`
- `cur_layer`
- `expert_id`
- `topk_rank`

*Note on `topk_rank`:* This rank is derived strictly from the existing inner top-k loop/index (`k`). It is NOT a newly calculated ranking, but rather a direct projection of the trace's indexed `top_1`, `top_2` selections.

---

## 4. DECODE STEP DOCUMENTATION

The simulator **does not** maintain a separate `decode_step` variable at the routing capture point. 

The available existing token progression index is: `token_id`

**Explanation:**
The `token_id` loop starts from the sequence's current generated length (`current_len`) and iterates sequentially through the tokens being processed. Because the decoding phase generates one token per sequence iteration, the token's position index natively behaves as the sequence decode step. 

If the implementation uses `token_id` as the trace's `decode_step` field, this represents an **EXISTING** simulator index used safely as the decode-position proxy. The simulator does not possess an independent `decode_step` variable.

---

## 5. PROCESSOR TRACE

**Verified Timing Path:**
`ModuleGraph::check_ready()` → `Executor::execution()` → `processor-specific execution` → `timing` → `TimeStamp` → `writeProcessorTrace()`

**Trace Fields:**
- `device_id`
- `layer_id`
- `operation`
- `processor`
- `start_time`
- `end_time`
- `duration`

**Explicit Statement:**
`src/module/timeboard.cpp`, specifically `TimeStamp::writeProcessorTrace()`, has complete access to all required fields. The processor type must be dynamically read from the existing `status.processor_type` field (translated to "GPU", "Logic", or "PIM"). Do **NOT** hard-code "GPU".

---

## 6. TRACE OUTPUT FILES

**File 1: Expert Routing Trace**
`reproduction/results/traces/gpu_baseline/expert_routing.csv`
- `request_id`: The global sequence/request ID being processed.
- `decode_step`: The token's index progression in the sequence, derived directly from the existing `token_id` loop iterator.
- `layer_id`: The current MoE layer encountering the token.
- `token_id`: Identical to `decode_step` (the sequential token index).
- `topk_rank`: The explicit trace-array rank (e.g., 1 for primary expert, 2 for secondary).
- `expert_id`: The destination expert ID selected by the synthetic trace.

**File 2: Processor Trace**
`reproduction/results/traces/gpu_baseline/processor_trace.csv`
- `device_id`: The physical device executing the operation.
- `layer_id`: The model layer depth index, parsed from the operation timestamp name.
- `operation`: The specific block function (e.g., `expertFFN`, `attention`).
- `processor`: The processing element running the operation ("GPU", "Logic", "PIM").
- `start_time`: Simulation nanoseconds when execution begins.
- `end_time`: Simulation nanoseconds when execution ends.
- `duration`: Computed elapsed delta (`end_time - start_time`).

---

## 7. OBSERVER-ONLY DESIGN

Instrumentation will **ONLY** observe existing values. 

It must **NOT** modify:
- Expert selection
- Routing probabilities
- Top-k selection
- Expert ordering
- Token ordering
- Scheduler decisions
- Processor assignment
- Execution ordering
- Timing calculations
- Existing statistics

The tracing layer must strictly be non-blocking and avoid introducing artificial delays into the simulation timeline.

---

## 8. TRACE WRITER DESIGN

If a minimal tracing utility is introduced (e.g., `trace_writer.h`), it must observe:
- **Filename/Responsibility**: A lightweight wrapper for file output.
- **Interface**: A simple `write()` method accepting strings or basic types.
- **Initialization**: Creates directories (`reproduction/results/traces/gpu_baseline/`) silently.
- **CSV Header Handling**: Emits headers only on file creation.
- **File Behavior**: Must truncate previous files strictly on new simulation runs to avoid appending to stale data.
- **Flushing**: Safely flushes state without creating multithreading blockages.
- **Deterministic**: Preserves exactly what the simulation produces without arbitrary delays.

---

## 9. BASELINE PROTECTION

The following files and directories are strictly **PROTECTED** and must not be overwritten or modified:
- `reproduction/configs/gpu_baseline.yaml`
- `reproduction/results/csv/gpu_baseline/`
- `reproduction/results/raw/gpu_baseline/`
- `reproduction/results/gpu_baseline_run.log`

---

## 10. DEBUG VALIDATION PLAN

**Sequence of Execution:**
1. Create a small debug workload derived from the validated baseline config.
2. Run an uninstrumented debug simulation (capture logs).
3. Run an instrumented debug simulation.
4. Compare simulator behavior identically.
5. Validate the expert-routing trace.
6. Validate the processor trace.
7. **Stop immediately if baseline simulation behavior changes.**

**Routing Validation Criteria:** Valid expert IDs, expected top-k counts per token, valid top-k ranks, valid layer bounds, sensible token tracking, consistent request IDs.
**Processor Validation Criteria:** Valid device counts, valid processor types ("GPU"/"Logic"/"PIM"), `start_time < end_time`, `duration = end_time - start_time`, logically advancing timestamps.

---

## 11. FULL BASELINE PLAN

After debug validation completely succeeds:
1. Run the identical validated baseline configuration with the new instrumentation.
2. DO NOT modify `reproduction/configs/gpu_baseline.yaml`.
3. Generate the required artifacts:
   - `reproduction/results/traces/gpu_baseline/expert_routing.csv`
   - `reproduction/results/traces/gpu_baseline/processor_trace.csv`
4. Guarantee that all original baseline results are preserved untouched.

---

## 12. OFF-CHIP DRAM

**OFF-CHIP DRAM IS NOT BEING IMPLEMENTED IN THIS GROUP.**

The current verified memory path routes `Device` → `DRAMInterface` → `Ramulator` → `current HBM configuration`. The current simulator fundamentally lacks the ability to preserve semantic expert identity inside a `DRAMRequest`.

The future off-chip DRAM phase will absolutely require an explicit expert-residency mapping design. It is not implemented in this documentation or phase.

---

## 13. REPRODUCTION_A100 DIRECTORY

`reproduction_A100/with_off_chip_dram/` is exclusively an isolated audit and planning workspace.

It is **NOT** the canonical reproduction directory. It will not be moved, merged, renamed, or deleted.

---

## 14. PHASE ROADMAP

- **Phase 1**: Build + baseline/source audit (Complete)
- **Phase 2**: Instrumentation (Current target)
- **Phase 3**: Expert access/locality analysis
- **Phase 4**: Training-free expert predictor
- **Phase 5**: Off-chip DRAM + expert residency
- **Phase 6**: Expert prefetcher
- **Phase 7**: Baseline vs prefetcher evaluation
- **Phase 8**: Duplex / PE / ET experiments

*(Later phases will not be implemented now).*

---

## 15. VERIFIED VS UNRESOLVED

### VERIFIED
- Routing does not dynamically use softmax probabilities or model inference logs. It directly processes fixed traces or random subsets.
- The only loop capable of granular top-k token assignment interception is `BatchedSequence::update_expert()`.
- Simulation timings are explicitly governed by `status.device_time` modifications inside `TopModuleGraph::set_pop_status()`.
- `processor_trace.csv` fields are natively retrievable inside `Timeboard::writeProcessorTrace()`.

### UNRESOLVED
- **Exact off-chip DRAM expert residency mechanism**: How to safely stall `ExpertFFN::forward()` if a required expert trace is absent from HBM.
- **Expert-to-memory-address mapping**: Creating a physical abstraction for prefetching that maps `e_id` to logical memory addresses, which currently does not natively exist inside `MMapController`.
- **Prefetch interaction**: Determining how prefetch `DRAMRequest`s interact asynchronously with the existing synchronous blocking memory model.
- **Duplex/PE/ET semantics**: How off-chip DRAM fetching latencies will behave under concurrent PIM/PE execution pipelines.
