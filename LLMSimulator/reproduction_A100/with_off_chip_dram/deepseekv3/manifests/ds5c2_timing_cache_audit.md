# Phase DS-5C-2: Final Off-Chip Timing-Cache Audit

## 1. Goal
Verify that the `ExecutionCache` and the expert residency tracking (LRU) correctly handle off-chip DDR5 W1/W2/W3 batch loads, and that equal-sized expert tensors do not erroneously reuse stale timing from a previous miss.

## 2. Call-Site Trace & Cache Bypass Logic
- The core of the execution cache is implemented in `Device::checkExecutionCache()` and `Device::addExecutionCache()`.
- The key is: `std::tuple<LayerType, ProcessorType, DRAMRequestType, long (size), MemoryTarget>`.
- In `src/hardware/layer_impl.cpp`, both `issueRamulator` and `issueRamulatorBatch` contain explicit bypass logic:
  ```cpp
  bool bypass_cache = (target == MemoryTarget::OFFCHIP_DRAM);
  if (bypass_cache || !device->checkExecutionCache(exec_status, key)) { ... }
  ```
- Because of this, **any request targeting `OFFCHIP_DRAM` strictly bypasses the execution timing cache**. The Ramulator simulator is invoked unconditionally, ensuring that distinct experts—even if identical in size—receive fresh, state-dependent memory timings (e.g., modeling row conflicts).

## 3. Expert Residency & Batch Submission
- In `src/module/linear.cpp`, a weight is marked for `OFFCHIP_DRAM` and its sisters are marked for loading (`load_sisters = true`) if `!device->is_expert_resident(...)`.
- The expert is immediately marked as resident via `device->mark_expert_resident(...)` before the simulation transitions to `LinearExecutionGPU`.
- Because LLMSimulator's execution graph evaluates strictly sequentially, `LinearExecutionGPU` entirely blocks while `issueRamulatorBatch` spins Ramulator for W1, W2, and W3. 
- By the time the graph evaluation reaches W2 and W3, the expert is already marked resident. Their `weight_target` resolves to `HBM`, which correctly skips DDR5 logic and hits the execution cache for rapid HBM read timings. 

## 4. Diagnostic Evidence
A deterministic C++ diagnostic (`scratch/ds5c2_timing_audit_test.cpp`) was executed against `issueRamulatorBatch`:
- **Expert 1 (MISS):** Submitted W1, W2, W3 to Ramulator as `OFFCHIP_DRAM`. Output memory duration: `3900340`.
- **Expert 2 (MISS):** Submitted W1, W2, W3 to Ramulator as `OFFCHIP_DRAM`. Output memory duration: `3901020`.
- **Result:** The timings correctly differed, proving the cache was bypassed and Ramulator generated fresh traces for the second expert. 
- **Residency API:** Verified that `mark_expert_resident` successfully toggles the boolean state of `is_expert_resident`.

## 5. Timing Accumulation
- The batch request correctly tracks per-request timestamps from Ramulator.
- `DRAMInterface::HandleRequest` calculates the aggregate duration using `max_end - min_start`, which accurately represents concurrent memory latency rather than blindly summing individual request latencies.
- `LinearExecutionGPU` safely assigns the aggregated `memory_cost` to the GPU's `time_ns`, ensuring the compute blocks on the complete tensor transfer.

## 6. Conclusion
**No code changes are required.** The execution-timing cache explicitly guards against suppressing off-chip workloads, the batching logic cleanly accumulates concurrent latency, and the LRU effectively blocks duplicate loads without data races. The DDR5 simulation pipeline is robust and ready for a full-scale workload.

**Smallest Next Action:** 
Exit the diagnostic phase and commence the full DeepSeek-V3 simulation.
