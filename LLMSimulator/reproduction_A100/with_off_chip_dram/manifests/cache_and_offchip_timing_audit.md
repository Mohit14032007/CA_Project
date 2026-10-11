# Investigation Report: Cache and Off-Chip Timing Audit
                                   
## 1. Executive Conclusion
Our audit confirms two critical characteristics of the current Phase 5-8 memory implementation:
1. **Residency is purely logical**: There is no 80 GB capacity check, no eviction policy, and no physical memory freeing. The `resident_experts` structure acts as an infinite-capacity "cold-start" tracker.
2. **Timing is identical due to the Simulator Execution Cache**: The reason all 248 misses take exactly 981,447 ns is that the simulator utilizes an internal `execution_time_cache` that keys memory latency solely by **Request Size** and **Memory Target**. Since all expert tensors (W1, W3, W2) are exactly identical in size (117.44 MB), Ramulator DDR5 simulation is only invoked on the *very first miss*. All subsequent misses hit the simulator's execution cache, completely bypassing Ramulator and returning the exact same duration.

## 2. Is there actually a finite HBM cache?
**No.** The current `resident_experts` mechanism is logical residency, not a physical finite HBM cache.
There is no HBM capacity accounting, no per-expert allocation, no physical expert addresses, and no eviction (LRU/FIFO) policy. Once marked resident, an expert remains in HBM forever.

## 3. Where is residency implemented?
- **State Storage**: `src/hardware/device.h` (Line 113) as `std::set<std::pair<int, int>> resident_experts;`
- **Checking/Marking**: `src/module/linear.cpp` (Lines 46-58) inside `Linear::forward()`.

## 4. Exact Cache/Residency State Machine
- **Data Structure**: `std::set<std::pair<int, int>>`
- **Key**: `(layer_id, expert_id)` (Logical identifier only, not address/tensor).
- **HIT**: If `resident_experts.find({layer_id, expert_id})` succeeds, `weight_target` remains `MemoryTarget::HBM`.
- **MISS**: If not found, `weight_target` is set to `MemoryTarget::OFFCHIP_DRAM`, `load_sisters` is set to `true`, and `device->mark_expert_resident()` immediately inserts it into the set.

## 5. Complete Memory Request Path
When an expert miss occurs, the exact source-code path is:
1. `src/module/linear.cpp`: `Linear::forward()` determines the miss, sets `weight_target = OFFCHIP_DRAM`.
2. `src/hardware/linear_impl.cpp`: `LinearExecutionGPU()` is invoked.
3. `src/hardware/layer_impl.cpp`: `issueRamulator()` checks `checkExecutionCache(..., size, OFFCHIP_DRAM)`.
   - If missing from cache, wraps tensor in `DRAMRequest`.
4. `src/hardware/device.cpp`: `Device::run_ramulator()` routes it to `offchip_dram_interface`.
5. `src/dram/dram_interface.cpp`: `DRAMInterface::HandleRequest()` passes it to the `MMapController`.
6. `src/dram/mmap_controller.cpp`: Translates the payload size into sequential Ramulator requests starting at address 0.
7. `Ramulator`: Simulates DDR5 timing.
8. `DRAMInterface::updateStatus()` retrieves `duration`, multiplies by `memory_scale_factor`.
9. `issueRamulator()` caches this final scaled latency into `cluster->execution_time_cache` so it never has to run Ramulator for this size/target again.

## 6. Exact Origin of 981447 ns
This number is **not fixed**, but it is **deterministic and scaled**.
1. Ramulator calculates the exact clock cycles required to stream 117.44 MB of data sequentially across a DDR5-3200 system (2 channels, 2 ranks).
2. In `src/dram/dram_interface.cpp` (Lines 53-54), the Ramulator duration is scaled:
   `time += (duration * memory_scale_factor);`
3. Because `Device::Device()` initialized `offchip_dram_interface` with the same `memory_scale_factor` as H100/A100 (which is **0.76923**), the raw Ramulator DDR5 cycles are multiplied by 0.76923 to produce the final simulator nanoseconds. The sum for the three 117.44 MB weight matrices evaluates to exactly 981,447 ns.

## 7. Why all misses have identical duration
In `src/hardware/layer_impl.cpp`, `issueRamulator()` constructs a `CacheKey`:
```cpp
CacheKey key = std::make_tuple(layer_type, processor_type, dram_request_type, tensor->getSize(), target);
```
**Notice that physical address is not part of the cache key.**
Because every W1, W3, and W2 matrix is exactly 117,440,512 bytes, the first miss caches the Ramulator execution time for `(LINEAR, GPU, Read, 117440512, OFFCHIP_DRAM)`. 
For every single one of the remaining 247 expert misses, the simulator execution cache intercepts the call and immediately returns the exact same 981,447 ns total without ever asking Ramulator.

## 8. Expected or Suspicious?
This is **Expected** for the simulator's architecture. The simulator uses this execution cache to achieve massive speedups (avoiding millions of cycle-accurate DRAM simulations for identical matrix multiplication sizes). While it loses per-expert physical address bank contention accuracy, it provides a perfectly stable baseline latency for streaming a large contiguous block of data.

## 9. Role of Ramulator
Ramulator is invoked **exactly once** for the W1 matrix size targeting DDR5. It provides the baseline cycle latency for that payload size.

## 10. Role of DDR5 Configuration
Parsed from `configs/dram_config_DDR5.yaml`, it defines `DDR5_3200BN`, 2 channels, 2 ranks. This physical configuration dictates the cycle count Ramulator returns for the initial cache-miss simulation.

## 11. Role of memory_scale_factor
`memory_scale_factor = 0.76923` acts as a direct multiplier on the raw Ramulator cycles (`src/dram/dram_interface.cpp`). It is currently inherited from the A100/H100 setup in `Device::Device()`.

## 12. W1/W2/W3 Aggregation Explanation
Are they one request or three? **They are THREE separate DRAM requests, but aggregated in the CSV telemetry.**
In `LinearExecutionGPU()` (`src/hardware/linear_impl.cpp`, lines 48-63):
1. `issueRamulator()` is called for `weight` (W1).
2. Because `load_sisters` is true, a loop immediately calls `issueRamulator()` for `sister` (W2) and then `sister` (W3).
3. The `memory_duration` of all three are summed into `weight_memory_cost`.
4. The `getSize()` of all three are summed into `total_size` (352,321,536 bytes).
5. A single row is written to `expert_load.csv` representing the total transaction.

## 13. What the current experiment DOES model
It successfully models the **Cold-Start Penalty**: the exact memory latency incurred when streaming a ~352 MB expert payload from DDR5 to HBM over an interconnect on its very first invocation. 

## 14. What it DOES NOT model
It does not model HBM capacity constraints, eviction logic (LRU), PCI-e/interconnect serialization limitations, or physical memory fragmentation (bank conflicts between different experts).

## 15. Recommended Next Step
Do not modify the `execution_time_cache`, as it is required for simulator performance. However, for Phase 10+, to accurately test eviction and capacity, we must replace the `std::set resident_experts` with an LRU queue that strictly caps resident capacity to 80 GB.

## Summary Table

| Question | Finding |
|---|---|
| Is there a finite HBM cache? | **NO** |
| Is residency logical or physical? | Logical (`std::set<layer_id, expert_id>`) |
| Where is residency stored? | `Device` class (`src/hardware/device.h`) |
| What causes a miss? | `is_expert_resident()` returns false |
| What causes a hit? | `is_expert_resident()` returns true |
| Is DDR5 actually invoked on miss? | Yes, via `MemoryTarget::OFFCHIP_DRAM` |
| Is Ramulator actually invoked? | Yes, but **ONLY ONCE** for the first expert |
| Why is duration identical? | Simulator `execution_time_cache` keys by **size** only |
| What is 981447 ns? | 3 sequential requests of 117MB in DDR5-3200, scaled by 0.76923 |
| Is 352321536 B one or three requests? | Three separate `issueRamulator` requests aggregated in CSV |
| Does memory_scale_factor matter? | Yes, scales raw cycles by 0.76923 |
| Is this behavior expected? | Yes, standard simulator optimization behavior |
