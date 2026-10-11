# DS-5C-5: Off-Chip DDR5 and LRU Source Audit

## Phase A — DDR5 Geometry Audit

**Files Inspected:**
- `src/dram/memory_config.h` (config definition `ddr5_32gb_x16_2ch`)
- `src/dram/mmap_controller.cpp` (`MMapController::getAddrVecLOGIC`, `addrToVecLOGIC`)
- `src/dram/ramulator2/src/frontend/impl/PIM_controller.cpp`
- `src/dram/ramulator2/src/addr_mapper/impl/linear_mappers.cpp`

**Answers:**
1. **Does the custom row count change Ramulator's instantiated DRAM organization or only LLMSimulator's address mapping?**
   It only changes LLMSimulator's mathematical mapping. Ramulator 2 only instantiates `Node`s down to the physical `Bank` level. Rows are tracked inside the `Bank` node via a dynamic `std::map<RowId_t, RowState_t> m_row_state`. Because it is a hash map, Ramulator unquestioningly accepts row ID `10,485,760` without bloating memory or changing hardware organization arrays.
2. **Are incoming row indices validated, truncated, wrapped, or accepted outside the preset's nominal organization?**
   They are accepted as-is. `PIM_controller.cpp` initializes a `Ramulator::Request` with the pre-calculated `addr_vec`. Ramulator's DDR5 implementation checks bank bounds but blindly accepts the row integer as a map key.
3. **Are high-order address bits preserved?**
   Yes. `addrToVecLOGIC` cascades modulo divisions upwards (col -> bank_under -> bg -> channel -> cube). The remainder lands in `row_idx = address % memory_config.num_row`. Because `num_row = 10485760`, it does not wrap.
4. **Can two distinct expert-weight addresses map to the same effective DRAM coordinates?**
   No. Our deterministic C++ mapping test proved that `4,294,967,296` maps to Row `65536` and `4,383,047,680` (one expert later) maps to Row `66880`. The maximum address (`1.285 TB`) maps safely to Row `9125888`, which is well below the `10485760` ceiling.
5. **Is the capacity formula consistent with the actual mapping implementation?**
   Yes: `10,485,760 rows * 64 col * 64 bytes * 4 banks * 4 bg * 2 rank * 2 ch / 2 (cube math) = 1.37 TB`.
6. **Does the YAML `RoBaRaCoCh` mapper control requests on the active path, or does the custom mapper supply explicit coordinates?**
   The custom `MMapController` supplies explicit coordinates. Ramulator's `RoBaRaCoCh` explicitly skips translation (`if (req.addr == -1)`) when a pre-populated `addr_vec` is supplied.
7. **What physical properties are represented faithfully, and what is merely a custom extended-capacity timing abstraction?**
   Faithful: Channel contention, rank interleaving, bankgroup/bank parallelism, column reads, and actual `ACT`/`PRE`/`RD` timing delays based on row-buffer hits/misses.
   Abstraction: The bank's physical capacity (simulating a chemically impossible die with 10.4 million rows).

## Phase B — LRU Source Audit

**Files Inspected:**
- `src/hardware/device.h` (state definitions, `reset_expert_cache`)
- `src/hardware/device.cpp` (`is_expert_resident`, `mark_expert_resident`)
- `src/module/linear.cpp`

**Test Results (via `eval/test_lru.cpp`):**
1. **Hits promote the correct list entry to MRU:** **Verified.** `lru_expert_list.splice` successfully moves the hit key to the end of the list.
2. **Eviction removes the oldest entry and its map entry:** **Verified.** The front of the list is popped, and `resident_experts.erase(lru_expert)` is called.
3. **`resident_bytes` is decremented by the evicted entry's actual size:** **FAILED (CRITICAL DEFECT).** The code executes `resident_bytes -= expert_size`, where `expert_size` is the function argument of the *incoming* entry! It does not track or subtract the size of the evicted entry.
4. **Inserting an entry larger than the cache cannot exceed capacity:** **FAILED.** Due to the above bug, if an entry larger than the cache is inserted, `resident_bytes - incoming_size` creates negative bytes, breaking the `while (resident_bytes + expert_size > cache_capacity_bytes)` loop permanently and allowing unbounded overflow.
5. **Repeated insertion of an already-resident entry does not duplicate it:** **FAILED.** `mark_expert_resident` pushes a duplicate key to the list and overwrites the map pointer. When the first list element is evicted, the map entry is erased, leaving a dangling list element. (However, simulation flow prevents this because `linear.cpp` checks `is_expert_resident` before marking).
6. **Cache reset clears the list, map, and byte counter consistently:** **Verified.** `Device::reset_expert_cache()` properly clears all three structures.
7. **The `(layer_id, expert_id)` key correctly represents the complete routed expert:** **Verified.**
8. **No operation can treat an expert as resident before its required weight load completes:** **Verified.** Although `mark_expert_resident` is called in software prior to simulating the memory stall, the simulation evaluates layer operations in strict sequence, meaning no other parallel actor will query the cache until the stall duration completes.
9. **The byte counter never becomes negative or exceeds capacity:** **FAILED.** Demonstrated to hit `-20` in the test.
10. **CSV logging does not alter cache behavior:** **Verified.**

## Summary & Safety for Full Simulation

**Is the current setup safe for the DeepSeek-V3 timing experiment?**
**Yes.**

Despite the critical bug in variable-size eviction and duplicate insertion, the DeepSeek-V3 architecture enforces that **all routed experts are strictly identical in size (88,080,384 bytes).**
- Because `evicted_size == incoming_size`, the bug (`resident_bytes -= incoming_size`) perfectly offsets the bytes mathematically.
- The `resident_bytes` will accurately track capacity exactly in multiples of 88 MB.
- Duplicate insertions are bypassed by the model's structural check (`is_expert_resident`).

**Proposed minimal fix (for future architectures only):**
To support models with heterogeneous expert sizes, `lru_expert_list` must be changed to `std::list<std::tuple<int, int, long long>>` to track the exact size of the specific expert being evicted, or an external size-lookup table must be used. No fix is strictly necessary to proceed with DeepSeek-V3.
