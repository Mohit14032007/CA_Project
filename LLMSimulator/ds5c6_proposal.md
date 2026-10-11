# Proposed Changes for DS-5C-6

## 1. DDR5 Capacity Resolution
**Files affected:** `eval/test_ddr5_mapping.cpp` (and the final manifest)
**Issue:** The previous report had an arithmetic error calculating `10485760 * 64 * 64 * 4 * 4 * 2 * 2`. The correct product is `2,748,779,069,440` bytes (~2.749 TB), not 1.374 TB.
**Action:** Update the mapping test to verify the highest representable byte address (`2,748,779,069,439`) and verify it does not wrap.

## 2. LRU Defects Fix
**Files affected:** `src/hardware/device.h`, `src/hardware/device.cpp`, `eval/test_lru.cpp`
**Bug 1:** `mark_expert_resident` subtracts the incoming `expert_size` instead of the evicted entry's size, breaking capacity constraints for oversized entries.
**Bug 2:** Duplicate insertions append stale list entries.
**Proposed Fix:**
- Change `lru_expert_list` to `std::list<std::tuple<int, int, long long>>` to track the exact size of each resident expert.
- Change `resident_experts` to `std::unordered_map<std::pair<int, int>, std::list<std::tuple<int, int, long long>>::iterator, PairHash>`.
- In `mark_expert_resident`, check `if (resident_experts.find(key) != resident_experts.end()) return;` to prevent duplicates.
- In the eviction loop, deduct `std::get<2>(lru_expert_tuple)` (the evicted size) from `resident_bytes`.
**Testing:** Re-run `test_lru.cpp` to prove that oversized entries and duplicates now correctly maintain invariants without negative `resident_bytes` or stale map references.

## 3. Minimal Timing Test
**Files affected:** `eval/test_minimal_timing.cpp`, `eval/CMakeLists.txt`
**Action:** Build a targeted executable that manually invokes `issueRamulatorBatch` with `MemoryTarget::OFFCHIP_DRAM` for a simulated 88MB expert load (W1/W2/W3). Verify bypass of execution cache, capture `per_request_duration`, and inspect the generated Ramulator command trace.
