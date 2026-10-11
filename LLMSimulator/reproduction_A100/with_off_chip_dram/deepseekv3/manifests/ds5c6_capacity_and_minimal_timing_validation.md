# DS-5C-6: Minimal Expert-Load Timing and DDR5 Capacity Verification

## 1. Objective and Context

The objective of DS-5C-6 was to complete the end-to-end verification of the off-chip DDR5 system integration, fix any latent defects in the `Device` LRU execution cache, and ensure that a minimal workload correctly translates logical addresses into Ramulator DDR5 timing events.

This manifest documents the changes made to correct the execution cache logic and the successful demonstration of the hardware execution path.

## 2. LRU Execution Cache Correction

Prior inspection (DS-5C-5) identified that `Device::mark_expert_resident` erroneously used the *incoming* expert size for eviction accounting rather than the *evicted* expert sizes. While mathematically safe for DeepSeek-V3 (where all experts are homogeneously 88,080,384 bytes), this was architecturally incorrect. Additionally, duplicate key insertions could inflate the list size.

### Fixes Applied to `src/hardware/device.cpp` & `device.h`:
- Changed `lru_expert_list` from `std::list<std::pair<int, int>>` to `std::list<std::tuple<int, int, long long>>` to retain the exact byte size of each cached expert.
- Updated the eviction `while` loop in `mark_expert_resident` to fetch the exact evicted size using `std::get<2>(lru_expert_list.front())`.
- Added a `std::find_if` guard to verify if an expert is already resident before tracking it, preventing duplicate key insertions.

### Validation via `test_lru.cpp`
A rigorous deterministic test (`test_lru`) was written to validate:
- Equal-size insertions, cache hits, and least-recently-used evictions.
- Variable-size insertions (which trigger multiple heterogeneous evictions).
- Duplicate insertions (safely bypassed without corrupting tracking).
- DeepSeek-V3 specific limits (908 complete 88MB experts correctly fit within the 80GB HBM capacity; the 909th triggers exactly one eviction).

All tests successfully passed.

## 3. Minimal Timing and Trace Verification

A new test (`eval/test_minimal_timing.cpp`) was created to isolate the production request pipeline. It bypasses the abstract Python/C++ operation graph and directly constructs three tensors representing one partitioned 88MB DeepSeek-V3 expert (W1, W2, W3).

### Memory Translation Consistency
The test mapped the tensors starting at `expert_start_address = 17 * 88,080,384 = 1,497,366,528`.
We traced this address through `MMapController::addrToVec()`. Based on the `ddr5_32gb_x16_2ch` configuration, the address is factored across:
- `granul` (64)
- `num_cube` (1)
- `num_channel` (2)
- `num_rank` (2)
- `num_bankgroup` (4)
- `num_bank` (4)
- `num_col` (64)

The cumulative divisor before computing the row index is `262,144`.
`1,497,366,528 / 262,144 = 5,712`.

### Trace Output Verification
Ramulator 2 was configured with the `TraceRecorder` plugin to capture DDR5 commands. The trace output (`log/cmd_ddr5.log.ch0`) confirmed that the theoretical address was perfectly executed as physical DDR5 memory commands:

```
1, ACT, 0, 0, 0, 0, 5712, 0, -1, Read, DRAM, Read
...
8, RD, 0, 0, 0, 0, 5712, 0, -1, Read, DRAM, Read
```

The `row` index in the trace is exactly `5712`, verifying that `MMapController` translates the extended capacity linearly and correctly interfaces with Ramulator's DDR5 model.

### Execution Cache Bypass
The test confirmed that `OFFCHIP_DRAM` correctly bypasses the `ExecutionCache`. Two consecutive, identical `issueRamulatorBatch` requests targeting `OFFCHIP_DRAM` triggered two full Ramulator simulation spans, confirming that off-chip fetches are never artificially suppressed by HBM timing caches.

## 4. Conclusion

- **Capacity**: The `num_row = 10,485,760` expansion correctly maps up to ~2.75 TB of off-chip address space without geometry conflicts or truncation.
- **Timing**: Ramulator successfully receives and schedules `ACT`/`RD` commands for off-chip tensor data.
- **Caching**: The LRU HBM execution cache correctly tracks expert sizes and ignores `OFFCHIP_DRAM` memory operations.

The LLMSimulator repository is now validated and structurally ready for full-scale DeepSeek-V3 workload evaluation.
