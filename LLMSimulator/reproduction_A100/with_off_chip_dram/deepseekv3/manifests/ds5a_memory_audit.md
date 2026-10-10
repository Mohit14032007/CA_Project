# DS-5A: Forensic Audit of HBM Cache + Large Off-Chip DDR5 Memory Model

## Objective
Audit the entire memory path from expert request, HBM LRU cache, `OFFCHIP_DRAM` target decision, `MMapController` address translation, to Ramulator DRAM timing in the DeepSeek-V3 experiment. Determine if the reported 7.97 ms latency for loading 88 MB of expert weights is physically sound or the result of a misconfiguration.

## Audit Findings

### 1. Source-Level Execution Path
The memory fetch for an out-of-cache expert follows a highly serialized path that artificially inflates latency by preventing pipelining:
- `Expert::forward` generates three separate `Linear` operations (`gate_proj` W1, `up_proj` W3, `down_proj` W2).
- The `Linear::forward` check sets `weight_target = OFFCHIP_DRAM` because the expert is not resident.
- `LinearExecutionGPU` calls `issueRamulator()` sequentially for `weight` (W1), and then loops over `sister_weights` (W2, W3).
- Inside `issueRamulator()`, the `OFFCHIP_DRAM` target is routed to `device->offchip_mmap_controller` to bypass the 80 GB HBM limit.
- `device->run_ramulator` invokes `DRAMInterface::HandleRequest()`, which pushes 64-byte chunks of the tensor to Ramulator and **calls `run()`, blocking until the entire 29.36 MB tensor is completely fetched from Ramulator.**
- Consequently, W1, W2, and W3 are fetched in strict serialization. The total 7.97 ms latency is simply the accumulated sum of three blocking ~2.65 ms fetches. 

### 2. Physical vs. Logical Capacity and Aliasing
The `LLMSimulator` does not instantiate memory payload. Memory capacity exists purely as logical address counters and Ramulator bounds.
- **Logical Space:** DeepSeek-V3 requires ~1.285 TB for 14592 experts. `MMapController` treats this as a contiguous address space up to `num_row = 10,485,760` per bank.
- **Physical Space:** Ramulator's `dram_config_DDR5.yaml` configuration defines exactly 32 GB of DDR5 (2 channels, 2 ranks, 4 bankgroups, 4 banks, 131072 rows, 1024 cols). 
- **Aliasing:** Ramulator's `RoBaRaCoCh` linear address mapper extracts the lower address bits based on its physical geometry and completely discards any high bits that exceed 32 GB. Therefore, the 1.285 TB logical space aliases onto the 32 GB physical DDR5 space with a ~40:1 ratio.
- **Conclusion on Storage:** Because the simulator only simulates timing and does not store actual tensor bytes on the host, this 32 GB aliasing is functionally safe and does not corrupt the model execution.

### 3. Address Mismatch (The 7.97 ms Root Cause)
The 7.97 ms latency translates to an abysmal memory bandwidth of ~11 GB/s (vs the theoretical 51.2 GB/s of 2-channel DDR5-3200 x16). The forensic audit uncovered a critical address mapping mismatch between `MMapController` and Ramulator DDR5 that destroys the row-buffer hit rate:
- `MMapController::getAddrVec()` hardcodes a 7-element vector tailored for **HBM3**: `{channel / 2, channel % 2, rank, bankgroup, bank, row, col}`.
- DDR5 in Ramulator2 defines only 6 levels: `{"channel", "rank", "bankgroup", "bank", "row", "column"}`.
- When the 7-element vector is passed to DDR5, all levels are shifted by 1 index:
  1. **Channel:** Receives `channel / 2`. Since `num_channel = 2`, this is always `0`. All traffic hits Channel 0; Channel 1 is completely idle.
  2. **Rank:** Receives `channel % 2` (0 or 1).
  3. **BankGroup:** Receives logical `rank`.
  4. **Bank:** Receives logical `bankgroup`.
  5. **Row:** Receives logical `bank` (cycles 0 to 3).
  6. **Column:** Receives logical `row` (cycles up to 10,485,760).
  7. *(Ignored)*: logical `col` is dropped.
- **The Result:** Ramulator thinks the row address is the logical bank address, which cycles `0, 1, 2, 3` repeatedly for adjacent memory fetches. Consequently, every single memory access to a bank is to a different "Row", resulting in a **100% row-buffer miss rate**. The constant Precharge + Activate penalty completely bottlenecks the DDR5 channel, yielding the artificial ~11 GB/s bandwidth.

## Summary & Next Steps
The 7.97 ms off-chip load latency is **physically incorrect**. It is artificially high due to a hardcoded HBM3 7-level address mapping being applied to a 6-level DDR5 Ramulator configuration, which creates 100% row-buffer misses and idles half the memory channels. 

**Phase DS-5B Requirements:**
1. Fix `MMapController::getAddrVec()` to dynamically output a 6-element vector for DDR5 configurations.
2. Optimize `LinearExecutionGPU` to issue W1, W2, and W3 fetches asynchronously to Ramulator, avoiding strictly serialized blocking overhead.
