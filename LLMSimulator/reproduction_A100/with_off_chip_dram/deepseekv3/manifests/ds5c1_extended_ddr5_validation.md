# Phase DS-5C-1: Extended-Capacity DDR5 Validation

## 1. Active Configuration
The LLMSimulator leverages the `ddr5_32gb_x16_2ch` preset inside `src/dram/memory_config.h`, which configures a 2-channel, 2-rank DDR5 subsystem. Crucially, the `num_row` parameter is artificially expanded to `10,485,760` rows (an 80x increase over the standard 131,072 rows for 32Gb devices).

## 2. Row-Count Propagation
The inflated `num_row` geometry defined in `MemoryConfig` propagates through the simulator as follows:
- `MMapController::addrToVec`: Consumes `memory_config` to perform spatial interleaving mapping, cleanly dividing logical bytes into channels, ranks, banks, columns, and lastly rows without integer overflow.
- `DRAMInterface::SendRequest`: Converts the `AddrVec` into a `PIMCommand`.
- `PIMController::send`: The Ramulator frontend constructs `Ramulator::Request` objects directly from the `AddrVec`, completely bypassing Ramulator's internal `AddrMapper`. This guarantees that the expanded row index is natively inserted into Ramulator's simulation pipeline.

## 3. Ramulator Hierarchy Dimensions
Ramulator 2 tracks DRAM row-buffer states locally within its `Bank` nodes using integer identifiers representing the open row. Because Ramulator does not statically allocate a memory array proportional to `num_row`, it naturally supports extended integer row IDs without segfaults or truncation.

## 4. Correct Capacity Calculations
- **Channels**: 2
- **Ranks per Channel**: 2
- **Bank Groups per Rank**: 4
- **Banks per Bank Group**: 4
- **Columns per Row**: 64
- **Address Granularity**: 64 bytes
- **Row Size (System-Wide)**: $2 \times 2 \times 4 \times 4 \times 64 \times 64 = 262,144$ bytes (256 KB)
- **Rows**: 10,485,760
- **Total Addressable Capacity**: $10,485,760 \times 262,144 = 2,748,779,069,440$ bytes (~2.75 TB / ~2.50 TiB).

## 5. Logical Allocation
Every expert weight allocation in `LLMSimulator` calls `MMapController::setNormal()`, which linearly increments a global byte offset. An 84 MiB expert (`88,080,384` bytes) consumes exactly 336 system-wide row increments. All 14,592 routed experts fit consecutively, consuming roughly 4,902,912 total rows.

## 6. Address-Translation Boundary Tests
A deterministic diagnostic linked directly against `libllm_system.so` yielded the following mapped coordinates for key logical boundaries:
- `Addr 0`: Ch:0 Ra:0 Bg:0 Ba:0 Ro:0 Co:0
- `Addr 64 (Next Cache Line)`: Ch:1 Ra:0 Bg:0 Ba:0 Ro:0 Co:0
- `Addr 262080 (End of Row)`: Ch:1 Ra:1 Bg:3 Ba:3 Ro:0 Co:63
- `Addr 34359738368 (After 32 GB original limit)`: Ch:0 Ra:0 Bg:0 Ba:0 Ro:131072 Co:0
- `Addr 1285268963327 (Highest Expert Boundary)`: Ch:1 Ra:1 Bg:3 Ba:3 Ro:4902911 Co:63

## 7. Alias Analysis
**No Physical Aliasing Occurs.**
The mathematical interleaving natively divides the byte offset by the system stride. Distinct logical byte addresses are translated to strictly distinct DRAM coordinates across the entire 1.285 TB range because the mapped `row` index never exceeds the supported 10,485,760 limit and does not wrap around. 

## 8. Limitations & Final Verdict
The model accurately represents the DDR5 latency and bandwidth characteristics for a theoretical massive-capacity setup. However, it relies exclusively on a custom extended-capacity timing abstraction rather than a standard JEDEC DDR5 organization. 

**Verdict:** PASS WITH CAVEATS
(The configuration provides functionally correct simulated timing for non-aliasing 1.285 TB access, but represents an abstraction rather than an exact physical memory module).
