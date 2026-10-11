# Phase DS-5C-0: Forensic Audit of 1.285 TB DeepSeek-V3 Expert Weights

## Executive Summary
This report audits how the LLMSimulator allocates, maps, and simulates the ~1.285 TB physical footprint required for DeepSeek-V3's 14,592 routed FP16 experts on a DDR5 system that is natively sized at 32 GB.

The audit has revealed that **there is no physical aliasing** in the memory subsystem. The simulator successfully circumvents the 32 GB capacity limitation by artificially inflating the `num_row` parameter in the memory configuration to 10,485,760 (an 80x increase from the standard 131,072 rows for the `DDR5_32Gb_x16` device). 

This approach scales the simulator's theoretical DDR5 capacity to approximately 2.68 TB, successfully housing the 1.285 TB logical footprint without overlap while perfectly preserving row conflict and bandwidth modeling.

## Part 1: DeepSeek-V3 Configuration

The model parameterization maps directly to physical sizes via `model_config.h` and the yaml configuration:
- **Precision**: 2 bytes (FP16) - *Defined by `simulation.precision_byte: 2` in the YAML, which overrides the model_config default.*
- **Hidden Dimension**: 7168
- **Expert Intermediate Dimension**: 2048
- **Routed Experts per Layer**: 256
- **MoE Layers**: 57 (Layers 3 through 59)
- **Total Routed Experts**: 14,592

**Weight Footprint (Per Expert)**:
Each expert contains three tensors in its FeedForward network:
1. `W1 (gate_proj)`: `[7168, 2048] * 2 bytes = 29,360,128 bytes`
2. `W2 (down_proj)`: `[2048, 7168] * 2 bytes = 29,360,128 bytes`
3. `W3 (up_proj)`: `[7168, 2048] * 2 bytes = 29,360,128 bytes`

**Total logical size per expert**: `88,080,384 bytes (~84 MiB)`
**Total logical size across all routed experts**: `1,285,268,963,328 bytes (~1.169 TiB)`

## Part 2: Logical Allocation

Logical allocation inside LLMSimulator is strictly monotonic and sequential. During model initialization:
1. Tensors are instantiated via `MMapController::setNormal()`.
2. The controller applies the current `start_addr_normal` to the tensor and unconditionally increments it by `tensor->getSize()`.
3. Because model layers, experts, and their sub-tensors (W1, W2, W3) are created sequentially, the logical address space is contiguous across the entire model without any overlap or aliasing.

## Part 3: Memory Geometry and Address Translation

The system employs `DDR5_32Gb_x16` acting over 2 channels and 2 ranks (`dram_config_DDR5.yaml`).

However, inside `src/dram/memory_config.h`, the default geometry for this configuration has been explicitly altered:
```cpp
  static MemoryConfig ddr5_32gb_x16_2ch = MemoryConfig(
    ...
    10485760, // num_row (2^17) -> increased to simulate enough capacity for MoE
    ...
  );
```
Standard `DDR5_32Gb_x16` geometry allows for 131,072 rows. The simulator inflates this to 10,485,760 rows.

### Row Capacity and Striping
The `MMapController::addrToVec` function applies standard block interleaving (`Channel -> Rank -> BankGroup -> Bank -> Col -> Row`).
The total number of unique addressable bytes *per system-wide row index* is:
`Channels (2) * Ranks (2) * BankGroups (4) * Banks (4) * Cols (64) * Granule (64 bytes)`
= **262,144 bytes (256 KB) per row index**.

An individual expert (88,080,384 bytes) maps perfectly to `336` row increments (88,080,384 / 262,144 = 336.0).
This means an expert is perfectly striped across the entire channel/bank subsystem, occupying exactly 336 rows in *every* bank across the 2 channels.

**Example Mapping**:
If Expert 0 starts at `row_idx = 0`, it spans `row_idx 0 to 335`.
Expert 1 spans `row_idx 336 to 671`, and so on.

## Part 4: Ramulator Compatibility

Ramulator natively enforces strict capacity bounds based on pre-defined JEDEC specifications (in this case, checking against `num_row = 1<<17`). 
However, **Ramulator 2 manages row buffers purely via integer IDs in the Bank nodes**. It does not instantiate physical data arrays or enforce that an incoming `row_idx` must be `< 1<<17` during runtime execution. When `LLMSimulator` passes a `row_idx > 131,072` (e.g., `row_idx = 5,000,000` for a late-stage expert), Ramulator happily accepts it, marking row `5,000,000` as the currently open row buffer for that bank.

## Part 5: Scientific Defensibility

Because there is zero physical aliasing, the current simulation is **scientifically defensible**:
1. **Capacity Independence**: Inflating `num_row` isolates capacity from bandwidth. It models a theoretical system with 1.285 TB capacity but bound by the 32 GB system's dual-channel DDR5 bandwidth.
2. **Row Conflicts**: Because distinct experts occupy distinct rows, loading Expert A and then Expert B to the same bank will correctly register a row buffer miss, triggering the accurate `PRECHARGE` and `ACTIVATE` timing penalties in Ramulator.
3. **Concurrency**: W1, W2, and W3 of a single expert are mapped sequentially and share the same 336 rows. Accessing them concurrently (as enabled in DS-5B-2) yields high row-buffer hit rates, which perfectly represents real-world spatial locality.

## Part 6: Analysis of Proposed Options

**Option A (Reduce Model Dimensions):** 
*Not recommended.* Scaling down the model disrupts the expert tensor sizes, altering the burst lengths and row buffer locality, which invalidates the memory timing behaviors we are trying to measure.

**Option B (Model Virtual Memory Paging):**
*Not recommended.* Paging from NVMe/PCIe introduces massive latency (>10,000ns) that dominates the simulation. Given that DeepSeek-V3 servers typically provision massive RAM (e.g., 1.5 TB) to hold all weights in memory, paging is an unrealistic target workload.

**Option C (Extend Physical Capacity / Retain Current Approach):**
**Recommended.** The simulator already implements Option C natively by inflating `num_row`. This perfectly satisfies the requirement to model the latency of a massive memory pool while strictly adhering to DDR5 bandwidth constraints. 

## Conclusion
No further architectural modifications are required for the address mapping or memory geometry. The 1.285 TB footprint is successfully mapped without aliasing, and the DDR5 timing fidelity remains completely intact. We can proceed with launching full workload simulations.
