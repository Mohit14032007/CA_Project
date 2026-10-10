# Phase DS-4A: DEEPSEEK-V3 DDR5/RAMULATOR AUDIT

## 1. Audit Objective
To determine exactly what memory system DS-4 simulated, analyze the Ramulator mapping and capacity representations, investigate execution-cache artifacts, and determine the scientific validity of the reported 243,771 ns latency and 361.32 GB/s effective bandwidth metrics.

## 2. DS-4 Configuration Inspected
- Model: DeepSeek-V3 FP16 (57 MoE layers, 256 routed experts/layer).
- HBM Cache: 80 GB.
- Off-chip DRAM: `reproduction_A100/with_off_chip_dram/configs/dram_config_DDR5.yaml`
- Execution mapping: `use_ramulator = true`

## 3. Exact DDR5 Geometry & Capacity
- **Configuration Preset:** `DDR5_32Gb_x16` (32 Gb density, x16 DQ width, internal prefetch 16).
- **Physical Ranks:** 2 (from YAML override)
- **Physical Channels:** 2 (from YAML override)
- **Capacity Derivation:** 
  - A 32-bit channel (`m_channel_width=32`) using x16 devices requires 2 devices per rank.
  - Rank capacity = 2 devices × 32 Gb = 64 Gb = 8 GB.
  - Channel capacity = 2 ranks × 8 GB = 16 GB.
  - **Total Capacity = 2 channels × 16 GB = 32 GB.**

## 4. Address Mapping Analysis (RoBaRaCoCh)
- The physical address is logically shifted by a transaction offset (`m_tx_offset` = 6 bits, representing 64B cache line).
- The mapper `RoBaRaCoCh` sequentially extracts the lower bits corresponding to Channel (1 bit), Column (6 bits), Rank (1 bit), BankGroup (2 bits), Bank (2 bits), and Row (17 bits).
- This consumes exactly 35 bits of the address (including tx_offset). 
- **Critical finding:** Any address bits beyond the 35th bit are ignored by `slice_lower_bits`. Thus, the physical address space wraps around strictly every 32 GB.

## 5. DeepSeek Expert Address Analysis
- `MMapController::setNormal` sequentially allocates unique contiguous memory blocks for every initialized Tensor by tracking a `start_addr_normal` pointer.
- The 14,592 routed experts (W1, W2, W3, ~29 MB each) are allocated fully unique and contiguous simulated addresses.
- **1.285 TB vs DDR5 Capacity:** Because the fully instantiated address space is 1.285 TB and the modeled physical space wraps at 32 GB, the modeled Ramulator space aliases heavily (~40 logical experts map to identical physical rows). 
- However, Ramulator models **timing only**, not storage contents. Aliasing only artificially impacts row-buffer hit rates, rather than causing data corruption. 

## 6. Execution-Cache Analysis & Ramulator Evaluations
- **Cache Key:** `device->checkExecutionCache` builds a key from `(layer_type, processor_type, dram_request_type, tensor->getSize(), target)`. It explicitly excludes `address` and `expert_id`.
- **Finding:** Every sister tensor (W1, W2, W3) shares the identical size of 29,360,128 bytes.
- Out of 172,215 total off-chip load requests (57,405 misses × 3 tensors), **Ramulator was called exactly 1 time**. 
- The cache reused the identical simulated latency for the remaining 172,214 tensor evaluations. 

## 7. Derivation of 243,771 ns and 361.32 GB/s
- **Why was the latency 243,771 ns?** 
  - The latency is the aggregate of W1 + W2 + W3 (3 × 81,257 ns).
  - The 81,257 ns latency for a single 29 MB tensor is abnormally fast for dual-channel DDR5.
- **The 32-Channel Chimera Artifact:** 
  - The off-chip `MMapController` is instantiated using the `hbm3_80GB` config (since `gpu_gen == "A100"`), which generates an `AddrVec` assuming **32 channels**.
  - Inside `Read.cpp`, the simulator filters these requests: `if (addr_vec.at(0) == 0)` only allows channels 0 and 1 into Ramulator.
  - Thus, **15/16ths of the 29 MB tensor traffic is silently dropped**, and Ramulator only simulates ~1.835 MB of traffic across its 2 DDR5 channels.
  - Ramulator models this 1.835 MB load at physically accurate DDR5 speeds (resulting in 81,257 ns), but the simulator logs this latency against the *entire* 29 MB payload.
  - This mathematically inflates the effective bandwidth by 16x, resulting in the 361.32 GB/s metric (which closely reflects a 32-channel DDR5 system).

## 8. Scientific Validity Analysis
- **Queueing/Contention/Prefetching:** Invalid. Because `checkExecutionCache` intercepts 99.9% of requests, memory controller queueing, channel contention, and prefetch vs. demand scheduling will fundamentally not exist.
- **Bandwidth:** Invalid. The 361.32 GB/s is an artifact of mixing a 32-channel HBM address generation scheme with a 2-channel Ramulator filter.

## 9. Final Classification
**MUST RERUN**

Before proceeding to DS-5 or prefetching research, the simulator must be fixed to correctly route off-chip DDR5 traffic.

## 10. Required Future Changes (DO NOT IMPLEMENT NOW)
1. Provide the off-chip `MMapController` with a distinct `MemoryConfig` matching the DDR5 geometry (2 channels), preventing the 15/16ths traffic drop.
2. The execution cache (`checkExecutionCache`) must be bypassed or modified for off-chip DRAM if realistic queueing, contention, or prefetching dynamics are to be measured.

---

### Audit Questionnaire Answers

1. **What is the actual DDR5 capacity?** 32 GB.
2. **What is the actual address range?** 35 physical bits (wrapping every 32 GB).
3. **Can it represent the full 1.285 TB expert address space?** No. 
4. **Can different experts alias?** Yes, heavily (modulo 32 GB).
5. **What exactly does RoBaRaCoCh do?** Slices the lowest 35 bits of the physical address into Channel, Column, Rank, BankGroup, Bank, and Row vectors while silently dropping higher bits.
6. **Does every miss independently reach Ramulator?** No.
7. **What exactly does checkExecutionCache cache?** Memory access latencies keyed strictly by payload size (and target type), entirely ignoring addresses.
8. **How many actual Ramulator timing evaluations occurred?** Exactly 1.
9. **What does 243,771 ns actually represent?** The cached Ramulator latency for processing 1/16th of a 29 MB tensor (1.835 MB) across 2 DDR5 channels, multiplied by 3 (for W1, W2, W3).
10. **Is 361.32 GB/s physically meaningful or only an effective simulated scalar?** It is an artifact scalar representing a chimera 32-channel DDR5 system.
11. **Is DS-4 suitable for load-vs-compute analysis?** No, the latency is highly artifactual.
12. **Is DS-4 suitable as the baseline for future prefetching research?** No, the execution cache fundamentally prevents simulation of prefetch vs. demand contention.
13. **Does DS-4 need to be rerun?** Yes.
14. **What should DS-5 be after this audit?** DS-5 must be fixing the off-chip MMapController configuration and the execution cache bypassing logic to yield a scientifically sound baseline.
