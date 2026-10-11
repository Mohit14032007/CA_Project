# B0 Baseline Validation

This document verifies the operational constraints of the isolated B0 Single-GPU HBM-only experiment configuration. 

**CHECK 1: Single GPU**
- **Evidence**: 
  - Configuration explicitly states `system.num_device: 1`.
  - Log output confirms the configuration: `mixtral_synthesis_32_2_GPU_N1_D1_TP1_DP1...`
  - Total parameter aggregation in log (`Total: 87.9521GB`) corresponds exactly to a single Mixtral instance, rather than a duplicated 4-device footprint.

**CHECK 2: No Logic-PIM**
- **Evidence**: 
  - Configuration specifies `processor_type: GPU`.
  - Simulator log traces (e.g., `Linear | 18.804us ... compute util: 0.0085791, memory util: 1.3133, Op/B 0.99959, GPU`) explicitly conclude with `GPU` instead of `PIM` or `LOGIC`. No Logic-PIM execution elements appear in the timeline.

**CHECK 3: No PE**
- **Evidence**: 
  - Configuration specifies `optimization.parallel_execution: off`.
  - The runtime trace output confirms sequential serialization in the `.csv` generation (e.g. `parallel_execution0` flag in filename).

**CHECK 4: ET = 1**
- **Evidence**: 
  - Configuration specifies `distribution.expert_tensor_degree: 1`.
  - The generated trace output filenames log `...TP1...` denoting the tensor parallel degree mapping, reflecting no distributed slicing across multiple GPUs.

**CHECK 5: No expert all-reduce**
- **Evidence**: 
  - Simulator trace log shows `moe_all_reduce_for_e_tp` takes exactly `0us` and consumes `0mJ` energy (`ACT: 0mJ | RD: 0mJ | WR: 0mJ | MAC: 0mJ`). It is effectively a no-op because there is only 1 device.

**CHECK 6: No multi-GPU expert sharding**
- **Evidence**: 
  - `num_device` = 1 and `expert_tensor_degree` = 1 dictates all 8 routed experts reside locally on the single device. The trace confirms memory sizes match standard local execution without node-transfer penalties.

**CHECK 7: Expert routing present**
- **Evidence**: 
  - The simulator log contains active routing boundaries (e.g., `moe_route | 0us | 32.603 - 32.603 | tensor(1, 4096) -> tensor(0, 4096) tensor(0, 4096) tensor(0, 4096) tensor(0, 4096) tensor(1, 4096) tensor(0, 4096) tensor(1, 4096) tensor(0, 4096)`), distributing the batched sequence effectively to experts 4 and 6 (denoted by the non-zero tensors).

**CHECK 8: Expert weight memory requests present**
- **Evidence**: 
  - Sub-modules of the selected experts (e.g., `expert_FFN_4 -> gate_proj -> Linear`) actively compute non-zero memory utilization (`memory util: 1.2864`) and register significant read energy (`RD: 3.2702mJ`), confirming that `issueRamulator()` processes expert `DRAMRequests` rather than analytic bypassed reads. 

**CHECK 9: Only one DRAMInterface/Ramulator instance is active**
- **Evidence**: 
  - Source code audit from Phase 1 confirmed that `src/hardware/device.cpp` only instantiates a singular `DRAMInterface::Create(...)` block per device. Because exactly one device is instantiated, there is exactly one Ramulator pipeline globally active.

**CHECK 10: No off-chip DRAM exists yet**
- **Evidence**: 
  - The single `DRAMInterface` manages a completely flat, unified `MMapController` addressing domain spanning the standard 80GB config (`mem_cap_limit: off`), with no separate latency penalty model or secondary PCIe-emulation interface existing in the C++ layer or the config.
