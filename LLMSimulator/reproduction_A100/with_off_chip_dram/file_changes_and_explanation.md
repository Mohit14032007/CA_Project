# File Changes and Explanation

## Overview
This document outlines the source code modifications planned and executed across Phase 3, 4, and 5 for the A100 Single-GPU MoE with Off-Chip DRAM experiment. The goal of these changes is to introduce a secondary DRAM tier (DDR5) and direct specific expert weight tensor memory requests to this tier when the expert is not resident in the primary HBM tier.

## Phase 3 & 4: Dual-Memory Infrastructure
- **`src/hardware/hardware_config.h`**
  - Added `std::string offchip_dram_cfg_path = "";` to `SystemConfig` to specify the configuration file for the off-chip memory tier.
- **`src/hardware/device.h` & `src/hardware/device.cpp`**
  - Added `DRAMInterface* offchip_dram_interface = nullptr;` to the `Device` class to manage the secondary memory tier.
  - Initialized `offchip_dram_interface` alongside the primary `dram_interface` using the new `offchip_dram_cfg_path`.
- **`eval/test.cpp`**
  - Added parsing logic to extract `offchip_dram_cfg_path` from the YAML configuration and populate `system_config.offchip_dram_cfg_path`.

## Phase 5: Expert Weight Routing and Telemetry
- **`src/hardware/base.h`**
  - Introduced the `MemoryTarget` enum (`HBM`, `OFFCHIP_DRAM`) to explicitly designate the target memory tier for each memory request.
- **`src/hardware/device.h` & `src/hardware/device.cpp`**
  - Added a `std::set<int> resident_experts;` to track which experts are currently loaded in HBM.
  - Added utility functions `bool is_expert_resident(int expert_id)` and `void mark_expert_resident(int expert_id)`.
- **`src/module/tensor.h`**
  - Added `bool is_expert_weight = false;` to identify tensors belonging to expert weights.
  - Added `int expert_id = -1;` to map the weight tensor to its corresponding expert.
  - Added `MemoryTarget weight_target = MemoryTarget::HBM;` to control where the execution layer routes the memory request.
  - Added `std::vector<Tensor::Ptr> sister_weights;` and `bool load_sisters = false;` to allow fetching all of an expert's weights concurrently upon a cache miss.
- **`src/module/expert.cpp`**
  - Modified the `ExpertFFN` (3-way) construction to extract `gate_proj`, `up_proj`, and `down_proj` weight tensors.
  - Tagged these tensors with `is_expert_weight = true` and `expert_id`.
  - Configured `sister_weights` to link the three projection tensors of each expert together.
- **`src/module/linear.cpp`**
  - Modified `Linear::forward` to check residency: if the input tensor's `expert_id` is not resident, the weight tensor's `weight_target` is set to `MemoryTarget::OFFCHIP_DRAM`, and `load_sisters` is enabled. It then marks the expert as resident. Otherwise, `weight_target` remains `MemoryTarget::HBM`.
- **`src/hardware/linear_impl.cpp`**
  - Modified `LinearExecutionGPU` (called by `BatchedLinearExecutionGPU` for experts) to use the `weight_target` of the weight tensor in the `issueRamulator` call, effectively routing the read request to the appropriate DRAM interface.
  - Handled the `load_sisters` flag by issuing concurrent `issueRamulator` read requests for the sister weights to the `OFFCHIP_DRAM` target.
  - Implemented telemetry logging to `expert_load.csv` to capture the start time, end time, duration, and hit/miss status of every expert weight access.

## Phase 5 Cleanup / Final Rerun
- **`src/hardware/linear_impl.cpp`**
  - **Function:** `LinearExecutionGPU`
  - **Change:** Moved the `expert_load.csv` telemetry logging block to execute *after* the `issueRamulator` calls, and explicitly captured the returned `ExecStatus::memory_duration` from the weight and sister loads.
  - **Reason:** The previous implementation calculated `memory_cost` before the simulated latency was added, resulting in `0 ns` recorded durations for all cache misses.
  - **Validation:** Cache misses now correctly display `~981,447 ns` (0.98 ms) latencies matching DDR5 delays, while hits correctly display `0 ns` additional load latencies.
- **`src/hardware/device.cpp`**
  - **Function:** `Device::Device`
  - **Change:** Commented out the `[PHASE 4 VALIDATION]` synthetic read request directed to Ramulator B.
  - **Reason:** Prevented the production experiment run from processing the diagnostic synthetic request, ensuring perfectly isolated telemetry data in Phase 5.
  - **Validation:** `run.log` no longer prints the validation message, and `expert_load.csv` does not receive phantom Ramulator B memory durations prior to `Iteration 0`.

## Phase 7: Full Workload Scaling
- **`src/hardware/linear_impl.cpp`**
  - **Change:** Updated the `expert_load.csv` generation logic to use dynamic file paths driven by the `TRACE_DIR` environment variable, rather than a hardcoded static string.
  - **Reason:** Prevented trace leakage between consecutive simulation runs (B0 and B1) executed by the shell script. Without this fix, the second run would append to or overwrite the first run's trace data in a shared location.
