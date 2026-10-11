# 1. Parent repository state
- Parent commit: `419252761fbdb95b789778a02256d458a5537ec7`
- Branch: `main`

# 2. Model configuration path
- File: `src/model/model_config.h`
- Class/Function: `ModelConfig` constructor
- Explanation: The configuration structure tracks `num_routed_expert`, `top_k`, `expert_freq`, etc., which define the routing parameters of the MoE model.

# 3. MoE routing path
- File: `src/module/route.cpp`
- Class/Function: `Route`
- Explanation: Defines the top-k token routing probabilities. The model instantiates this for selecting which experts process which tokens.

# 4. Expert selection path
- File: `src/scheduler/scheduler.cpp`
- Class/Function: `Scheduler` and `Scheduler::getMaxMetadata()`
- Line number/range: ~144-173
- Explanation: In the scheduler, expert selections for requests are made. The uniform integer distribution `% model_config.num_routed_expert` is used.

# 5. Expert weight representation
- File: `src/hardware/cluster.cpp`
- Class/Function: `Cluster`
- Explanation: Tensors and weights for `ExpertFFN` operations are tracked and their data sizes/offsets are managed.

# 6. Expert execution path
- File: `src/module/expert.cpp` / `src/module/decoder.cpp`
- Class/Function: `ExpertFFN::forward()` and `Decoder::forward()`
- Line number/range: `decoder.cpp:134` (Create), `expert.cpp:136` (forward)
- Explanation: The decoder instantiates an `ExpertFFN` block (`get_module("expertFFN")`). The `ExpertFFN::forward` method constructs the forward-pass operations and tracks the timing stamps.

# 7. Current HBM memory path
- File: `src/hardware/device.cpp`
- Class/Function: `Device::run_ramulator()` and `Device::run_ideal()`
- Line number/range: ~173-202
- Explanation: The `Device` submits `DRAMRequest`s either through `run_ramulator` which pushes directly to `DRAMInterface`, or computes simple idealized times via `run_ideal`.

# 8. DRAMInterface construction
- File: `src/hardware/device.cpp`
- Class/Function: `Device::Device()` constructor
- Line number/range: ~59
- Explanation: A single `DRAMInterface` is currently created per `Device` instance: `dram_interface = DRAMInterface::Create(dram_cfg_path, memory_scale_factor);`

# 9. MMapController path
- File: `src/dram/mmap_controller.h` and `src/dram/mmap_controller.cpp`
- Explanation: MMapController is responsible for taking a logical tensor request address and translating it into the channel/rank/bank topology used by Ramulator.

# 10. Ramulator integration
- File: `src/dram/dram_interface.cpp`
- Class/Function: `DRAMInterface::Create()`
- Line number/range: ~15-18
- Explanation: Ramulator2 is invoked using `Ramulator::Config::parse_config_file()` and `Ramulator::Factory::create_memory_system()`. The `DRAMInterface` acts as a wrapper around these Ramulator objects.

# 11. Current memory configuration
- File: `dram_config_HBM3_80GB.yaml` (default used by A100 device configuration)
- Explanation: The A100 baseline and duplex experiments rely on HBM3. `memory_pool_dram_config.yaml` is available but not explicitly selected in the `Device` logic unless modified. HBM3 is currently the only active timing configuration in the A100 experiments.

# 12. Current processor/timing trace path
- File: `src/module/timeboard.cpp`
- Class/Function: `Timeboard::write_processor_trace()`
- Line number/range: ~147-149
- Explanation: `processor_trace.csv` is written out based on the collected `TimeStamp` objects appended to the timeboard. `ExpertFFN` execution is captured through string markers like `expertFFN`, `moe_gather`.

# 13. Current limitations relevant to off-chip DRAM experiment
- Memory channels and `DRAMInterface` are explicitly single-instance per `Device` (only HBM).
- The `ExpertFFN` weights are assumed to exist within this single HBM memory pool. 
- There is no mechanism to track a "load from off-chip to HBM" transfer latency before scheduling an `expertFFN` operation.

# 14. Proposed future insertion points
1. **Device memory-interface construction**: `Device::Device()` in `src/hardware/device.cpp`. Must add a second `DRAMInterface` instance specifically for off-chip (e.g., `offchip_dram_interface`).
2. **expert weight address generation/residency**: `ExpertFFN::forward()` or `Device::run_ramulator()`. Logic must be added to route memory read requests to the off-chip interface instead of the HBM interface if the expert is not resident.
3. **expert load completion**: A new latency addition step inside `Decoder::forward` or `ExpertFFN::forward` to stall the GPU pipeline while the off-chip DRAM is fetched.

# 15. Recommended architecture for Phase 2 onward
We recommend augmenting `Device` with `offchip_dram_interface`. When `expertFFN` operations are scheduled, the simulation must first compute a memory-transfer operation `OffChipToHBM` through `offchip_dram_interface`. The latency of this transfer should be appended as a new timestamp type before the standard `expertFFN` computation can begin on the HBM interface.
