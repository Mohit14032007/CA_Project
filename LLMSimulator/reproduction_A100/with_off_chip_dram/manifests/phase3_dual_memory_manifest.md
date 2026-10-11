# Phase 3 Dual-Memory Manifest

## 1. Objective
Modify the `Device` memory ownership layer so that a single GPU Device can own two independent `DRAMInterface` instances (HBM and Off-chip DRAM). No requests are routed to the off-chip DRAM in this phase.

## 2. Original architecture
    Device
      |
      +-- one DRAMInterface (dram_interface)

## 3. New architecture
    Device
      |
      +-- HBM DRAMInterface (dram_interface)
      |
      +-- Off-chip DRAMInterface (offchip_dram_interface)

## 4. Modified source files

### `src/hardware/device.h`
- **Original responsibility**: Manages the single GPU memory and compute logic block.
- **Modification**: Added `DRAMInterface_Ptr offchip_dram_interface` and `MMapController_Ptr offchip_mmap_controller`.
- **Why required**: Provides structural capacity to manage a second tier of Ramulator instances on a per-device basis.
- **Connected files**: `src/hardware/device.cpp`, `src/hardware/hardware_config.h`
- **Phase**: Phase 3

### `src/hardware/device.cpp`
- **Original responsibility**: Implements initialization and execution dispatches for the `Device`.
- **Modification**: Added explicit instantiations for `offchip_dram_interface` and `offchip_mmap_controller` inside the `Device::Device` constructor. 
- **Why required**: Actually allocates the independent Ramulator backend objects. We duplicate the initialization path using the safe preexisting configuration for placeholder safety.
- **Connected files**: `src/dram/dram_interface.cpp`, `src/dram/mmap_controller.cpp`
- **Phase**: Phase 3

## 5. DRAMInterface ownership model
`Device` directly owns standard `std::shared_ptr` objects for `dram_interface` and `offchip_dram_interface`. It acts as the exclusive orchestrator. 

## 6. Constructor creation path
```text
Device::Device()
   ↓
dram_interface = DRAMInterface::Create(dram_cfg_path)
mmap_controller = MMapController::Create(memory_config)
   ↓
offchip_dram_interface = DRAMInterface::Create(dram_cfg_path)  // Placeholder
offchip_mmap_controller = MMapController::Create(memory_config) // Placeholder
```

## 7. Configuration path
Configuration is completely unmodified compared to B0 except for output directory strings. The dual instantiation piggybacks off `a100_gpu_baseline.yaml`'s derived safe parameters temporarily. 

## 8. Request-routing path
Unchanged. `Device::run_ramulator()` natively directs purely to `dram_interface` ignoring the new interface completely.

## 9. What remains unchanged
All `Device::run_ramulator()` calls, Ramulator internal channel mappings, HBM cycle limits, and MoE `ExpertFFN` tensor addressing remains completely unchanged. 

## 10. What is explicitly deferred to Phase 4/5
- Loading specific technology configuration files for the off-chip instance (e.g. DDR5 logic vs HBM)
- Marking expert tensors and conditionally routing them via `run_ramulator`.
- Implementing expert prefetch and eviction limits.
