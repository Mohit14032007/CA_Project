# Phase 3 Validation

**CHECK 1: Does Device own two DRAMInterface instances?**
- **Evidence**: Yes. Modified `src/hardware/device.h` explicitly declares `DRAMInterface_Ptr dram_interface;` and `DRAMInterface_Ptr offchip_dram_interface;` within the `Device` class scope.

**CHECK 2: Are they independently constructed?**
- **Evidence**: Yes. In `src/hardware/device.cpp`, two separate factory calls to `DRAMInterface::Create(...)` uniquely heap-allocate distinct interfaces: `dram_interface = DRAMInterface::Create(...)` followed by `offchip_dram_interface = DRAMInterface::Create(...)`.

**CHECK 3: Does each interface have an independent Ramulator backend?**
- **Evidence**: Yes. Inside `DRAMInterface::DRAMInterface`, `Ramulator::Factory::create_frontend()` and `create_memory_system()` construct distinct heap objects for the `frontend` and `memory_system` internal state vectors, isolated via the class `this` scope. There are no global statically-shared Ramulator structures violating instance independence.

**CHECK 4: Does the existing HBM request path remain unchanged?**
- **Evidence**: Yes. `Device::run_ramulator` explicitly executes `dram_interface->HandleRequest(request, 0);`. No branching logic or conditions were added.

**CHECK 5: Do expert-weight requests still reach HBM?**
- **Evidence**: Yes. The `diff` between B0 logs and Phase 3 logs shows `0` substantive differences. Sub-module `Linear` trace logs confirm matching read (`RD`) energy costs strictly tied to the active HBM unit.

**CHECK 6: Do non-expert requests still reach HBM?**
- **Evidence**: Yes. Standard sequence attention matrix requests (`self_attention`, `QKV_proj`) log precisely the same memory accesses matching B0 baseline logs.

**CHECK 7: Does the second interface receive zero requests?**
- **Evidence**: Yes. `offchip_dram_interface` is structurally inaccessible from any memory insertion functions within the `Device` class due to lack of source-level branching logic in Phase 3. 

**CHECK 8: Is Logic-PIM still disabled?**
- **Evidence**: Yes. Configuration (`processor_type: GPU`) ensures complete absence of `LOGIC` or `PIM` entries in the output trace.

**CHECK 9: Is ET still 1?**
- **Evidence**: Yes. `TP1` configuration mapping appears explicitly inside the execution CSV headers, precisely mirroring B0 logs.

**CHECK 10: Is there still exactly one GPU?**
- **Evidence**: Yes. `num_device: 1` logs generate `GPU_N1_D1` output paths identical to B0.
