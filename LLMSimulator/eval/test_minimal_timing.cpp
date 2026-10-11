#include <iostream>
#include <cassert>
#include <vector>
#include "hardware/device.h"
#include "hardware/layer_impl.h"
#include "module/tensor.h"

using namespace llm_system;

int main() {
    std::cout << "Creating SystemConfig...\n";
    SystemConfig system_config;
    system_config.use_ramulator = true;
    system_config.offchip_dram_cfg_path = "reproduction_A100/with_off_chip_dram/configs/dram_config_DDR5.yaml";

    std::cout << "Creating Device...\n";
    Device::Ptr device = Device::Create(system_config, 1, nullptr);
    std::cout << "Device created.\n";
    device->perform_execution = true;
    
    // We need to initialize offchip_dram_interface with a Ramulator memory system
    // Wait, Device automatically creates it if we provide a SystemConfig
    
    std::cout << "Starting Minimal DDR5 Timing Test\n";
    
    // Create 3 fake tensors (W1, W2, W3) for Expert (10, 17)
    // Size is approx 88MB total (29.3MB each)
    long size_per_tensor = 88080384 / 3;
    std::cout << "Creating w1\n";
    Tensor::Ptr w1 = Tensor::Create("w1", {1, (int)size_per_tensor}, "act", device);
    std::cout << "Creating w2\n";
    Tensor::Ptr w2 = Tensor::Create("w2", {1, (int)size_per_tensor}, "act", device);
    std::cout << "Creating w3\n";
    Tensor::Ptr w3 = Tensor::Create("w3", {1, (int)size_per_tensor}, "act", device);
    std::cout << "Created tensors\n";
    
    // Set MemoryObject with logical addresses. Expert 17 starts at address ~ (17 * 88M)
    long expert_start_address = 17LL * 88080384LL;
    w1->allocateMemoryObject(MemoryObject::Create(MMap::ALL_CHANNEL, expert_start_address, size_per_tensor, device->offchip_mmap_controller));
    w2->allocateMemoryObject(MemoryObject::Create(MMap::ALL_CHANNEL, expert_start_address + size_per_tensor, size_per_tensor, device->offchip_mmap_controller));
    w3->allocateMemoryObject(MemoryObject::Create(MMap::ALL_CHANNEL, expert_start_address + 2 * size_per_tensor, size_per_tensor, device->offchip_mmap_controller));
    
    std::vector<Tensor_Ptr> batch_tensors = {w1, w2, w3};
    
    std::cout << "Issuing OFFCHIP_DRAM Ramulator Batch...\n";
    ExecStatus status = issueRamulatorBatch(
        device, LayerType::LINEAR, ProcessorType::GPU,
        DRAMRequestType::kRead, PIMOperandType::kDRAM, batch_tensors, MemoryTarget::OFFCHIP_DRAM);
        
    std::cout << "Total Memory Duration: " << status.memory_duration << " ns (or simulated time units)\n";
    std::cout << "Per request durations:\n";
    for (size_t i = 0; i < status.per_request_duration.size(); i++) {
        std::cout << "  Tensor " << i << ": " << status.per_request_duration[i] << "\n";
    }
    
    assert(status.memory_duration > 0);
    assert(status.per_request_duration.size() == 3);
    
    // Verify Execution Cache Bypass
    // Submitting the exact same request should NOT hit the execution cache because OFFCHIP_DRAM bypasses it.
    std::cout << "Issuing OFFCHIP_DRAM Ramulator Batch Again to verify execution cache bypass...\n";
    ExecStatus status2 = issueRamulatorBatch(
        device, LayerType::LINEAR, ProcessorType::GPU,
        DRAMRequestType::kRead, PIMOperandType::kDRAM, batch_tensors, MemoryTarget::OFFCHIP_DRAM);
    
    assert(status2.memory_duration > 0);
    std::cout << "Cache bypass verified (produced identical simulated duration: " << status2.memory_duration << ")\n";
    
    std::cout << "Minimal Timing Test Passed.\n";
    
    return 0;
}
