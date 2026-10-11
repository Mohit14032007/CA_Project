#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>
#include "dram/mmap_controller.h"
#include "dram/memory_config.h"
#include "hardware/cluster.h"
#include "hardware/device.h"
#include "hardware/layer_impl.h"
#include "module/tensor.h"

using namespace llm_system;

int main() {
    SystemConfig sys_config;
    sys_config.gpu_gen = "A100";
    sys_config.offchip_dram_cfg_path = "./reproduction_A100/with_off_chip_dram/configs/dram_config_DDR5.yaml";
    Cluster::Ptr cluster = Cluster::Create(sys_config, nullptr);
    Device::Ptr device = Device::Create(sys_config, 0, cluster);
    device->setPerformExecution(true);
    
    // Create tensors representing W1, W2, W3 for two distinct experts (Expert 1 and Expert 2)
    // They will have identical sizes, but distinct logical addresses.
    long long tensor_size = 29360128; // 29.36 MB

    // We only need to test issueRamulatorBatch with MemoryTarget::OFFCHIP_DRAM
    auto t1_w1 = Tensor::Create("E1_W1", {7168, 2048}, "weight", device, 2);
    auto t1_w2 = Tensor::Create("E1_W2", {2048, 7168}, "weight", device, 2);
    auto t1_w3 = Tensor::Create("E1_W3", {7168, 2048}, "weight", device, 2);
    
    // Assign addresses explicitly to ensure distinctness in Ramulator
    device->setMemoryObject(t1_w1); t1_w1->weight_target = MemoryTarget::OFFCHIP_DRAM;
    device->setMemoryObject(t1_w2); t1_w2->weight_target = MemoryTarget::OFFCHIP_DRAM;
    device->setMemoryObject(t1_w3); t1_w3->weight_target = MemoryTarget::OFFCHIP_DRAM;
    
    auto t2_w1 = Tensor::Create("E2_W1", {7168, 2048}, "weight", device, 2);
    auto t2_w2 = Tensor::Create("E2_W2", {2048, 7168}, "weight", device, 2);
    auto t2_w3 = Tensor::Create("E2_W3", {7168, 2048}, "weight", device, 2);
    
    device->setMemoryObject(t2_w1); t2_w1->weight_target = MemoryTarget::OFFCHIP_DRAM;
    device->setMemoryObject(t2_w2); t2_w2->weight_target = MemoryTarget::OFFCHIP_DRAM;
    device->setMemoryObject(t2_w3); t2_w3->weight_target = MemoryTarget::OFFCHIP_DRAM;
    
    std::cout << "E1_W1 Addr: " << t1_w1->getMemoryObject()->getLogicAddr() << "\n";
    std::cout << "E2_W1 Addr: " << t2_w1->getMemoryObject()->getLogicAddr() << "\n";

    std::vector<Tensor_Ptr> e1_batch = {t1_w1, t1_w2, t1_w3};
    std::vector<Tensor_Ptr> e2_batch = {t2_w1, t2_w2, t2_w3};

    std::cout << "\n--- Submitting Expert 1 (MISS) ---\n";
    ExecStatus status1 = issueRamulatorBatch(device, LayerType::LINEAR, ProcessorType::GPU, DRAMRequestType::kRead, PIMOperandType::kDRAM, e1_batch, MemoryTarget::OFFCHIP_DRAM);
    std::cout << "Status1 memory_duration: " << status1.memory_duration << "\n";
    std::cout << "Status1 per_request_duration size: " << status1.per_request_duration.size() << "\n";
    
    std::cout << "\n--- Submitting Expert 2 (MISS, identical sizes) ---\n";
    ExecStatus status2 = issueRamulatorBatch(device, LayerType::LINEAR, ProcessorType::GPU, DRAMRequestType::kRead, PIMOperandType::kDRAM, e2_batch, MemoryTarget::OFFCHIP_DRAM);
    std::cout << "Status2 memory_duration: " << status2.memory_duration << "\n";
    std::cout << "Status2 per_request_duration size: " << status2.per_request_duration.size() << "\n";

    if (status1.memory_duration == status2.memory_duration) {
        std::cout << "WARNING: Timing is perfectly identical. Did it hit the cache, or is the trace identical?\n";
    } else {
        std::cout << "SUCCESS: Timing differs, meaning cache was bypassed correctly and trace was re-evaluated!\n";
    }
    
    std::cout << "\n--- Testing Resident Status API ---\n";
    bool is_res_e1 = device->is_expert_resident(0, 1);
    std::cout << "Is Expert 1 resident initially? " << (is_res_e1 ? "Yes" : "No") << "\n";
    
    std::cout << "Marking Expert 1 as resident...\n";
    device->mark_expert_resident(0, 1, tensor_size * 3);
    
    bool is_res_e1_after = device->is_expert_resident(0, 1);
    std::cout << "Is Expert 1 resident after mark? " << (is_res_e1_after ? "Yes" : "No") << "\n";
    
    return 0;
}
