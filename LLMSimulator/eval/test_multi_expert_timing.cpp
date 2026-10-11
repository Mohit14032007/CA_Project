#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cassert>
#include "hardware/cluster.h"
#include "hardware/device.h"
#include "hardware/layer_impl.h"
#include "module/tensor.h"
#include "dram/dram_interface.h"
#include "dram/mmap_controller.h"

using namespace llm_system;

void print_coordinates(const std::string& label, const Ramulator::AddrVec_t& coords) {
    std::cout << "    " << label << " Coordinates: [";
    for (size_t i = 0; i < coords.size(); ++i) {
        std::cout << coords[i] << (i + 1 < coords.size() ? ", " : "");
    }
    std::cout << "] (Ch, Rk, BG, Ba, Ro, Co)\n";
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "DS-5C-7: Controlled Multi-Expert Timing & Cache Validation\n";
    std::cout << "======================================================================\n" << std::flush;

    auto host_start_total = std::chrono::high_resolution_clock::now();

    SystemConfig system_config;
    system_config.gpu_gen = "A100";
    system_config.use_ramulator = true;
    system_config.offchip_dram_cfg_path = "reproduction_A100/with_off_chip_dram/configs/dram_config_DDR5.yaml";

    Cluster::Ptr cluster = Cluster::Create(system_config, nullptr);
    Device::Ptr device = Device::Create(system_config, 0, cluster);
    device->perform_execution = true;

    // Expert cache capacity set to 100 MB: holds exactly 1 DeepSeek-V3 expert (~88 MB)
    // This allows controlled testing of hits, cold misses, and capacity evictions.
    device->cache_capacity_bytes = 100 * 1024 * 1024LL;
    device->reset_expert_cache();

    std::cout << "[INIT] Cluster and Device instantiated.\n";
    std::cout << "[INIT] HBM Execution Cache initialized (entries: " << cluster->execution_time_cache.size() << ").\n";
    std::cout << "[INIT] Expert LRU Capacity: " << device->cache_capacity_bytes << " bytes (100 MB).\n";

    const long long expert_size = 88080384LL; // 88,080,384 bytes
    const long long size_per_tensor = expert_size / 3; // 29,360,128 bytes each for W1, W2, W3

    struct AccessStep {
        int step_id;
        int expert_id;
        std::string description;
        bool expect_lru_hit;
    };

    // Controlled multi-expert access sequence:
    // Step 0: Expert 10 (Cold Miss) -> fetched from DDR5, inserted to LRU
    // Step 1: Expert 10 (Repeat Access -> LRU Hit) -> served from HBM
    // Step 2: Expert 17 (Cold Miss) -> fetched from DDR5, evicts Expert 10
    // Step 3: Expert 10 (Capacity Miss) -> evicted earlier, re-fetched from DDR5, evicts Expert 17
    // Step 4: Expert 42 (Cold Miss) -> distinct region, fetched from DDR5, evicts Expert 10
    std::vector<AccessStep> sequence = {
        {0, 10, "Expert 10 Cold Miss (Fetch from Off-Chip DDR5)", false},
        {1, 10, "Expert 10 Repeat Access (LRU Hit -> Resident in HBM)", true},
        {2, 17, "Expert 17 Cold Miss (Fetch from DDR5, Evict Expert 10)", false},
        {3, 10, "Expert 10 Capacity Miss (Evicted earlier, Re-fetch DDR5)", false},
        {4, 42, "Expert 42 Cold Miss (Distinct region, Fetch DDR5, Evict E10)", false}
    };

    int request_counter = 0;

    for (const auto& step : sequence) {
        std::cout << "\n----------------------------------------------------------------------\n";
        std::cout << "Step " << step.step_id << ": " << step.description << "\n";
        std::cout << "----------------------------------------------------------------------\n";

        auto host_step_start = std::chrono::high_resolution_clock::now();
        int e_id = step.expert_id;
        int layer_id = 1;

        // Check LRU resident status
        bool lru_hit = device->is_expert_resident(layer_id, e_id);
        std::cout << "  [LRU Decision] Expert " << e_id << " Resident: " 
                  << (lru_hit ? "HIT" : "MISS") 
                  << " (Expected: " << (step.expect_lru_hit ? "HIT" : "MISS") << ")\n";
        assert(lru_hit == step.expect_lru_hit);

        // Memory addresses for W1, W2, W3
        long long expert_base_addr = e_id * expert_size;
        long long w1_addr = expert_base_addr;
        long long w2_addr = expert_base_addr + size_per_tensor;
        long long w3_addr = expert_base_addr + 2 * size_per_tensor;

        // Create tensors
        Tensor::Ptr w1 = Tensor::Create("W1_E" + std::to_string(e_id), {1, (int)size_per_tensor}, "weight", device);
        Tensor::Ptr w2 = Tensor::Create("W2_E" + std::to_string(e_id), {1, (int)size_per_tensor}, "weight", device);
        Tensor::Ptr w3 = Tensor::Create("W3_E" + std::to_string(e_id), {1, (int)size_per_tensor}, "weight", device);

        std::vector<Tensor_Ptr> batch_tensors = {w1, w2, w3};

        if (!lru_hit) {
            // MISS: Load from OFFCHIP_DRAM
            w1->weight_target = MemoryTarget::OFFCHIP_DRAM;
            w2->weight_target = MemoryTarget::OFFCHIP_DRAM;
            w3->weight_target = MemoryTarget::OFFCHIP_DRAM;

            w1->allocateMemoryObject(MemoryObject::Create(MMap::ALL_CHANNEL, w1_addr, size_per_tensor, device->offchip_mmap_controller));
            w2->allocateMemoryObject(MemoryObject::Create(MMap::ALL_CHANNEL, w2_addr, size_per_tensor, device->offchip_mmap_controller));
            w3->allocateMemoryObject(MemoryObject::Create(MMap::ALL_CHANNEL, w3_addr, size_per_tensor, device->offchip_mmap_controller));

            // Log request metadata & mapped coordinates (Task 5)
            std::cout << "  [Requests Metadata]\n";
            Ramulator::AddrVec_t c1 = device->offchip_mmap_controller->getAddrVec(w1_addr, 0);
            Ramulator::AddrVec_t c2 = device->offchip_mmap_controller->getAddrVec(w2_addr, 0);
            Ramulator::AddrVec_t c3 = device->offchip_mmap_controller->getAddrVec(w3_addr, 0);

            std::cout << "    Req " << ++request_counter << ": W1_E" << e_id << " Addr=" << w1_addr << " Size=" << size_per_tensor << " B Target=OFFCHIP_DRAM\n";
            print_coordinates("W1", c1);
            std::cout << "    Req " << ++request_counter << ": W2_E" << e_id << " Addr=" << w2_addr << " Size=" << size_per_tensor << " B Target=OFFCHIP_DRAM\n";
            print_coordinates("W2", c2);
            std::cout << "    Req " << ++request_counter << ": W3_E" << e_id << " Addr=" << w3_addr << " Size=" << size_per_tensor << " B Target=OFFCHIP_DRAM\n";
            print_coordinates("W3", c3);

            double sim_time_before = device->offchip_dram_interface->time;
            size_t exec_cache_size_before = cluster->execution_time_cache.size();

            // Submit batch to Ramulator DDR5
            ExecStatus status = issueRamulatorBatch(
                device, LayerType::LINEAR, ProcessorType::GPU,
                DRAMRequestType::kRead, PIMOperandType::kDRAM, batch_tensors, MemoryTarget::OFFCHIP_DRAM);

            double sim_time_after = device->offchip_dram_interface->time;
            size_t exec_cache_size_after = cluster->execution_time_cache.size();

            // Task 8 Check: OFFCHIP_DRAM must bypass ExecutionCache
            assert(exec_cache_size_before == exec_cache_size_after);
            std::cout << "  [Cache Bypass] ExecutionCache size unchanged (" << exec_cache_size_after << "). OFFCHIP_DRAM bypassed timing cache.\n";

            // Task 6 Check: Batch timing vs per-request timing
            std::cout << "  [Timing Analysis]\n";
            std::cout << "    Simulated Start Time:      " << std::fixed << std::setprecision(2) << sim_time_before << " ns\n";
            std::cout << "    Simulated Completion Time: " << sim_time_after << " ns\n";
            std::cout << "    Simulated Memory Duration: " << status.memory_duration << " ns\n";
            std::cout << "    Ramulator Clock Delta:     " << (sim_time_after - sim_time_before) << " ns\n";
            assert(status.memory_duration == (sim_time_after - sim_time_before));

            std::cout << "    Per-Request Durations (Cumulative timestamps from batch start):\n";
            for (size_t i = 0; i < status.per_request_duration.size(); i++) {
                std::cout << "      W" << (i + 1) << " completion latency: " << status.per_request_duration[i] << " ns\n";
            }
            assert(status.per_request_duration.size() == 3);
            assert(status.per_request_duration[0] < status.per_request_duration[1]);
            assert(status.per_request_duration[1] < status.per_request_duration[2]);
            assert(status.per_request_duration[2] == status.memory_duration);

            // Mark resident in LRU cache
            device->mark_expert_resident(layer_id, e_id, expert_size);
            std::cout << "  [LRU State] Expert " << e_id << " marked resident. Current resident bytes: " 
                      << device->resident_bytes << " B (LRU entries: " << device->lru_expert_list.size() << ")\n";

        } else {
            // HIT: Access HBM
            w1->weight_target = MemoryTarget::HBM;
            w2->weight_target = MemoryTarget::HBM;
            w3->weight_target = MemoryTarget::HBM;

            // In HBM, assign via device memory mapping
            device->setMemoryObject(w1);
            device->setMemoryObject(w2);
            device->setMemoryObject(w3);

            std::cout << "  [Requests Metadata]\n";
            std::cout << "    Req " << ++request_counter << ": W1_E" << e_id << " Addr=" << w1->getMemoryObject()->getLogicAddr() << " Target=HBM (Cache Hit)\n";
            std::cout << "    Req " << ++request_counter << ": W2_E" << e_id << " Addr=" << w2->getMemoryObject()->getLogicAddr() << " Target=HBM (Cache Hit)\n";
            std::cout << "    Req " << ++request_counter << ": W3_E" << e_id << " Addr=" << w3->getMemoryObject()->getLogicAddr() << " Target=HBM (Cache Hit)\n";

            CacheKey hbm_key = std::make_tuple(
                LayerType::LINEAR, ProcessorType::GPU, DRAMRequestType::kRead,
                expert_size, MemoryTarget::HBM);

            bool in_exec_cache_before = (cluster->execution_time_cache.find(hbm_key) != cluster->execution_time_cache.end());
            std::cout << "  [Execution Cache State Before] Key present: " << (in_exec_cache_before ? "YES" : "NO") << "\n";

            double sim_time_before = device->dram_interface->time;

            ExecStatus status = issueRamulatorBatch(
                device, LayerType::LINEAR, ProcessorType::GPU,
                DRAMRequestType::kRead, PIMOperandType::kDRAM, batch_tensors, MemoryTarget::HBM);

            double sim_time_after = device->dram_interface->time;

            bool in_exec_cache_after = (cluster->execution_time_cache.find(hbm_key) != cluster->execution_time_cache.end());
            std::cout << "  [Execution Cache State After] Key present: " << (in_exec_cache_after ? "YES" : "NO") << "\n";
            assert(in_exec_cache_after);

            std::cout << "  [HBM Timing]\n";
            std::cout << "    Batch Memory Duration: " << status.memory_duration << " ns\n";
            std::cout << "    HBM Clock Delta:       " << (sim_time_after - sim_time_before) << " ns\n";

            // If it was already in execution cache, HBM simulation clock delta is 0!
            if (in_exec_cache_before) {
                std::cout << "    Execution Cache HIT verified: HBM Ramulator skipped (Clock delta = 0 ns).\n";
                assert(sim_time_after == sim_time_before);
            } else {
                std::cout << "    Execution Cache MISS verified: Simulated and cached into cluster execution_time_cache.\n";
            }
        }

        auto host_step_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> host_step_ms = host_step_end - host_step_start;
        std::cout << "  [Host Wall-Clock Runtime] Step " << step.step_id << ": " << host_step_ms.count() << " ms\n";
    }

    // Explicit Verification of Task 8: Execution Cache Bypass vs Reuse
    std::cout << "\n======================================================================\n";
    std::cout << "Verification of ExecutionCache Bypass vs Reuse Invariants (Task 8)\n";
    std::cout << "======================================================================\n";

    // 1. Two consecutive OFFCHIP_DRAM batches with identical parameters
    std::cout << "Test 8.1: Consecutive identical OFFCHIP_DRAM batches...\n";
    Tensor::Ptr off_t = Tensor::Create("off_test", {1, (int)size_per_tensor}, "weight", device);
    off_t->weight_target = MemoryTarget::OFFCHIP_DRAM;
    off_t->allocateMemoryObject(MemoryObject::Create(MMap::ALL_CHANNEL, 100000000LL, size_per_tensor, device->offchip_mmap_controller));
    std::vector<Tensor_Ptr> off_batch = {off_t};

    double t0 = device->offchip_dram_interface->time;
    ExecStatus off_stat1 = issueRamulatorBatch(device, LayerType::LINEAR, ProcessorType::GPU, DRAMRequestType::kRead, PIMOperandType::kDRAM, off_batch, MemoryTarget::OFFCHIP_DRAM);
    double t1 = device->offchip_dram_interface->time;
    ExecStatus off_stat2 = issueRamulatorBatch(device, LayerType::LINEAR, ProcessorType::GPU, DRAMRequestType::kRead, PIMOperandType::kDRAM, off_batch, MemoryTarget::OFFCHIP_DRAM);
    double t2 = device->offchip_dram_interface->time;

    std::cout << "  First OFFCHIP call duration:  " << off_stat1.memory_duration << " ns (DDR5 delta: " << (t1 - t0) << " ns)\n";
    std::cout << "  Second OFFCHIP call duration: " << off_stat2.memory_duration << " ns (DDR5 delta: " << (t2 - t1) << " ns)\n";
    assert(t1 - t0 > 0);
    assert(t2 - t1 > 0);
    std::cout << "  -> OFFCHIP_DRAM unconditionally invokes DDR5 Ramulator (ExecutionCache bypassed).\n";

    // 2. Two consecutive HBM batches with identical parameters
    std::cout << "Test 8.2: Consecutive identical HBM batches...\n";
    Tensor::Ptr hbm_t = Tensor::Create("hbm_test", {1, (int)size_per_tensor}, "weight", device);
    hbm_t->weight_target = MemoryTarget::HBM;
    device->setMemoryObject(hbm_t);
    std::vector<Tensor_Ptr> hbm_batch = {hbm_t};

    double h0 = device->dram_interface->time;
    ExecStatus hbm_stat1 = issueRamulatorBatch(device, LayerType::LINEAR, ProcessorType::GPU, DRAMRequestType::kRead, PIMOperandType::kDRAM, hbm_batch, MemoryTarget::HBM);
    double h1 = device->dram_interface->time;
    ExecStatus hbm_stat2 = issueRamulatorBatch(device, LayerType::LINEAR, ProcessorType::GPU, DRAMRequestType::kRead, PIMOperandType::kDRAM, hbm_batch, MemoryTarget::HBM);
    double h2 = device->dram_interface->time;

    std::cout << "  First HBM call duration:  " << hbm_stat1.memory_duration << " ns (HBM delta: " << (h1 - h0) << " ns)\n";
    std::cout << "  Second HBM call duration: " << hbm_stat2.memory_duration << " ns (HBM delta: " << (h2 - h1) << " ns)\n";
    assert(h1 - h0 > 0);
    assert(h2 - h1 == 0); // Exact ExecutionCache Hit
    assert(hbm_stat1.memory_duration == hbm_stat2.memory_duration);
    std::cout << "  -> HBM correctly populates and reuses ExecutionCache (Ramulator skipped on hit).\n";

    auto host_end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> host_total_sec = host_end_total - host_start_total;

    std::cout << "\n======================================================================\n";
    std::cout << "SUCCESS: All DS-5C-7 timing, cache, and coordination checks PASSED!\n";
    std::cout << "Total Host Wall-Clock Execution Time: " << host_total_sec.count() << " seconds\n";
    std::cout << "Total Final Simulated DDR5 Clock:     " << device->offchip_dram_interface->time << " ns\n";
    std::cout << "Total Final Simulated HBM Clock:      " << device->dram_interface->time << " ns\n";
    std::cout << "======================================================================\n" << std::flush;

    return 0;
}
