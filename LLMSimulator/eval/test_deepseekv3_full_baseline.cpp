#include <iostream>
#include <fstream>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <vector>
#include <cmath>
#include <filesystem>

#include "common/type.h"
#include "dram/dram_interface.h"
#include "dram/memory_config.h"
#include "dram/mmap_controller.h"
#include "hardware/cluster.h"
#include "hardware/device.h"
#include "hardware/hardware_config.h"
#include "model/model.h"
#include "model/model_config.h"
#include "module/tensor.h"
#include "scheduler/scheduler.h"

using namespace llm_system;

static size_t count_file_lines(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return 0;
    size_t lines = 0;
    std::string line;
    while (std::getline(file, line)) {
        lines++;
    }
    return lines;
}

static uintmax_t get_file_size(const std::string& path) {
    std::error_code ec;
    uintmax_t size = std::filesystem::file_size(path, ec);
    return ec ? 0 : size;
}

void run_full_baseline(int num_layers, int expected_moe_layers, const std::string& stage_name, const std::string& trace_dir) {
    std::filesystem::create_directories(trace_dir);
    std::filesystem::remove(trace_dir + "/expert_routing.csv");
    std::filesystem::remove(trace_dir + "/expert_load.csv");
    std::filesystem::remove(trace_dir + "/expert_cache.csv");

    auto host_start = std::chrono::high_resolution_clock::now();

    // 1. Hardware System Configuration
    SystemConfig system_config;
    system_config.gpu_gen = "A100";
    system_config.num_node = 1;
    system_config.num_device = 1;
    system_config.use_ramulator = true;
    const char* ddr5_cfg_env = std::getenv("OFFCHIP_DRAM_CFG");
    system_config.offchip_dram_cfg_path = ddr5_cfg_env ? ddr5_cfg_env : "reproduction_A100/with_off_chip_dram/configs/dram_config_DDR5_retry.yaml";
    system_config.processor_type = {ProcessorType::GPU};
    system_config.high_processor_type = ProcessorType::GPU;
    system_config.low_processor_type = ProcessorType::LOGIC;
    system_config.decode_mode = true;
    system_config.prefill_mode = false;
    system_config.exit_out_of_memory = false;
    system_config.mem_cap_limit = false;

    // 2. Model Configuration: Exact DeepSeek-V3 architecture
    ModelConfig model_config = deepseekV3;
    model_config.e_tp_dg = 1;
    model_config.ne_tp_dg = 1;
    model_config.num_layers = num_layers;
    model_config.first_k_dense = 3; // First 3 layers are dense
    model_config.num_routed_expert = 256;
    model_config.top_k = 8;
    model_config.num_shared_expert = 1;
    model_config.hidden_dim = 7168;
    model_config.expert_intermediate_dim = 2048;
    model_config.ffn_way = 3;
    model_config.precision_byte = 2; // FP16
    model_config.input_len = 1;
    model_config.output_len = 2;
    model_config.dataset = "synthesis";

    size_t expected_experts = (size_t)expected_moe_layers * (size_t)model_config.top_k;
    long long expected_resident_bytes = (long long)expected_experts * 88080384LL;

    std::cout << "[CONFIG] Architecture: HiddenDim=" << model_config.hidden_dim 
              << ", RoutedExperts=" << model_config.num_routed_expert 
              << ", TopK=" << model_config.top_k 
              << ", DenseLayers=" << model_config.first_k_dense 
              << ", MoELayers=" << (model_config.num_layers - model_config.first_k_dense) 
              << ", ExpectedColdExperts=" << expected_experts 
              << " (" << expected_resident_bytes << " B / " << (expected_resident_bytes / (1024.0*1024.0*1024.0)) << " GB)\n" << std::flush;

    // 3. Instantiate Scheduler, Cluster, and Model Graph
    Scheduler::Ptr scheduler = Scheduler::Create(system_config, model_config, "none", 1, 8192, 1);
    Cluster::Ptr cluster = Cluster::Create(system_config, scheduler);
    Model model(model_config, cluster, scheduler);
    cluster->set_dependency();

    Device::Ptr device = cluster->get_device(0);
    device->cache_capacity_bytes = 80000000000LL; // 80 GB standard HBM budget
    device->reset_expert_cache();

    std::cout << "[INIT] Model Graph instantiated successfully. Cluster ready.\n" << std::flush;

    // 4. Trace DDR5 log before execution
    std::string ddr5_log_path = "log/cmd_ddr5.log.ch0";
    size_t initial_ddr5_lines = count_file_lines(ddr5_log_path);
    uintmax_t initial_ddr5_bytes = get_file_size(ddr5_log_path);
    double initial_sim_time = device->offchip_dram_interface->time;

    // 5. Populate Scheduler with 1 Decode Sequence
    scheduler->clear();
    scheduler->initRunningQueue();
    scheduler->pushDummySeq(model_config.input_len, model_config.output_len);
    scheduler->fillRunningQueue();

    std::cout << "\n--- Pass 1: Cold Start Execution (Fetch all required routed experts from DDR5) ---\n" << std::flush;
    auto metadata_pass1 = scheduler->setMetadata();
    assert(!metadata_pass1.empty());

    auto pass1_host_start = std::chrono::high_resolution_clock::now();
    cluster->run(metadata_pass1);
    auto pass1_host_end = std::chrono::high_resolution_clock::now();

    double pass1_sim_time = device->offchip_dram_interface->time - initial_sim_time;
    std::chrono::duration<double> pass1_host_sec = pass1_host_end - pass1_host_start;

    std::cout << "  Pass 1 Completed.\n";
    std::cout << "  Host Wall-Clock Runtime: " << pass1_host_sec.count() << " s\n";
    std::cout << "  Simulated Off-Chip DDR5 Time: " << pass1_sim_time << " ns (" << (pass1_sim_time / 1e6) << " ms)\n";
    std::cout << "  Resident Experts in Cache: " << device->lru_expert_list.size() 
              << ", Resident Bytes: " << device->resident_bytes << " B\n" << std::flush;
    assert(device->lru_expert_list.size() == expected_experts);
    assert(device->resident_bytes == expected_resident_bytes);

    // 6. Inspect Generated CSV Traces for Pass 1
    std::cout << "\n--- Trace Inspection for " << stage_name << " (Pass 1) ---\n";
    size_t routing_lines_pass1 = count_file_lines(trace_dir + "/expert_routing.csv");
    size_t load_lines_pass1 = count_file_lines(trace_dir + "/expert_load.csv");
    size_t cache_lines_pass1 = count_file_lines(trace_dir + "/expert_cache.csv");

    std::cout << "  expert_routing.csv: " << routing_lines_pass1 << " lines\n";
    std::cout << "  expert_load.csv:    " << load_lines_pass1 << " lines\n";
    std::cout << "  expert_cache.csv:   " << cache_lines_pass1 << " lines\n";

    size_t current_ddr5_lines = count_file_lines(ddr5_log_path);
    uintmax_t current_ddr5_bytes = get_file_size(ddr5_log_path);
    std::cout << "  DDR5 Trace Lines Added: " << (current_ddr5_lines - initial_ddr5_lines) 
              << " (" << ((current_ddr5_bytes - initial_ddr5_bytes) / (1024.0 * 1024.0)) << " MB)\n" << std::flush;

    // 7. Verify Pass 2: Warm Replay Execution (All experts already resident in HBM -> Hit Test)
    std::cout << "\n--- Pass 2: Warm Replay Execution (Expect 100% HBM Cache Hits) ---\n" << std::flush;
    double sim_time_before_pass2 = device->offchip_dram_interface->time;
    double hbm_time_before_pass2 = device->dram_interface->time;

    // Replay the same sequence: reset layer cursor to 0
    for (auto& bseq : metadata_pass1) {
        bseq->cur_layer = 0;
    }
    auto pass2_host_start = std::chrono::high_resolution_clock::now();
    cluster->run(metadata_pass1);
    auto pass2_host_end = std::chrono::high_resolution_clock::now();

    double pass2_ddr5_delta = device->offchip_dram_interface->time - sim_time_before_pass2;
    double pass2_hbm_delta = device->dram_interface->time - hbm_time_before_pass2;
    std::chrono::duration<double> pass2_host_sec = pass2_host_end - pass2_host_start;

    std::cout << "  Pass 2 Completed.\n";
    std::cout << "  Host Wall-Clock Runtime: " << pass2_host_sec.count() << " s\n";
    std::cout << "  DDR5 Clock Delta on Replay: " << pass2_ddr5_delta << " ns (0 ns expected)\n";
    std::cout << "  HBM Clock Delta on Replay:  " << pass2_hbm_delta << " ns\n";
    assert(pass2_ddr5_delta == 0.0);
    assert(device->lru_expert_list.size() == expected_experts);
    assert(device->resident_bytes == expected_resident_bytes);
    std::cout << "  -> SUCCESS: All experts hit the HBM cache on replay! Zero DDR5 latency incurred.\n" << std::flush;

    auto host_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_host_sec = host_end - host_start;
    std::cout << "\n" << stage_name << " Completed in " << total_host_sec.count() << " seconds.\n" << std::flush;

    size_t final_routing_lines = count_file_lines(trace_dir + "/expert_routing.csv");
    size_t final_load_lines = count_file_lines(trace_dir + "/expert_load.csv");
    size_t final_cache_lines = count_file_lines(trace_dir + "/expert_cache.csv");

    // Write completion status file upon actual successful completion of the full run
    std::string status_path = trace_dir + "/ds5c10_completion_status.txt";
    std::ofstream status_file(status_path);
    if (status_file.is_open()) {
        status_file << "STATUS=SUCCESS\n";
        status_file << "EXIT_CODE=0\n";
        status_file << "STAGE=" << stage_name << "\n";
        status_file << "MOE_LAYERS=" << expected_moe_layers << "\n";
        status_file << "TOTAL_LAYERS=" << num_layers << "\n";
        status_file << "EXPECTED_COLD_EXPERTS=" << expected_experts << "\n";
        status_file << "EXPECTED_RESIDENT_BYTES=" << expected_resident_bytes << "\n";
        status_file << "HOST_WALLCLOCK_SECONDS=" << std::fixed << std::setprecision(4) << total_host_sec.count() << "\n";
        status_file << "PASS1_HOST_SECONDS=" << std::fixed << std::setprecision(4) << pass1_host_sec.count() << "\n";
        status_file << "PASS1_SIMULATED_DDR5_NS=" << std::fixed << std::setprecision(0) << pass1_sim_time << "\n";
        status_file << "PASS1_SIMULATED_DDR5_MS=" << std::fixed << std::setprecision(4) << (pass1_sim_time / 1e6) << "\n";
        status_file << "PASS2_HOST_SECONDS=" << std::fixed << std::setprecision(4) << pass2_host_sec.count() << "\n";
        status_file << "PASS2_DDR5_DELTA_NS=" << std::fixed << std::setprecision(0) << pass2_ddr5_delta << "\n";
        status_file << "PASS2_HBM_DELTA_NS=" << std::fixed << std::setprecision(0) << pass2_hbm_delta << "\n";
        status_file << "RESIDENT_EXPERTS_COUNT=" << device->lru_expert_list.size() << "\n";
        status_file << "RESIDENT_BYTES=" << device->resident_bytes << "\n";
        status_file << "ROUTING_RECORDS_PASS1=" << routing_lines_pass1 << "\n";
        status_file << "LOAD_RECORDS_PASS1=" << load_lines_pass1 << "\n";
        status_file << "CACHE_RECORDS_PASS1=" << cache_lines_pass1 << "\n";
        status_file << "FINAL_ROUTING_RECORDS=" << final_routing_lines << "\n";
        status_file << "FINAL_LOAD_RECORDS=" << final_load_lines << "\n";
        status_file << "FINAL_CACHE_RECORDS=" << final_cache_lines << "\n";
        status_file.close();
        std::cout << "  Completion status written to " << status_path << "\n" << std::flush;
    }
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "DS-5C-10: DeepSeek-V3 Full-Model Memory-System Baseline Execution\n";
    std::cout << "======================================================================\n" << std::flush;

    // Phase A: Architectural & Mapping Audit Preflight
    MemoryConfig ddr5_cfg = ddr5_32gb_x16_2ch;
    auto mmap = MMapController::Create(ddr5_cfg);

    long long granul = ddr5_cfg.granul;
    long long num_cube = ddr5_cfg.num_cube;
    long long num_channel = ddr5_cfg.num_channel;
    long long num_rank = ddr5_cfg.num_rank;
    long long num_bg = ddr5_cfg.num_bankgroup;
    long long num_bank = ddr5_cfg.num_bank;
    long long num_col = ddr5_cfg.num_col;
    long long num_row = ddr5_cfg.num_row;

    long long divisor_before_row = granul * num_cube * num_channel * num_rank * num_bg * num_bank * num_col;
    long long total_capacity = divisor_before_row * num_row;
    long long max_expert_addr = 14592LL * 88080384LL; // 1,285,268,963,328 bytes

    long long max_row_gpu = max_expert_addr / divisor_before_row;
    double row_utilization = (double)max_row_gpu / (double)num_row * 100.0;

    std::cout << "  DDR5 Configuration: " << num_channel << " Channels, " << num_rank << " Ranks, "
              << num_bg << " BankGroups, " << num_bank << " Banks, " << num_col << " Cols, "
              << num_row << " Rows, " << granul << " B Granularity\n";
    std::cout << "  Row Step Size: " << divisor_before_row << " bytes (262,144 bytes)\n";
    std::cout << "  Total Addressable Capacity: " << total_capacity << " bytes (~2.74878 TB / 2.5 TiB)\n";
    std::cout << "  DeepSeek-V3 14,592 Routed Experts Footprint: " << max_expert_addr << " bytes (~1.285 TB)\n";
    std::cout << "  Production GPU Mapper Max Row: " << max_row_gpu << " (Row Capacity Utilization: " 
              << std::fixed << std::setprecision(3) << row_utilization << "%)\n";

    const char* trace_dir_env = std::getenv("TRACE_DIR");
    std::string full_exp_dir = trace_dir_env ? trace_dir_env : "reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/raw";

    // Execute Full Baseline: 60 total layers, 3 dense, 57 MoE layers
    run_full_baseline(60, 57, "DeepSeek-V3 Full Baseline", full_exp_dir);

    std::cout << "\n======================================================================\n";
    std::cout << "SUCCESS: DeepSeek-V3 Full Baseline Run Complete!\n";
    std::cout << "======================================================================\n" << std::flush;

    return 0;
}
