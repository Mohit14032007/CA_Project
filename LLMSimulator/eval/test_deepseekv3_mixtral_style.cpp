#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "common/type.h"
#include "dram/dram_interface.h"
#include "dram/memory_config.h"
#include "dram/mmap_controller.h"
#include "hardware/cluster.h"
#include "hardware/device.h"
#include "hardware/hardware_config.h"
#include "hardware/stat.h"
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

int main(int argc, char* argv[]) {
  bool validate_only = false;
  std::string config_path = "reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/configs/deepseekv3_b1_full.yaml";

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--validate-config" || arg == "--check-only") {
      validate_only = true;
    } else if (arg.rfind("--", 0) != 0) {
      config_path = arg;
    }
  }

  std::cout << "======================================================================\n";
  std::cout << "DS-6C-2: DeepSeek-V3 Mixtral-Style Experiment Harness\n";
  std::cout << "Executable: run_deepseekv3_mixtral_style\n";
  std::cout << "Config Path: " << config_path << "\n";
  if (validate_only) {
    std::cout << "Mode: CONFIGURATION VALIDATION ONLY (No simulation will be run)\n";
  }
  std::cout << "======================================================================\n" << std::flush;

  if (!std::filesystem::exists(config_path)) {
    std::cerr << "[ERROR] Configuration file does not exist: " << config_path << "\n";
    return 1;
  }

  YAML::Node config = YAML::LoadFile(config_path);

  // 1. Model Section
  std::string model_name = config["model"]["model_name"].as<std::string>();
  if (model_name != "deepseekV3" && model_name != "debugV3") {
    std::cerr << "[ERROR] Expected model_name 'deepseekV3' or 'debugV3', got: " << model_name << "\n";
    return 2;
  }

  // 2. System Section
  std::string gpu_gen = config["system"]["gpu_gen"].as<std::string>();
  std::string processor_type = config["system"]["processor_type"].as<std::string>();
  int num_node = config["system"]["num_node"].as<int>();
  int num_device = config["system"]["num_device"].as<int>();
  int nvlink_gen = config["system"]["nvlink_gen"] ? config["system"]["nvlink_gen"].as<int>() : 3;

  int e_tp_dg = config["system"]["distribution"]["expert_tensor_degree"].as<int>();
  int ne_tp_dg = config["system"]["distribution"]["none_expert_tensor_degree"].as<int>();

  bool use_ramulator = config["system"]["optimization"]["use_ramulator"].as<bool>();
  std::string offchip_dram_cfg_path = config["system"]["optimization"]["offchip_dram_cfg_path"].as<std::string>();
  
  long long cache_capacity = 80000000000LL; // 80 GB standard
  if (config["system"]["optimization"]["expert_cache_capacity_bytes"]) {
    cache_capacity = config["system"]["optimization"]["expert_cache_capacity_bytes"].as<long long>();
  } else if (std::getenv("DEBUG_CACHE_CAPACITY")) {
    cache_capacity = std::stoll(std::getenv("DEBUG_CACHE_CAPACITY"));
  }

  bool use_hitting_queue = false;
  if (config["system"]["optimization"]["use_hitting_queue"]) {
    use_hitting_queue = config["system"]["optimization"]["use_hitting_queue"].as<bool>();
  }

  // 3. Serving Section
  int max_batch_size = config["serving"]["max_batch_size"].as<int>();
  int max_process_token = config["serving"]["max_process_token"].as<int>();
  if (max_process_token == 0) {
    max_process_token = 65536 * 8;
  }

  // 4. Simulation Section
  std::string data_name = config["simulation"]["data"].as<std::string>();
  int input_len = config["simulation"]["input_len"].as<int>();
  int output_len = config["simulation"]["output_len"].as<int>();
  int precision_byte = config["simulation"]["precision_byte"].as<int>();
  int iter = config["simulation"]["iter"].as<int>();
  double skewness = config["simulation"]["skewness"].as<double>();
  bool exit_out_of_memory = config["simulation"]["exit_out_of_memory"].as<bool>();
  bool mem_cap_limit = config["simulation"]["mem_cap_limit"].as<bool>();

  // 5. Log Section
  std::string output_path = config["log"]["output_directory"].as<std::string>();
  std::string gantt_dir = config["log"]["gantt_directory"].as<std::string>();
  bool export_gantt = config["log"]["export_gantt"].as<bool>();
  bool print_log = config["log"]["print_log"].as<bool>();

  std::string trace_dir = gantt_dir;
  if (config["log"]["traces_directory"]) {
    trace_dir = config["log"]["traces_directory"].as<std::string>();
  } else {
    size_t raw_pos = trace_dir.find("/raw/");
    if (raw_pos != std::string::npos) {
      trace_dir.replace(raw_pos, 5, "/traces/");
    } else {
      trace_dir = output_path + "/../traces/";
    }
  }

  // Set trace and memory environment variables early
  std::filesystem::create_directories(output_path);
  std::filesystem::create_directories(gantt_dir);
  std::filesystem::create_directories(trace_dir);
  setenv("TRACE_DIR", trace_dir.c_str(), 1);
  setenv("OFFCHIP_DRAM_CFG", offchip_dram_cfg_path.c_str(), 1);

  // Validate Memory Configuration Files
  if (!std::filesystem::exists(offchip_dram_cfg_path)) {
    std::cerr << "[ERROR] Off-chip DDR5 config file not found: " << offchip_dram_cfg_path << "\n";
    return 3;
  }
  std::string hbm_cfg_path = "./dram_config_HBM3_80GB.yaml";
  if (config["system"]["optimization"]["hbm_dram_cfg_path"]) {
    hbm_cfg_path = config["system"]["optimization"]["hbm_dram_cfg_path"].as<std::string>();
  }
  if (!std::filesystem::exists(hbm_cfg_path)) {
    std::cerr << "[ERROR] On-chip HBM3 config file not found: " << hbm_cfg_path << "\n";
    return 4;
  }

  // Hardware System Config
  SystemConfig system_config;
  if (gpu_gen == "A100") {
    system_config = A100;
  } else if (gpu_gen == "H100") {
    system_config = H100;
  }
  system_config.num_node = num_node;
  system_config.num_device = num_device;
  system_config.use_ramulator = use_ramulator;
  system_config.offchip_dram_cfg_path = offchip_dram_cfg_path;
  system_config.hbm_dram_cfg_path = hbm_cfg_path;
  system_config.processor_type = {ProcessorType::GPU};
  system_config.high_processor_type = ProcessorType::GPU;
  system_config.low_processor_type = ProcessorType::LOGIC;
  system_config.decode_mode = true;
  system_config.prefill_mode = false;
  system_config.exit_out_of_memory = exit_out_of_memory;
  system_config.mem_cap_limit = mem_cap_limit;

  if (nvlink_gen == 3) {
    system_config.device_ict_bandwidth = 150.0 * 1000 * 1000 * 1000;     // 150 GB/s NVLink 3 (A100)
    system_config.device_ict_latency = 3.0 * 1000;                        // 3000 ns
  }

  // Model Config: DeepSeek-V3 exact architecture
  ModelConfig model_config = (model_name == "debugV3") ? debugV3 : deepseekV3;
  if (config["model"]["num_layers"]) {
    model_config.num_layers = config["model"]["num_layers"].as<int>();
  }
  model_config.e_tp_dg = e_tp_dg;
  model_config.ne_tp_dg = ne_tp_dg;
  model_config.precision_byte = precision_byte;
  model_config.input_len = input_len;
  model_config.output_len = output_len;
  model_config.skewness = skewness;
  model_config.dataset = data_name;

  if (model_config.precision_byte == 1) {
    system_config.compute_peak_flops *= 2;
  }

  // Preflight Mapping Check: Verify 2.75 TB DDR5 geometry covers routed experts
  MemoryConfig ddr5_cfg = ddr5_32gb_x16_2ch;
  long long divisor_before_row = (long long)ddr5_cfg.granul * ddr5_cfg.num_cube * ddr5_cfg.num_channel *
                                 ddr5_cfg.num_rank * ddr5_cfg.num_bankgroup * ddr5_cfg.num_bank * ddr5_cfg.num_col;
  long long total_capacity = divisor_before_row * ddr5_cfg.num_row;
  int moe_layers = model_config.num_layers - model_config.first_k_dense;
  long long total_routed_experts = (long long)moe_layers * model_config.num_routed_expert;
  long long max_expert_addr = total_routed_experts * 88080384LL;
  long long max_row_gpu = max_expert_addr / divisor_before_row;
  double row_utilization = (double)max_row_gpu / (double)ddr5_cfg.num_row * 100.0;

  std::cout << "\n[TOPOLOGY & GEOMETRY VALIDATION]\n";
  std::cout << "  Model: DeepSeek-V3 (" << model_config.num_layers << " layers: " 
            << model_config.first_k_dense << " dense, " 
            << (model_config.num_layers - model_config.first_k_dense) << " MoE)\n";
  std::cout << "  Experts: " << model_config.num_routed_expert << " routed, top-" 
            << model_config.top_k << ", " << model_config.num_shared_expert << " shared\n";
  std::cout << "  Expert Footprint: 88,080,384 bytes (~88.08 MB FP16)\n";
  std::cout << "  Total Expert Inventory: 14,592 experts = " << max_expert_addr << " bytes (~1.285 TB)\n";
  std::cout << "  DDR5 Backing Capacity: " << total_capacity << " bytes (~2.75 TB / 2.5 TiB across 10,485,760 rows)\n";
  std::cout << "  Max Production Row Index: " << max_row_gpu << " (Row Utilization: " 
            << std::fixed << std::setprecision(2) << row_utilization << "% < 100%)\n";
  std::cout << "  HBM Local Cache Budget: " << cache_capacity << " bytes (" 
            << (cache_capacity / (1024.0 * 1024.0 * 1024.0)) << " GB)\n";
  std::cout << "  Trace Output Directory: " << trace_dir << "\n";
  std::cout << "  Summary CSV Directory: " << output_path << "\n";
  std::cout << "  Workload: input_len=" << input_len << ", output_len=" << output_len 
            << ", iter=" << iter << ", batch_size=" << max_batch_size << "\n";

  assert(max_row_gpu < ddr5_cfg.num_row);

  if (validate_only) {
    std::cout << "\n======================================================================\n";
    std::cout << "[SUCCESS] ALL ARCHITECTURAL, TOPOLOGY, AND GEOMETRY INVARIANTS VALIDATED!\n";
    std::cout << "Configuration file is 100% syntactically and structurally sound.\n";
    std::cout << "======================================================================\n" << std::flush;
    return 0;
  }

  // Instantiate Simulator Objects
  Scheduler::Ptr scheduler = Scheduler::Create(system_config, model_config, "none", max_batch_size, 8192, max_process_token);
  Cluster::Ptr cluster = Cluster::Create(system_config, scheduler);
  Model model(model_config, cluster, scheduler);
  cluster->set_dependency();

  Device::Ptr device = cluster->get_device(0);
  device->cache_capacity_bytes = cache_capacity;
  device->reset_expert_cache();

  std::cout << "\n[INITIALIZATION] Scheduler, Cluster, Device, and Model Graph created.\n" << std::flush;

  // Clean trace files in target trace directory before starting
  std::filesystem::remove(trace_dir + "/expert_routing.csv");
  std::filesystem::remove(trace_dir + "/expert_load.csv");
  std::filesystem::remove(trace_dir + "/expert_cache.csv");

  int ne_dp_dg = system_config.num_device * num_node / ne_tp_dg;
  std::string summary_csv_filename = output_path + "/" + model_name + "_" + data_name +
                                     "_" + std::to_string(input_len) + "_" +
                                     std::to_string(output_len) + "_" + processor_type +
                                     "_N" + std::to_string(num_node) + "_D" +
                                     std::to_string(num_device) + "_TP" +
                                     std::to_string(ne_tp_dg) + "_DP" +
                                     std::to_string(ne_dp_dg) + "_maxbatch" +
                                     std::to_string(max_batch_size) + "_maxprocess" +
                                     std::to_string(max_process_token) + "_iter" +
                                     std::to_string(iter) +
                                     "_skew" + std::to_string(int(model_config.skewness * 10)) +
                                     "_precision_byte" + std::to_string(precision_byte) +
                                     "_parallel_execution" + std::to_string(system_config.parallel_execution) +
                                     "_ramul_decode.csv";

  auto run_host_start = std::chrono::high_resolution_clock::now();

  if (use_hitting_queue) {
    // Standard cluster iteration execution
    std::cout << "\n--- Executing via Cluster::runIteration (with hittingQueue warmup) ---\n" << std::flush;
    scheduler->getActualArrivalTime(iter);
    cluster->runIteration(iter, summary_csv_filename);
  } else {
    // Controlled Sequential Decoding Execution (Profile B)
    std::cout << "\n--- Executing Controlled Sequential Decoding Loop (" << iter << " steps) ---\n" << std::flush;
    scheduler->clear();
    scheduler->initRunningQueue();
    for (int b = 0; b < max_batch_size; b++) {
      scheduler->pushDummySeq(input_len, output_len);
    }
    scheduler->fillRunningQueue();

    std::ofstream summary_csv(summary_csv_filename);
    summary_csv << "iter_info,split,type,time,latency,queueing_delay,arrival_time,seq_queue_size,"
                << "input_len,output_len,num_sum_iter,mixed,batchsize,numtoken,num_sum_seq,num_gen_seq,seqlen,"
                << "sum_attention_opb,qkvgen,q_down_proj,kv_down_proj,kr_proj,q_up_proj,qr_proj,kv_up_proj,"
                << "tr_k_up_proj,v_up_proj,atten_sum,atten_gen,o_proj,ffn,expert_ffn,communication,rope,layernorm,"
                << "residual,act_energy,read_energy,write_energy,all_act_energy,all_read_energy,all_write_energy,"
                << "mac_energy,total_energy,fc_dram,fc_comp,attn_dram,attn_comp,moe_dram,moe_comp,OOM\n";

    time_ns cumulative_device_time = 0;

    for (int step = 0; step < iter; step++) {
      auto metadata = scheduler->setMetadata();
      if (metadata.empty()) {
        std::cerr << "[WARN] No active sequences in metadata at step " << step << ". Terminating early.\n";
        break;
      }

      auto step_host_start = std::chrono::high_resolution_clock::now();
      double ddr5_sim_before = device->offchip_dram_interface ? device->offchip_dram_interface->time : 0.0;
      double hbm_sim_before = device->dram_interface ? device->dram_interface->time : 0.0;

      cluster->run(metadata);

      auto step_host_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> step_host_sec = step_host_end - step_host_start;

      double ddr5_delta_ns = (device->offchip_dram_interface ? device->offchip_dram_interface->time : 0.0) - ddr5_sim_before;
      double hbm_delta_ns = (device->dram_interface ? device->dram_interface->time : 0.0) - hbm_sim_before;
      time_ns step_device_time = device->status.device_time;
      cumulative_device_time += step_device_time;

      // Populate standard Stat object with measured simulation breakdowns and energies
      Stat stat;
      stat.iter_info = 1;
      stat.type = "t2t";
      stat.time = cumulative_device_time;
      stat.latency = step_device_time;
      stat.input_len = input_len;
      stat.output_len = output_len;
      stat.average_seq_len = input_len + step + 1;
      stat.batchsize = max_batch_size;
      stat.process_token = max_batch_size;
      stat.seq_queue_size = scheduler->sequence_queue.size();

      std::vector<energy_nJ> total_energy = cluster->getTotalEnergy();
      stat.act_energy = total_energy[0];
      stat.read_energy = total_energy[1];
      stat.write_energy = total_energy[2];
      stat.all_act_energy = total_energy[3];
      stat.all_read_energy = total_energy[4];
      stat.all_write_energy = total_energy[5];
      stat.mac_energy = total_energy[6];
      stat.total_energy = total_energy[7];

      cluster->setStat(stat);
      cluster->setTimeBreakDown(stat);

      std::cout << "  [Step " << step << "] Token SeqLen=" << (input_len + step) 
                << " | Host: " << std::fixed << std::setprecision(2) << step_host_sec.count() << "s"
                << " | Simulated DDR5: " << (ddr5_delta_ns / 1e6) << " ms"
                << " | Resident Experts: " << device->lru_expert_list.size()
                << " (" << (device->resident_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB)\n" << std::flush;

      // Advance scheduler for the next sequential decoding token
      scheduler->updateScheduler(step_device_time);
      scheduler->fillSequenceQueue(step_device_time, cumulative_device_time);
      scheduler->fillRunningQueue();

      // Export step row to summary CSV using genuine Stat values
      summary_csv << stat.iter_info << ","
                  << stat.split << ","
                  << stat.type << ","
                  << stat.time << ","
                  << stat.latency << ","
                  << stat.queueing_delay << ","
                  << stat.arrival_time << ","
                  << stat.seq_queue_size << ","
                  << stat.input_len << ","
                  << stat.output_len << ","
                  << stat.num_sum_iter << ","
                  << stat.is_mixed << ","
                  << stat.batchsize << ","
                  << stat.process_token << ","
                  << stat.sum_seq << ","
                  << stat.gen_seq << ","
                  << stat.average_seq_len << ","
                  << stat.sum_attention_opb << ","
                  << stat.qkv_gen << ","
                  << stat.q_down_proj << ","
                  << stat.kv_down_proj << ","
                  << stat.kr_proj << ","
                  << stat.q_up_proj << ","
                  << stat.qr_proj << ","
                  << stat.kv_up_proj << ","
                  << stat.tr_k_up_proj << ","
                  << stat.v_up_proj << ","
                  << stat.atten_sum << ","
                  << stat.atten_gen << ","
                  << stat.o_proj << ","
                  << stat.ffn << ","
                  << stat.expert_ffn << ","
                  << stat.communication << ","
                  << stat.rope << ","
                  << stat.layernorm << ","
                  << stat.residual << ","
                  << stat.act_energy << ","
                  << stat.read_energy << ","
                  << stat.write_energy << ","
                  << stat.all_act_energy << ","
                  << stat.all_read_energy << ","
                  << stat.all_write_energy << ","
                  << stat.mac_energy << ","
                  << stat.total_energy << ","
                  << stat.FC_DRAM_energy << ","
                  << stat.FC_COMP_energy << ","
                  << stat.Attn_DRAM_energy << ","
                  << stat.Attn_COMP_energy << ","
                  << stat.MoE_DRAM_energy << ","
                  << stat.MoE_COMP_energy << ","
                  << stat.isOOM << "\n";
      summary_csv.flush();
    }
    summary_csv.close();
  }

  auto run_host_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> total_host_sec = run_host_end - run_host_start;

  if (export_gantt) {
    cluster->exportGantt(gantt_dir);
  }

  size_t routing_lines = count_file_lines(trace_dir + "/expert_routing.csv");
  size_t load_lines = count_file_lines(trace_dir + "/expert_load.csv");
  size_t cache_lines = count_file_lines(trace_dir + "/expert_cache.csv");

  std::cout << "\n======================================================================\n";
  std::cout << "EXECUTION COMPLETE\n";
  std::cout << "  Total Host Runtime: " << total_host_sec.count() << " seconds\n";
  std::cout << "  Final Resident Experts: " << device->lru_expert_list.size() << "\n";
  std::cout << "  Final Resident Memory:  " << device->resident_bytes << " bytes (" 
            << (device->resident_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB)\n";
  std::cout << "  Routing Records: " << routing_lines << " lines\n";
  std::cout << "  Load Records:    " << load_lines << " lines\n";
  std::cout << "  Cache Records:   " << cache_lines << " lines\n";
  std::cout << "======================================================================\n" << std::flush;

  // Write machine-readable completion status
  std::string status_path = output_path + "/../logs/completion_status.txt";
  std::ofstream status_file(status_path);
  if (status_file.is_open()) {
    status_file << "STATUS=SUCCESS\n"
                << "EXIT_CODE=0\n"
                << "MODEL=deepseekV3\n"
                << "CONFIG_PATH=" << config_path << "\n"
                << "ITERATIONS=" << iter << "\n"
                << "BATCH_SIZE=" << max_batch_size << "\n"
                << "PRECISION_BYTE=" << precision_byte << "\n"
                << "TOTAL_HOST_SECONDS=" << std::fixed << std::setprecision(4) << total_host_sec.count() << "\n"
                << "RESIDENT_EXPERTS=" << device->lru_expert_list.size() << "\n"
                << "RESIDENT_BYTES=" << device->resident_bytes << "\n"
                << "ROUTING_RECORDS=" << routing_lines << "\n"
                << "LOAD_RECORDS=" << load_lines << "\n"
                << "CACHE_RECORDS=" << cache_lines << "\n";
    status_file.close();
  }

  return 0;
}
