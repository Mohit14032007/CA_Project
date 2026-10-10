#include "hardware/device.h"

#include <random>
#include <string>
#include <stdexcept>

#include "common/assert.h"
#include "dram/dram_interface.h"
#include "dram/dram_request.h"
#include "dram/mmap_controller.h"
#include "dram/pimkernel/pim_kernel.h"
#include "module/module_graph.h"
#include "module/tensor.h"

namespace llm_system {

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<int> dis(0, 5);

Device::Device(SystemConfig config, int device_total_rank, Cluster_ptr cluster)
    : config(config),
      device_total_rank(device_total_rank),
      cluster(cluster),
      status() {
  compute_peak_flops = config.compute_peak_flops;
  memory_bandwidth = config.memory_bandwidth;
  memory_capacity = config.memory_capacity;
  device_local_rank = device_total_rank % config.num_device;

  top_module_graph = TopModuleGraph::Create(status);

  std::string dram_cfg_path;
  if(config.gpu_gen == "H100" || config.gpu_gen == "A100"){
    dram_cfg_path = "./dram_config_HBM3_80GB.yaml";
  }
  else if(config.gpu_gen == "B100" || config.gpu_gen == "B200"){
    dram_cfg_path = "./dram_config_HBM3E_192GB.yaml";
  }
  YAML::Node cfg = YAML::LoadFile(dram_cfg_path);

  double memory_scale_factor = 0;
  MemoryConfig memory_config = MemoryConfig(config.num_cube, config.num_logic_cube);
  if(config.gpu_gen == "H100" || config.gpu_gen == "A100"){
    // H100, HBM3 80GB, 5.2Gbps
    memory_scale_factor = 0.76923;
    memory_config = hbm3_80GB;
    memory_config.num_cube = config.num_cube;
    memory_config.num_logic_cube = config.num_logic_cube;
  }
  else if((config.gpu_gen == "B100") || (config.gpu_gen == "B200")){
    // B100, HBM3E 192GB, 8Gbps
    memory_scale_factor = 0.5; // 8.0Gbps pin rate's ideal bandwidth = 8000, tCK = 2000, 1 / 2GHz = 0.5
    memory_config = hbm3e_192GB;
    memory_config.num_cube = config.num_cube;
    memory_config.num_logic_cube = config.num_logic_cube;
  }

  dram_interface = DRAMInterface::Create(dram_cfg_path, memory_scale_factor);
  mmap_controller = MMapController::Create(memory_config);
  // Phase 4: Instantiate second memory tier for off-chip DRAM.
  std::string offchip_cfg = config.offchip_dram_cfg_path;
  if (offchip_cfg.empty()) {
      offchip_cfg = dram_cfg_path; // fallback to HBM placeholder if not provided
  }
  double offchip_scale_factor = memory_scale_factor;
  MemoryConfig offchip_memory_config = memory_config;
  if (offchip_cfg.find("DDR5") != std::string::npos) {
      offchip_scale_factor = 0.625;
      offchip_memory_config = ddr5_32gb_x16_2ch;
  }
  offchip_dram_interface = DRAMInterface::Create(offchip_cfg, offchip_scale_factor);
  offchip_mmap_controller = MMapController::Create(offchip_memory_config);

  // Phase 4 Validation Only: Synthetic test request to Ramulator B (Disabled for production)
  /*
  std::cout << "[PHASE 4 VALIDATION] Submitting synthetic read to Off-chip DRAM (Ramulator B)..." << std::endl;
  auto mem_obj = MemoryObject::Create(MMap::ALL_CHANNEL, 0, 64, offchip_mmap_controller);
  auto req = DRAMRequest::Create(DRAMRequestType::kRead);
  req->AddOperand(mem_obj, PIMOperandType::kDRAM);
  std::list<DRAMRequest::Ptr> req_list;
  req_list.push_back(req);
  offchip_dram_interface->HandleRequest(req_list, 0);
  std::cout << "[PHASE 4 VALIDATION] Synthetic request completed." << std::endl;
  std::cout << "[PHASE 4 VALIDATION] Off-chip Memory duration: " 
            << offchip_dram_interface->getExecStatus().memory_duration << " ns" << std::endl;
  */

  use_ramulator = config.use_ramulator;
  perform_execution = false;
}

void Device::set_dependency() { top_module_graph->set_dependency(); }

bool Device::check_module_graph_remain() {
  return top_module_graph->check_module_graph_remain();
};

void Device::run(std::vector<BatchedSequence::Ptr> sequences_metadata_list) {
  int dp_rank = device_total_rank / model_config.ne_tp_dg;
  top_module_graph->run(sequences_metadata_list.at(dp_rank));
}

void Device::restartGraph() { top_module_graph->restart_graph(); }

void Device::connectTopModuleGraph() {
  top_module_graph->connectDevice(get_ptr());
}

void Device::reset_status() { status = StatusBoard(); }
void Device::reset_timeboard() { top_module_graph->reset_timeboard(); }

void Device::add_module(std::string name, Module_ptr module) {
  cluster->add_module(device_total_rank, name, module);
}

void Device::setMemoryObject(Tensor::Ptr tensor) {
  if (tensor->weight_target == MemoryTarget::OFFCHIP_DRAM && offchip_mmap_controller != nullptr) {
    offchip_mmap_controller->setMemoryObject(tensor);
  } else {
    mmap_controller->setMemoryObject(tensor);
  }
}

void Device::addExecutionCache(ExecStatus& exec_status, CacheKey key) {
  auto& cache = cluster->execution_time_cache;
  cache.emplace(key, exec_status);
}

void Device::addExecutionCache(ExecStatus& exec_status, LayerType layer_type,
                               ProcessorType processor_type,
                               DRAMRequestType dram_reqeust_type, long size, MemoryTarget target) {
  CacheKey key =
      std::make_tuple(layer_type, processor_type, dram_reqeust_type, size, target);
  auto& cache = cluster->execution_time_cache;
  cache.emplace(key, exec_status);
}

bool Device::checkExecutionCache(ExecStatus& exec_status, CacheKey key) {
  auto& cache = cluster->execution_time_cache;
  if (const auto& cache_iter = cache.find(key); cache_iter != cache.end()) {
    exec_status = (*cache_iter).second;
    return true;
  } else {
    return false;
  }
}

bool Device::checkExecutionCache(CacheKey key) {
  auto& cache = cluster->execution_time_cache;
  if (const auto& cache_iter = cache.find(key); cache_iter != cache.end()) {
    ExecStatus exec_status = (*cache_iter).second;
    ExecStatus& _status = dram_interface->getExecStatus();
    _status = exec_status;
    return true;
  } else {
    return false;
  }
}

void Device::setExecStatus(ExecStatus& exec_status_) {
  exec_status = exec_status_;
}

ExecStatus Device::getHighExecStatus() {
  ExecStatus return_status = high_exec_status;
  high_exec_status = ExecStatus();
  return return_status;
}

ExecStatus Device::getLowExecStatus() {
  ExecStatus return_status = low_exec_status;
  low_exec_status = ExecStatus();
  return return_status;
}

ExecStatus Device::getExecStatus() {
  ExecStatus return_status = exec_status;
  exec_status = ExecStatus();
  return return_status;
}

// check whether execution must be performed, and ramulator
void Device::execution(LayerType layer_type,
                       const std::vector<Tensor::Ptr>& tensor_list,
                       const BatchedSequence::Ptr sequences_metadata,
                       const LayerInfo layer_info) {
  if (perform_execution) {
    dram_interface->resetCounter();
    cluster->executor.execution(layer_type, tensor_list, sequences_metadata,
                                config.processor_type, layer_info,
                                use_ramulator, get_ptr());
  }
}

void Device::execution_ramulator(LayerType layer_type,
                                 std::vector<Tensor::Ptr> tensor_list) {
  Tensor::Ptr input = tensor_list.at(0);
  Tensor::Ptr weight = tensor_list.at(1);
  Tensor::Ptr output = tensor_list.at(2);

  std::cout << weight->name << std::endl;
}

void Device::run_ramulator(DRAMRequest_Ptr dram_request, MemoryTarget target) {
  std::list<DRAMRequest::Ptr> request;
  request.push_back(dram_request);
  if (target == MemoryTarget::OFFCHIP_DRAM && offchip_dram_interface != nullptr) {
    offchip_dram_interface->HandleRequest(request, 0);
  } else {
    dram_interface->HandleRequest(request, 0);
  }
}

void Device::run_ideal(DRAMRequestType dram_request_type, Tensor_Ptr tensor){
  long total_size = tensor->getSize(); // Byte
  if (total_size == 0) {
    return;
  }
  MemoryConfig memory_config = mmap_controller->getConfig();
  int num_cube = memory_config.num_cube;
  int num_channel = memory_config.num_channel; // 32 (not legacy, pCH)
  int num_col = memory_config.num_col;
  int granul = mmap_controller->getGranul();

  long total_read = total_size / granul;
  long rw_cmd_to_cube_0 = (total_read % num_cube == 0) ? (total_read / num_cube) : ((total_read / num_cube) + 1);

  long rw_cmd_to_pCH_0 = (rw_cmd_to_cube_0 % num_channel == 0) ? (rw_cmd_to_cube_0 / num_channel) : ((rw_cmd_to_cube_0 / num_channel) + 1);
  long rw_cmd_to_pCH_1 = (rw_cmd_to_cube_0 % num_channel == 1) ? (rw_cmd_to_cube_0 / num_channel) : ((rw_cmd_to_cube_0 / num_channel) + 1);

  dram_interface->resetCounter();
  dram_interface->getExecStatus().act_count = (((rw_cmd_to_pCH_0 + rw_cmd_to_pCH_1) < num_col) ? 1 : ((rw_cmd_to_pCH_0 + rw_cmd_to_pCH_1) / num_col));
  if(dram_request_type == DRAMRequestType::kRead){
    dram_interface->getExecStatus().read_count = (rw_cmd_to_pCH_0 + rw_cmd_to_pCH_1);
  }
  else if(dram_request_type == DRAMRequestType::kWrite){
    dram_interface->getExecStatus().write_count = (rw_cmd_to_pCH_0 + rw_cmd_to_pCH_1);
  }
}

void Device::initializeDRAM(int ProcessorType, DramEnergy dramEnergy) {
  int num_pseudo_ch = 0;
  if (ProcessorType == (int)(ProcessorType::GPU)) {
    num_pseudo_ch = mmap_controller->getConfig().num_cube * mmap_controller->getConfig().num_channel / 2;
  }
  else {
    num_pseudo_ch = mmap_controller->getConfig().num_cube * mmap_controller->getConfig().num_channel;
  }
  dramEnergy.kACT_energy_j_ *= num_pseudo_ch;
  dramEnergy.kREAD_energy_j_ *= num_pseudo_ch;
  dramEnergy.kWRITE_energy_j_ *= num_pseudo_ch;

  dramEnergy.kALL_ACT_energy_j_ *= num_pseudo_ch;
  dramEnergy.kALL_READ_energy_j_ *= num_pseudo_ch;
  dramEnergy.kALL_WRITE_energy_j_ *= num_pseudo_ch;

  top_module_graph->initializeDRAM(ProcessorType, dramEnergy);
}

bool Device::is_expert_resident(int layer_id, int expert_id) {
  static bool init = false;
  static std::ofstream csv;
  if (!init) {
    const char* trace_dir_env = std::getenv("TRACE_DIR");
    std::string path = "expert_cache.csv";
    if (trace_dir_env) path = std::string(trace_dir_env) + "/expert_cache.csv";
    csv.open(path, std::ios::app);
    csv.seekp(0, std::ios::end);
    if (csv.tellp() == 0) {
      csv << "layer_id,expert_id,action,resident_bytes\n";
    }
    init = true;
  }

  auto it = resident_experts.find({layer_id, expert_id});
  if (it != resident_experts.end()) {
    // Cache HIT: Move to MRU position (back of the list)
    lru_expert_list.erase(it->second);
    lru_expert_list.push_back({layer_id, expert_id});
    it->second = std::prev(lru_expert_list.end());
    csv << layer_id << "," << expert_id << ",HIT," << resident_bytes << "\n";
    csv.flush();
    return true;
  }
  csv << layer_id << "," << expert_id << ",MISS," << resident_bytes << "\n";
  csv.flush();
  return false;
}

void Device::mark_expert_resident(int layer_id, int expert_id, long long expert_size) {
  // If it's already resident, we just return (is_expert_resident already moves to MRU)
  if (resident_experts.find({layer_id, expert_id}) != resident_experts.end()) {
    return;
  }

  static std::ofstream csv;
  static bool init = false;
  if (!init) {
    const char* trace_dir_env = std::getenv("TRACE_DIR");
    std::string path = "expert_cache.csv";
    if (trace_dir_env) path = std::string(trace_dir_env) + "/expert_cache.csv";
    csv.open(path, std::ios::app);
    init = true;
  }

  // Cache MISS: Evict LRU experts until there is enough capacity
  while (resident_bytes + expert_size > cache_capacity_bytes && !lru_expert_list.empty()) {
    auto lru_expert = lru_expert_list.front();
    lru_expert_list.pop_front();
    resident_experts.erase(lru_expert);
    resident_bytes -= expert_size; 
    csv << lru_expert.first << "," << lru_expert.second << ",EVICT," << resident_bytes << "\n";
  }

  // Insert the new expert at MRU position
  lru_expert_list.push_back({layer_id, expert_id});
  resident_experts[{layer_id, expert_id}] = std::prev(lru_expert_list.end());
  resident_bytes += expert_size;
  csv << layer_id << "," << expert_id << ",INSERT," << resident_bytes << "\n";
  csv.flush();
}

};  // namespace llm_system
