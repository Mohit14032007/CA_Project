#include <memory>

#include "common/type.h"
#include "dram/dram_interface.h"
#include "dram/dram_request.h"
#include "hardware/layer_impl.h"
#include "module/tensor.h"

namespace llm_system {
class DRAMRequest;
class Tensor;

using Tensor_Ptr = std::shared_ptr<Tensor>;
using DRAMRequest_Ptr = std::shared_ptr<DRAMRequest>;

ExecStatus LinearExecutionGPU(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr output,
                              bool use_ramulator) {
  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  double m = input->shape[0];
  double k = input->shape[1];
  double n = weight->shape[1];

  hw_metric total_flops = 2.0 * m * k * n;
  hw_metric total_memory_size = (m * k + k * n + m * n) * weight->precision_byte;

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  exec_status.compute_duration = compute_duration;
  
  if (use_ramulator) {
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input, MemoryTarget::HBM);

    time_ns start_time = device->get_time();
    ExecStatus weight_status =
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, weight, weight->weight_target);
    exec_status += weight_status;
    time_ns weight_memory_cost = weight_status.memory_duration;
    
    if (weight->load_sisters) {
      for (auto sister : weight->sister_weights) {
        ExecStatus sister_status =
            issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                           DRAMRequestType::kRead, PIMOperandType::kDRAM, sister, MemoryTarget::OFFCHIP_DRAM);
        exec_status += sister_status;
        weight_memory_cost += sister_status.memory_duration;
      }
      weight->load_sisters = false;
    }

    if (weight->is_expert_weight) {
      time_ns memory_cost = 0;
      if (weight->weight_target == MemoryTarget::OFFCHIP_DRAM) {
          memory_cost = weight_memory_cost;
      } else {
          // It was a hit, no OFFCHIP_DRAM delay
          memory_cost = 0;
      }
      time_ns end_time = start_time + memory_cost;

      static bool init_expert_load = false;
      static std::ofstream csv;
      static int request_id = 0;
      if (!init_expert_load) {
        const char* trace_dir_env = std::getenv("TRACE_DIR");
        std::string path = "expert_load.csv";
        if (trace_dir_env) {
          path = std::string(trace_dir_env) + "/expert_load.csv";
        }
        csv.open(path, std::ios::app);
        csv.seekp(0, std::ios::end);
        if (csv.tellp() == 0) {
          csv << "request_id,layer_id,expert_id,weight_tensor,size_bytes,start,end,duration,memory_source,memory_destination,cache_hit_or_miss\n";
        }
        init_expert_load = true;
      }
      
      long total_size = weight->getSize();
      for (auto sister : weight->sister_weights) {
          total_size += sister->getSize();
      }

      std::string memory_source = (weight->weight_target == MemoryTarget::OFFCHIP_DRAM) ? "OFFCHIP_DRAM" : "HBM";
      std::string hit_miss = (weight->weight_target == MemoryTarget::OFFCHIP_DRAM) ? "miss" : "hit";

      csv << request_id++ << ","
          << weight->expert_layer_id << ","
          << weight->expert_id << ","
          << weight->name << "," // could also say "all_weights"
          << total_size << ","
          << start_time << ","
          << end_time << ","
          << memory_cost << ","
          << memory_source << ","
          << "HBM" << ","
          << hit_miss << "\n";
      csv.flush();
    }

    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output, MemoryTarget::HBM);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, weight);
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  exec_status.opb = total_flops / total_memory_size;

  return exec_status;
};

ExecStatus LinearExecutionLogic(Device_Ptr device, Tensor_Ptr input,
                                Tensor_Ptr weight, Tensor_Ptr output,
                                bool use_ramulator) {
  auto config = device->config;
  hw_metric compute_peak_flops =
      config.logic_memory_bandwidth * config.logic_op_b;
  hw_metric memory_bandwidth = config.logic_memory_bandwidth;

  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  double m = input->shape[0];
  double k = input->shape[1];
  double n = weight->shape[1];

  hw_metric total_flops = 2.0 * m * k * n;
  hw_metric total_memory_size = (m * k + k * n + m * n) * weight->precision_byte;

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }
  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::LOGIC,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, weight);

    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);

    exec_status += getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, weight);
    
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

ExecStatus LinearExecutionPIM(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr output,
                              bool use_ramulator) {
  auto config = device->config;
  hw_metric compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric memory_bandwidth = config.pim_memory_bandwidth;

  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  double m = input->shape[0];
  double k = input->shape[1];
  double n = weight->shape[1];

  hw_metric total_flops = 2.0 * m * k * n;
  hw_metric total_memory_size = (m * k + k * n + m * n) * weight->precision_byte;

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }
  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::PIM,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, weight);

    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);

    exec_status += getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, weight);
    
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

ExecStatus BatchedLinearExecutionGPU(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr output,
                              bool use_ramulator, bool duplicated_input) {
  assertTrue(input->shape.size() == 3, "input tensor is not 3D tensor");
  assertTrue(weight->shape.size() == 3, "weight tensor is not 3D tensor");
  assertTrue(output->shape.size() == 3, "output tensor is not 3D tensor");

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  int num_heads = input->shape[0];
  std::vector<int> input_orig_shape = input->shape;
  std::vector<int> weight_orig_shape = weight->shape;
  std::vector<int> output_orig_shape = output->shape;

  int m = input->shape[1];
  int k = input->shape[2];
  int n = weight->shape[2];

  hw_metric total_flops = 2.0 * m * k * n * 1.0 * num_heads;
  
  hw_metric total_memory_size;
  if(duplicated_input){
    total_memory_size = 1.0 * (m * k + k * n * 1.0  * num_heads + m * n * 1.0  * num_heads) * weight->precision_byte;
  }
  else{
    total_memory_size = 1.0 * (m * k + k * n + m * n) * 1.0  * num_heads * weight->precision_byte;
  }

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    if(duplicated_input){
      input->setShape({m, k});
    }
    else{
      input->setShape({m, k * num_heads});
    }
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    
    weight->setShape({k * num_heads, n});
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, weight);

    output->setShape({m, n * num_heads});
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);

    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, weight);
    
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  exec_status.opb = total_flops / total_memory_size;

  input->setShape(input_orig_shape);
  weight->setShape(weight_orig_shape);
  output->setShape(output_orig_shape);

  return exec_status;
};

ExecStatus BatchedLinearExecutionLogic(Device_Ptr device, Tensor_Ptr input,
                                Tensor_Ptr weight, Tensor_Ptr output,
                                bool use_ramulator, bool duplicated_input) {
  auto config = device->config;
  hw_metric compute_peak_flops =
      config.logic_memory_bandwidth * config.logic_op_b;
  hw_metric memory_bandwidth = config.logic_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  double m = input->shape[0];
  double k = input->shape[1];
  double n = weight->shape[1];

  hw_metric total_flops = 2.0 * m * k * n;
  hw_metric total_memory_size = (m * k + k * n + m * n) * weight->precision_byte;

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }
  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::LOGIC,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, weight);

    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);

    exec_status += getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, weight);
    
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

ExecStatus BatchedLinearExecutionPIM(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr output,
                              bool use_ramulator, bool duplicated_input) {
  auto config = device->config;
  hw_metric compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric memory_bandwidth = config.pim_memory_bandwidth;

  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  double m = input->shape[0];
  double k = input->shape[1];
  double n = weight->shape[1];

  hw_metric total_flops = 2.0 * m * k * n;
  hw_metric total_memory_size = (m * k + k * n + m * n) * weight->precision_byte;

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }
  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::PIM,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, weight);

    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);

    exec_status += getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, weight);
    
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

}  // namespace llm_system