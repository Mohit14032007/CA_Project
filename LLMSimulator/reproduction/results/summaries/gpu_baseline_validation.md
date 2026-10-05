# GPU Baseline Validation (Phase B)

**Phase B Status: PASS**

## 1. Exact Configuration Used
- **Model**: `mixtral`
- **Hardware**: 4x H100 GPU (1 Node)
- **Processor Type**: `GPU`
- **Parallel Execution (PE)**: `off`
- **Expert Tensor Degree (ET)**: `1`
- **Non-Expert Tensor Degree**: `4`

## 2. Exact Model Configuration (Mixtral 47B)
- **Model Name**: `mixtral`
- **Number of Layers**: 32
- **Hidden Dimension**: 4096
- **Intermediate Dimension**: 14336
- **Number of Attention Heads**: 32
- **Number of KV Heads**: 8 (GQA Group Size = 4)
- **Number of Routed Experts**: 8
- **Top-k**: 2
- **Weight Precision**: FP16 (precision_byte: 2)
- **Input Length**: 2048
- **Output Length**: 128
- **Max Sequence Length**: 32768
- **Batch Size**: 32

## 3. Exact Hardware/System Configuration
- **GPU Generation**: H100
- **Number of Nodes**: 1
- **Number of Devices**: 4
- **HBM Capacity**: 80 GB per device
- **NVLink**: 5th Generation (900 GB/s bidirectional bandwidth)
- **InfiniBand**: 800 (100 GB/s)

## 4. Exact Parallelism Configuration
- **none_expert_tensor_degree**: 4. This implies that non-expert layers (like Attention and regular FeedForward) are tensor parallelized across all 4 devices. This matches standard execution for a 4-device system.
- **expert_tensor_degree (ET)**: 1. This implements Expert Parallelism (EP). With 8 experts and 4 devices, each device stores and processes exactly 2 whole experts. This avoids communication during expert computation and accurately reflects a baseline GPU MoE distribution.
- **Data Parallelism**: 1 (Calculated as `Total Devices / ne_tp_dg` -> 4/4 = 1).

## 5. Exact Processor Configuration
- **processor_type**: `GPU`
- **high_processor_type**: `GPU`
- **low_processor_type**: `GPU` (Implicitly, since Logic-PIM is disabled)
- **PE Status**: `off`
- **ET Status**: `off` (Degree is 1, so no expert tensor slicing)

## 6. Devices Actually Used
Inspected the raw execution output directory. The following device log files were created and populated:
- `device_0`
- `device_1`
- `device_2`
- `device_3`

## 7. Output Files Found
- **Raw traces**: `reproduction/results/raw/gpu_baseline/device_0` .. `device_3`
- **CSV Statistics**: `reproduction/results/csv/gpu_baseline/mixtral_synthesis_2048_128_GPU_N1_D4_TP4_DP1_maxbatch32_maxprocess524288_iter5_skew0_precision_byte2_parallel_execution0_decode.csv`
- **Run Log**: `reproduction/results/gpu_baseline_run.log`

## 8. Relevant Metrics Available
- End-to-end Latency
- Queueing Delay
- Attention execution times
- Expert FFN execution times
- Communication overheads
- Energy consumption (read/write/act)
- Compute and memory utilization
- Op/B

## 9. Errors/Warnings
- None. The simulation ran correctly to completion without any Out Of Memory (OOM) errors or truncation.

## 10. Deviations from the Paper
- None identified at this stage. The GPU baseline effectively disables Logic-PIM and correctly configures a 4-device EP setup as expected for the conventional evaluation baseline.
