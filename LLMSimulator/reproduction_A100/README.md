# LLMSimulator A100-80GB Reproduction Study

This is a comprehensive reproduction study of the Duplex architecture and its optimizations, using an **NVIDIA A100 (80GB)** as the reference baseline instead of the H100.

This directory is an entirely independent experimental tree designed to safely isolate A100 artifacts from previous studies.

## 1. Project Purpose
The purpose of this study is to measure the effectiveness of Logic-PIM integration (Duplex), the Processing Element (PE) dynamic load balancer, and Expert Tensor Parallelism (ET) when constrained by the interconnects, compute capacity, and HBM bandwidth of the older A100 generation, thereby evaluating architectural resilience across GPU generations.

## 2. A100 Target Hardware
The simulator contains a native `A100` preset (`src/hardware/hardware_config.h`). 
Notable specs include:
- **Memory Capacity**: 80 GB
- **Memory Bandwidth**: 2.039 TB/s (vs H100's 3.352 TB/s)
- **Compute Peak FLOPS (FP16)**: 312 TFLOPS (vs H100's 989.4 TFLOPS)
- **Device Interconnect**: NVLink Gen 3 (150 GB/s bandwidth, 3.0 us latency)

## 3. Simulator Target Commit
- **Commit:** `419252761fbdb95b789778a02256d458a5537ec7`
- **Branch:** `main`

## 4. Workload & Model Target
To perfectly isolate the hardware shift, the workload and model remain identically constrained to the previous study:
- **Model**: Mixtral 47B (`model_name: mixtral`)
- **Workload Mode**: Decode mode (`decode_mode: on`)
- **Sequence Context**: Input length 2048 (`input_len: 2048`)
- **Measured Iterations**: 5 execution steps (`iter: 5`) following the 10,000-step cache warmup.
- **Batch Size**: 32 (`max_batch_size: 32`)

## 5. Experiment Matrix
| Configuration | Processor | PE | ET | TP | DP |
|---------------|-----------|----|----|----|----|
| **A100-GPU Baseline** | GPU | OFF | 1 | 4 | 1 |
| **A100-Duplex** | GPU+LOGIC | OFF | 1 | 4 | 1 |
| **A100-Duplex+PE** | GPU+LOGIC | ON | 1 | 4 | 1 |
| **A100-Duplex+PE+ET** | GPU+LOGIC | ON | 4 | 4 | 1 |

## 6. Output Organization
- `configs/`: Immutable execution configurations.
- `logs/`: `concise_run.log` outputs ensuring no console spam.
- `results/raw/`: Raw device logs (if intentionally generated).
- `results/csv/`: Formatted performance loop metrics (fixed for the 5th iteration delay bug).
- `results/traces/`: Standardized chronological processor and routing execution timelines.
- `plots/`: Categorized analytical visualization outputs.

## 7. Instrumentation Policy
- **Console Output**: Minimized. Raw TimeBoard trees will NOT be indiscriminately redirected into standard out, maintaining sensible log files.
- **Processor Trace**: Strict chronological tracking format (`device_id, layer_id, operation, processor, start, end, duration`).
- **Routing Trace**: Accurate 10,240-record output per simulation matching `request, token, layer, rank, expert` dimensions.

## 8. Known Issues Addressed
- **CSV Output Bug**: Fixed. The previous version dropped the final iteration metrics from the loop export due to a modulo mismatch. This has been patched directly in `src/hardware/cluster.cpp` prior to launching the A100 simulations.
- **Trace Spam**: Device trees will be written solely via controlled telemetry exports rather than blind console dumping.

## 9. Planned Phases
- **Phase A0**: Hardware Audit and Project Stand-up (Current)
- **Phase B**: Baseline A100 Simulation
- **Phase C**: Duplex Logic-PIM Simulation
- **Phase D**: PE and ET Extension
- **Phase E**: Cross-Generational Comparison (A100 vs H100)
