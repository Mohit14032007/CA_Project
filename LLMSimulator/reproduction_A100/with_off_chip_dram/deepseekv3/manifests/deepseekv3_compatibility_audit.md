# DeepSeek-V3 Compatibility Audit

## 1. Executive Summary
LLMSimulator already possesses partial, hardcoded configuration support for DeepSeek-V3, including the correct FP8 precision, dimensions, and expert counts. The simulator's core MoE construction path, routing trace logic, and logical off-chip residency tracking are generally compatible and can scale to DeepSeek's requirements without source modification. However, the exact architectural mix of dense and MoE layers is strictly hardcoded in `LLM::LLM()`, shared experts have parallelization constraints, and the compute model's execution cache must be carefully monitored given the new tensor shapes. No source code modifications are required for a purely synthetic B1 evaluation, but real-world routing traces and memory eviction will require further Phase DS-2 development.

## 2. DeepSeek-V3 Reference Architecture
- **Layers**: 61 total (60 primary + 1 MTP). First 3 layers are dense.
- **Dimensions**: Hidden size = 7168, Routed expert intermediate size = 2048.
- **Experts**: 256 routed experts + 1 shared expert.
- **Routing**: Top-K = 8.
- **Quantization**: FP8 expert weights.

## 3. Current LLMSimulator Model Support
- **Model Name parsing**: Evaluated in `eval/test.cpp` (`model_name == "deepseekV3"`).
- **Supported values**: Present in `src/model/model_config.h` (Line 107) as:
  `ModelConfig deepseekV3 = ModelConfig(7168, 128, 60, 128, 128, 131072, 18432, 2048, 1, 1 /*precision_byte*/, 256, 1 /*shared*/, 1, 8 /*top_k*/, 3, 3 /*first_k_dense*/, 1536, 512, 128, 64, 129280, true, true, 0.0, "deepseekV3");`
- **Configuration Class**: `ModelConfig` inside `src/model/model_config.h`.
- **Configurability**: Currently hardcoded as a static struct. Fields include `first_k_dense`, `num_routed_expert`, `num_shared_expert`, `precision_byte`, etc.

## 4. Model Construction Path
For `deepseekV3`, the model is constructed in `src/model/llm.cpp` (`LLM::LLM()` lines 42-54):
1. Loops `layer = 0` to `first_k_dense` (3): constructs `Decoder::Create`.
2. Loops `layer = first_k_dense` to `num_layers`: constructs `MoEDecoder::Create`.
This path explicitly handles the dense-first structure natively.

## 5. Expert Routing Path
In `src/scheduler/sequence.cpp` (`BatchedSequence::update_expert`):
1. Determines `expert_list` either from synthetic (Random/Zipfian) or from a loaded real sequence.
2. Iterates `top_k` (which is 8).
3. Adds expert counts and logs to `expert_routing.csv`.

## 6. Expert Tensor Construction Path
In `src/module/expert.cpp` (`ExpertFFN::ExpertFFN()`, lines 78-116):
1. Creates `FeedForward3Way` for each routed expert.
2. Extracts `w1` (gate), `w3` (up), `w2` (down) from internal `Linear` modules.
3. Tags them with `is_expert_weight = true` and cross-links them via `sister_weights = {w3, w2}`.
The shapes will inherently adapt to `model_config` (hidden_dim=7168, expert_intermediate_dim=2048).

## 7. Expert Size Calculation
Based on actual tensor representation:
- Mathematical parameters per weight: `7168 * 2048 = 14,680,064`
- Precision: `precision_byte = 1` (FP8)
- W1 Bytes: 14.68 MB
- W3 Bytes: 14.68 MB
- W2 Bytes: 14.68 MB
- **Total simulated tensor bytes per expert**: `44,040,192` (~44 MB)

## 8. Layer Structure Compatibility
**Supported**: `src/model/llm.cpp` already has a specific `if (model_config.model_name == "deepseekV3")` branch that allocates `first_k_dense` (3) standard `Decoder` blocks followed by `MoEDecoder` blocks for the rest.

## 9. Shared Expert Compatibility
**Supported**: `src/module/expert.cpp` (Line 118) allocates `FeedForward3Way` for `num_shared_expert`.
Inside `ExpertFFN::forward()` (Line 228), it explicitly computes the shared expert output and adds it to the routed expert result.
However, note that it currently executes the shared expert sequentially *after* the routed experts on the GPU.

## 10. Top-k Compatibility
**Supported**: `top_k = 8` is properly passed through `ModelConfig` and honored in `BatchedSequence::update_expert` loops.

## 11. 256-Expert Compatibility
**Supported**: `num_routed_expert = 256` allocates 256 `FeedForward3Way` submodules. The `expert_routing.csv` logging logic correctly iterates over all 256 IDs.

## 12. Precision/FP8 Compatibility
**Supported**: `precision_byte = 1` is explicitly defined in `deepseekV3`'s config. In `LinearExecutionGPU` (`src/hardware/linear_impl.cpp`, Line 28), memory size is calculated as `(m*k + k*n + m*n) * weight->precision_byte`. So LLMSimulator natively supports FP8 memory payload modeling by literally cutting the memory sizes and Ramulator payloads in half compared to FP16.

## 13. Routing-Trace Compatibility
**Supported**: The CSV includes `request_id,token_id,layer_id,topk_rank,expert_id`. The format seamlessly handles 256 experts and `topk_rank` up to 8. Note: the `deepseekV3` uses identical synthetic routing functions (Zipfian/Random) to Mixtral unless actual traces are supplied.

## 14. Off-Chip DRAM Compatibility
**Supported**: The dual-memory interface logic is agnostic to model size. It intercepts tensor metadata (`is_expert_weight`), which is correctly applied to DeepSeek experts in `src/module/expert.cpp`. Ramulator DDR5 simulation will correctly scale down its duration since the expert tensor sizes are smaller (~44MB instead of Mixtral's ~352MB).

## 15. HBM Residency Compatibility
**Supported**: `std::set<std::pair<int, int>> resident_experts` is a Red-Black tree that easily scales to 14,592 unique identifiers (57 MoE layers * 256 experts). It remains a logical "cold-start" tracker without physical constraints.

## 16. Compute-Model Compatibility
**Supported**: `LinearExecutionGPU` calculates `total_flops = 2.0 * m * k * n` and `total_memory_size = ... * 1`. The execution cache relies on sizes. Since all 256 routed experts have the same shape, the compute cache logic remains valid and will avoid repeating identical GPU math latency calculations.

## 17. Source Files Requiring Future Change
- **Physical Capacity / Eviction**: `src/hardware/device.h` must replace `std::set resident_experts` with an LRU tracking physical bytes to enforce an 80 GB limit.
- **Configurability**: `eval/test.cpp` and `model_config.h` hardcode the model instead of parsing from YAML.

## 18. What Does NOT Need to Change
- **Off-chip Memory Interface**: `run_ramulator` and `issueRamulator` natively handle the new tensor sizes.
- **Metadata Tagging**: `sister_weights` correctly identifies W1/W3/W2 regardless of dimension.
- **Trace Logging**: `processor_trace.csv` and `expert_load.csv` remain fully compatible.

## 19. Expected Total Expert Storage
- 1 Routed Expert = ~44.04 MB
- 1 Layer (256 routed experts) = 11.27 GB
- 57 MoE Layers = **~642.6 GB** of routed experts
- Shared Experts = ~396 MB per layer = **~22.5 GB** total
- **Total Expert Weights** = **~665 GB**

## 20. Expected HBM vs DRAM Relationship
- 80 GB HBM can only hold ~12% of the DeepSeek-V3 expert weights.
- Off-chip DDR5 must hold the remaining 585+ GB.
- Capacity enforcement (eviction) is therefore absolutely mandatory for realistic performance evaluation, as HBM will quickly overflow.

## 21. Risks and Scientific Validity Issues
1. **No Eviction**: Without eviction, running 665 GB of experts into an 80 GB HBM model will incorrectly assume infinite cache capacity, corrupting results after the initial warmup phase.
2. **Sequential Shared Expert**: The simulator calculates execution time sequentially. In reality, shared experts compute concurrently with routed experts. The current model may artificially inflate `T_compute`.
3. **Synthetic Routing**: DeepSeek's expert selection is highly specialized (bias, load balancing). Synthetic Zipfian routing will not produce scientifically valid correlation analysis.

## 22. Recommended Phase DS-2 Implementation Plan
1. **Eviction Implementation**: Overhaul `resident_experts` into a physically constrained 80 GB LRU cache.
2. **Real DeepSeek Traces**: Port real DeepSeek-V3 routing traces (as planned for Phase 10) instead of synthetic generation.
3. **Run Initial Baseline**: Run a B1 evaluation to measure the cold-start and steady-state eviction penalties.
