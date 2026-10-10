# Phase DS-2 Validation: DeepSeek-V3 FP16 Model Representation

## 1. Objective
The objective of this phase is to establish and validate the exact DeepSeek-V3 model representation instantiated by the current LLMSimulator source tree and to prove that the simulator can natively run the DeepSeek-V3 model with FP16 precision (`precision_byte = 2`) without source code modifications.

## 2. Source Files Inspected
- `src/model/model_config.h` (Model configurations and default precision)
- `eval/test.cpp` (YAML parsing and model instantiation logic)
- `src/model/llm.cpp` (Model layer structure)
- `src/module/expert.cpp` (Routed and shared expert construction, metadata tagging)
- `src/module/layer.cpp` (Feedforward block implementations)
- `src/module/tensor.cpp` / `tensor.h` (Memory sizing logic)
- `src/hardware/cluster.cpp` (Global memory footprint calculation)

## 3. Exact Model Construction Path
`eval/test.cpp` parses `model_name` -> loads `deepseekV3` struct from `model_config.h` -> overrides `precision_byte` from YAML -> calls `LLM::LLM()`.
`LLM::LLM()` explicitly branches `if (model_config.model_name == "deepseekV3")` and iterates to create `first_k_dense` standard `Decoder` blocks, then `MoEDecoder` blocks for the remainder.
Inside `MoEDecoder`, `ExpertFFN` instantiates routed experts via `FeedForward3Way` and subsequently instantiates `num_shared_expert` additional `FeedForward3Way` instances.

## 4. Exact Layer Count
- **Total Layer Count**: 60 layers (as defined by `ModelConfig deepseekV3 = ModelConfig(..., 60, ...)`).

## 5. Dense/MoE Layer Breakdown
- **Dense Layers**: 3 (layers 0, 1, 2).
- **MoE Layers**: 57 (layers 3 through 59).
- *Verified in*: `src/model/llm.cpp:49`

## 6. Routed Expert Count
- **Routed Experts**: 256 per MoE layer.

## 7. Top-k
- **Top-k Active Experts**: 8 per token.

## 8. Shared Expert Count
- **Shared Experts**: 1 per MoE layer.

## 9. Hidden Dimension
- **Hidden Dimension**: 7168

## 10. Expert Intermediate Dimension
- **Expert Intermediate Dimension**: 2048

## 11. Exact Expert Tensor Shapes
All routed and shared experts use `FeedForward3Way` composed of three tensors.
- **W1 (Gate)**: `[7168, 2048]` (Input: hidden_dim, Output: expert_intermediate_dim)
- **W3 (Up)**: `[7168, 2048]` (Input: hidden_dim, Output: expert_intermediate_dim)
- **W2 (Down)**: `[2048, 7168]` (Input: expert_intermediate_dim, Output: hidden_dim)

## 12. FP8 / Default Precision Path
In `src/model/model_config.h:107`, DeepSeek-V3 is statically hardcoded with `precision_byte = 1`.

## 13. FP16 Override Path
In `eval/test.cpp:216`, the simulator evaluates:
`model_config.precision_byte = config["simulation"]["precision_byte"].as<int>();`
This explicitly overwrites the hardcoded model precision with whatever is provided in the configuration YAML, meaning no source code edits are required to force FP16.

## 14. Whether precision_byte=2 was Actually Validated
**Yes.** We executed `ds2_fp16_model.yaml` (which set `precision_byte: 2`) using `synthesis` trace data. The simulator booted successfully and logged:
`ACT: 0.0332184GB, Weight: 1228.65GB, Cache: 0.472412GB`
This memory footprint confirms FP16 instantiation, as an FP8 model would log roughly half the weight footprint.

## 15. Exact FP16 Expert Size
For a single routed expert (W1 + W3 + W2):
- Parameters = `(7168 * 2048 * 3)` = 44,040,192 parameters.
- Bytes at FP16 = `44,040,192 * 2` = 88,080,384 bytes.

## 16. Total Routed Expert Storage
- 1 MoE Layer (256 experts) = 22,548,578,304 bytes.
- 57 MoE Layers = 1,285,268,963,328 bytes.
- **Decimal**: ~1285.27 GB (1.28 TB).
- **Binary**: exactly 1197 GiB.

## 17. Shared Expert Storage
The simulator instantiates the shared expert using `expert_intermediate_dim` (2048) rather than DeepSeek's native larger dimension.
- 1 Shared Expert = 88,080,384 bytes.
- 57 Shared Experts = 5,020,581,888 bytes.
- **Decimal**: ~5.02 GB.
- **Binary**: ~4.67 GiB.

## 18. Total Expert Storage (Routed + Shared)
- **Decimal**: 1290.29 GB
- **Binary**: 1201.67 GiB

## 19. Decimal and Binary Units
*Note: The simulator's console output uses GiB math (`/ 1024 / 1024 / 1024`) but prints the string "GB". The `1228.65 GB` logged by the simulator is technically 1228.65 GiB (which includes non-MoE dense weights and attention parameters).*

## 20. 80 GB HBM Capacity Comparison
- Total HBM Capacity = 80 GB (approx 74.5 GiB).
- `80 GB / 1290.29 GB = 0.062`
- An 80 GB HBM chip can only physically hold **~6.2%** of the DeepSeek-V3 expert weights at FP16.

## 21. Logical Residency Limitation
Because `resident_experts` is a purely logical abstraction that never evicts elements, running the B1 simulation for DeepSeek-V3 without an eviction policy will effectively give the model a 1.2+ TB infinite cache, which destroys the scientific integrity of the memory traffic analysis.

## 22. Shared-Expert Offloading Status
In `src/module/expert.cpp:79-85`, the simulator tags tensors with `is_expert_weight = true` **only inside the loop for routed experts**. The shared expert is constructed outside this loop and is never tagged.
**Limitation**: Therefore, the current off-chip DRAM logic will bypass the shared expert, treating it as standard HBM-resident model weight. For Phase 10/B1 experiments, "all expert weights initially off-chip" strictly means **routed experts only**.

## 23. Implementation Discrepancies
- The simulator utilizes `expert_intermediate_dim` (2048) for the shared expert, whereas the public DeepSeek-V3 architecture often provisions a significantly larger, unrouted standard MLP for the shared pathway.

## 24. Any Limitations
No source modification is needed to boot the model. However, scientific evaluation is strictly blocked until an LRU physical cache limit is introduced, as the logical residency will silently overflow HBM capacity and inflate performance.

## 25. Exact Next-Step Recommendation
**Phase DS-3**: Implement a physical 80 GB LRU eviction mechanism into `Device::is_expert_resident()` before generating DeepSeek-V3 DDR5 timing analyses.
