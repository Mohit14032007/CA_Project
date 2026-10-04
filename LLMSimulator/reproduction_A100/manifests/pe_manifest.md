# Processing Element (PE) Manifest

## `parallel_execution = false`
When PE is disabled, the processor mapping is static based on the `processor_type` string. In `GPU+LOGIC` mode, the `TopModuleGraph` permanently assigns dense non-expert layers (Attention, LayerNorm) to the GPU, and sparse expert layers to the Logic-PIM.

## `parallel_execution = true`
When PE is enabled, a dynamic load balancer evaluates execution costs per layer. If the GPU is heavily bottlenecked (e.g. during severe expert straggling), the PE can migrate expert processing back to the GPU to alleviate the wait, effectively co-processing the MoE workload.

- **Expert Co-processing**: The PE evaluates if sending specific tokens to the GPU for processing will yield a shorter completion time than waiting for the Logic-PIM queue.
- **Attention Co-processing**: Attention inherently remains on the GPU.
- **Processor-Selection Logic**: Defined in `src/module/expert.cpp` (specifically the migration loops in `evaluateExpertLoad` or similar routing structures where `cost_diff` is calculated).
- **Routing & Outputs**: PE strictly changes *where* operations execute, not *what* executes. It does NOT change routing decisions, nor does it alter the mathematical model outputs.

Note: Do not assume PE will improve performance on the A100. It depends entirely on whether the A100 GPU compute capability is sufficient to absorb migrated MoE work without delaying its own dense layer tasks.
