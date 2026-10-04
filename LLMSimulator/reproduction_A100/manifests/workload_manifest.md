# Workload Manifest

The experiment specifically targets **decode mode** (`decode_mode: on`) using synthetic data generation.

## Configuration Parameters
- **Input Length**: 2048 (Represents the pre-existing KV cache context state).
- **Maximum Generation (Output Length)**: 128 (Configured max capacity).
- **Batch Size**: 32 concurrent requests.
- **Warm-up Iterations**: 10,000 scheduler steps.
- **Measured Iterations**: 5 execution steps.

## Full Request Definition
A single full request requires processing 2048 input tokens (prefill) and generating up to 128 output tokens. However, this simulation specifically isolates the steady-state decode phase.

## Prefill / Decode Semantics
- `prefill_mode = off`
- `decode_mode = on`
This guarantees the benchmark is **decode-only**. The 2048 `input_len` represents the *existing* sequence context size loaded into the KV cache, it does NOT mean 2048 tokens are processed in the measured forward pass. Only 1 token per sequence is processed per iteration.

## Warm-up
The simulator executes `hittingQueue(10000)` before recording metrics. This forces the scheduler to process 10,000 unmeasured iterations to resolve initial KV cache loading penalties, routing anomalies, and pipeline startup transients, ensuring the measured trace records a true steady-state scenario. These tokens are entirely excluded from the performance results.

## Measured Run
The official execution traces cover 5 distinct decode iterations.
- 32 active sequences
- × 1 newly generated token / sequence / step
- = 32 tokens / step
- Total measured processed tokens = 32 × 5 = **160 newly processed tokens**.

## Routing Records
Every processed token traverses the Mixtral neural network.
- 160 tokens
- × 32 MoE layers
- × 2 top-k expert selections per layer
- = **10,240 routing records**.

This explicitly distinguishes the 160 processed tokens from the 10,240 internal network routing decisions.
