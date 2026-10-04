# Model Manifest: Mixtral 47B

The simulator uses the internal preset `mixtral` defined in `src/model/model_config.h`.

- **Number of layers**: 32
- **Hidden dimension**: 4096
- **Intermediate dimension**: 14336 (for dense FFN equivalence)
- **Expert intermediate dimension**: 14336
- **Attention heads**: 32
- **KV heads**: 8 (Grouped-Query Attention)
- **Routed experts**: 8
- **Shared experts**: 0
- **Top-k**: 2
- **Precision**: FP16 (2 bytes)
- **Attention Architecture**: Standard RoPE + GQA
- **Expert Architecture**: Sparse MoE (8 experts, top-2 routing)

## Parameter Definitions

**Total model parameters (47B)**:
Represents the total physical size of the model weights distributed across the devices, including all 8 experts in every layer.

**Active parameters per token (13B)**:
Only 2 out of the 8 experts are evaluated for any individual token (Top-2). The active parameters refer to the dense components (Attention, LayerNorm) plus exactly 2 experts.

**Routed experts per token**:
Each token is dynamically routed to exactly 2 experts per layer via a router network, producing load imbalances that the Duplex architecture aims to resolve.
