# Parallelism Manifest

- **Tensor Parallelism (TP)**: 4
- **Data Parallelism (DP)**: 1

## Expert Tensor Parallelism (ET = 1)
When `expert_tensor_degree = 1`, experts are statically placed on specific physical devices without being mathematically sharded. In a 4-device topology with 8 experts:
- Device 0 holds Expert 0, 1
- Device 1 holds Expert 2, 3
- Device 2 holds Expert 4, 5
- Device 3 holds Expert 6, 7
This causes severe temporal straggling if routing heavily favors Expert 0, as Device 0 must sequentially process all Expert 0 tokens while other devices sit idle waiting at the `moe_gather` barrier.

## Expert Tensor Parallelism (ET = 4)
When `expert_tensor_degree = 4`, every single expert (0 through 7) is mathematically sliced (tensor-sharded) across all 4 devices. 
- A token routed to Expert 0 will be computed collaboratively: Device 0 computes 1/4 of Expert 0, Device 1 computes 1/4, etc.
- This forces perfect spatial load balancing because no device ever sits idle regardless of the routing distribution, but it incurs a high communication penalty via `moe_all_reduce_for_e_tp` to sum the partial results.

## Definitions
- **Tensor Parallelism**: Slicing dense non-expert weights (e.g. Attention, LayerNorm) across devices.
- **Data Parallelism**: Replicating identical model weights across devices to handle larger batch sizes (unused here).
- **Expert Tensor Parallelism**: Slicing the expert weights themselves across devices.
- **Expert Placement**: Assigning whole, intact experts to specific devices (the default behavior when ET=1).
