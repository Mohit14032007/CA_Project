# Expert Tensor Parallelism (ET) Manifest

## `expert_tensor_degree = 1`
Experts are placed wholly on specific devices. Tokens routed to those experts must be transmitted via `moe_scatter` to the owner device, computed sequentially by that device, and returned via `moe_gather`. This minimizes synchronization overhead but creates massive compute imbalance if routing heavily prefers a single expert.

## `expert_tensor_degree = 4`
Every expert is divided into 4 shards, and one shard is placed on each device.

- **Expert Sharding**: The dense matrices of the expert FFN are split evenly.
- **Device Participation**: All 4 devices participate simultaneously in computing every routed token.
- **Partial Results & All-Reduce**: Each device computes a partial activation. These partials must be reduced via `moe_all_reduce_for_e_tp` before the final expert output is realized.
- **Communication Overhead**: The cost of `moe_all_reduce_for_e_tp` is paid at every expert layer, bounded by the NVLink bandwidth (150 GB/s on A100).
- **Source Implementation**: Defined in `src/module/expert.cpp` and `src/module/timeboard.cpp`.

Note: ET trades structural load-imbalance for constant communication overhead. The experiment explicitly measures whether the A100's slower 150 GB/s interconnect ruins this tradeoff compared to the H100.
