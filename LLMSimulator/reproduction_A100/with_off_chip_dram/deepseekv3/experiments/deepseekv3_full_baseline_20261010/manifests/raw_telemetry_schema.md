# Raw Telemetry Schema Documentation

**Experiment:** DeepSeek-V3 Full Baseline  
**Directory:** `raw/`  

---

## 1. `expert_routing.csv`
Logs routing decisions per token per layer:
- `request_id` (int): Sequential sequence scheduling request identifier.
- `token_id` (int): 1-indexed token sequence position.
- `layer_id` (int): 0-indexed transformer layer position.
- `topk_rank` (int): Rank of the selected expert within top-k (1 to 8).
- `expert_id` (int): Routed expert identifier (0 to 255).

---

## 2. `expert_load.csv`
Logs memory transfers for individual weight matrices and batch aggregates:
- `request_id` (int): Unique invocation identifier of `LinearExecutionGPU`.
- `layer_id` (int): 0-indexed layer identifier of the executing expert.
- `expert_id` (int): Expert identifier (0 to 255).
- `weight_tensor` (string): Matrix identifier (`W1_gate`, `W3_up`, `W2_down`, or `all_weights`).
- `size_bytes` (long): Size in bytes (29,360,128 B for single tensor, 88,080,384 B for full batch).
- `start` (double): Simulation start timestamp in nanoseconds.
- `end` (double): Simulation completion timestamp in nanoseconds.
- `duration` (double): Duration of transfer in nanoseconds (cumulative completion time from batch start for misses; 0 for hits).
- `memory_source` (string): `OFFCHIP_DRAM` for cold loads, `HBM` for cache hits.
- `memory_destination` (string): Always `HBM`.
- `cache_hit_or_miss` (string): `miss` for cold DDR5 loads, `hit` for HBM resident accesses.

---

## 3. `expert_cache.csv`
Logs all HBM cache state transitions:
- `layer_id` (int): Layer identifier of the expert.
- `expert_id` (int): Expert identifier.
- `action` (string):
  - `MISS`: Expert probe was not found in cache.
  - `INSERT`: Expert marked resident in cache following load completion.
  - `HIT`: Expert found resident in cache; promoted to MRU position.
  - `EVICT`: Expert evicted from cache due to capacity pressure.
- `resident_bytes` (long): Total bytes resident in HBM cache immediately following this event.
