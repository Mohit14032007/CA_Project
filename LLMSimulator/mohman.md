# The Definitive Guide to LLMSimulator's Off-Chip DRAM & MoE Memory Architecture

This document is a comprehensive, first-principles learning guide covering the dual-memory infrastructure, off-chip DDR5 integration, expert residency modeling, and telemetry added to LLMSimulator for the MoE memory bottleneck experiments.

Every explanation here is based directly on the actual current C++ source code of the repository.

---

## 1. What We Are Trying to Model

### The Problem
We have a single GPU running a Mixture-of-Experts (MoE) model. Each MoE layer contains a set of "experts" (FeedForward networks). For each token, the model dynamically selects a small subset of these experts.

Expert weights are extremely large. High Bandwidth Memory (HBM) on the GPU is extremely fast but capacity-limited (e.g., 80 GB). If the total size of all experts exceeds HBM capacity, we must store the remaining expert weights in slower, off-chip system memory (DDR5) and transfer them into HBM just-in-time when they are selected for computation. 

This creates a scenario where memory movement over the interconnect (e.g., PCIe/CXL) becomes a severe bottleneck.

### The Architecture
We model the following conceptual architecture:

```
      GPU Compute Cores
             |
             +---------- HBM (Fast, Limited Capacity)
             |
             +---------- Off-chip DDR5 (Slow, High Capacity)
                             |
                   Simulated by Ramulator
```

### B0 vs B1
- **B0 (Baseline 0):** Assumes all expert weights are already available in HBM. Memory accesses go directly to HBM without any off-chip delay.
- **B1 (Baseline 1):** Expert weights are initially conceptually located in off-chip DDR5. When an expert is selected for the first time, it must be loaded from DDR5 into HBM before the GPU can execute it. 

### Capacity and Residency (Crucial Caveat)
Currently, our simulator uses a logical residency model (`std::set<std::pair<int, int>> resident_experts`).
- **32 layers × 8 experts = 256 logical experts.**
- Each expert has three weight matrices (W1, W3, W2), collectively taking up approximately 352.32 MB (FP16).
- 256 experts × 352.32 MB = **~90.19 GB.**

This exceeds an 80 GB A100. However, the current simulator *does not physically enforce this 80 GB limit*. Once an expert is loaded into HBM, it is marked as "resident" forever. There is no eviction. Thus, this simulates a "cold start" or infinite-capacity cache, rather than a physically constrained cache requiring continuous swapping.

---

## 2. What is Ramulator?

To simulate memory realistically, we cannot just hardcode "a read takes 100ns." Real DRAM involves complex timing constraints, channels, ranks, banks, rows, and columns. A request might hit an open row (fast) or require closing a row and opening a new one (slow).

**Ramulator** is a cycle-accurate DRAM simulator. We integrate it into LLMSimulator to model these exact electrical timing constraints.

### The Integration Layers

1. **LLMSimulator (Device/Linear):** Knows "I need to read Tensor X."
2. **DRAMInterface:** Bridges LLMSimulator and the memory simulator. Keeps track of simulated time.
3. **MMapController:** Maps abstract tensor sizes/locations to simulated physical memory addresses (Channels, Ranks, Banks).
4. **Ramulator (DDR5/HBM timing model):** Receives the translated physical addresses, queues the requests, simulates the JEDEC timing constraints, and calculates the exact clock cycle the data will be ready.

The beauty of this architecture is that LLMSimulator doesn't care about DRAM timing rules; it just asks `DRAMInterface` to handle a request and advance time.

---

## 3. Original LLMSimulator Memory Architecture

Before our modifications, LLMSimulator had a single, unified memory path representing the device's main memory (HBM).

Here is the original call chain for an expert weight access:

1. **`ExpertFFN::forward()`** calls **`Linear::forward()`**.
2. **`Linear::forward()`** prepares the input, weight (`A`), and output tensors, then triggers device execution.
3. **`LinearExecutionGPU()`** (in `src/hardware/linear_impl.cpp`): 
   - Requests a memory read for the input.
   - Requests a memory read for the weight tensor.
   - Requests a memory write for the output tensor.
4. **`issueRamulator()`** (in `src/hardware/layer_impl.cpp`):
   - Takes the tensor, creates a `DRAMRequest`.
   - Calls `device->run_ramulator(dram_request)`.
5. **`Device::run_ramulator()`** (in `src/hardware/device.cpp`):
   - Passes the request to the single `dram_interface`.
6. **`DRAMInterface::HandleRequest()`**:
   - Forwards to `MMapController`, which forwards to `Ramulator`.

The simulator only modeled one type of memory (HBM) and assumed everything was present there.

---

## 4. Memory Objects and Tensors

In LLMSimulator, neural network data is represented by a `Tensor`. 

Every `Tensor` has a shape (e.g., `[hidden_dim, intermediate_size]`). It calculates its byte size using its shape and precision (`precision_byte`, typically 2 bytes for FP16).

Each `Tensor` owns a `MemoryObject` (`src/dram/dram_type.h`). The `MemoryObject` is what the hardware simulator actually cares about. It holds the size of the memory block.

### Example Expert
For Llama/Mixtral styles, an expert consists of three Linear layers: `gate_proj` (W1), `up_proj` (W3), and `down_proj` (W2).
Assuming hidden dimension `4096` and intermediate dimension `14336` in FP16:
- **W1 (gate):** 4096 × 14336 × 2 = 117,440,512 bytes
- **W3 (up):** 4096 × 14336 × 2 = 117,440,512 bytes
- **W2 (down):** 14336 × 4096 × 2 = 117,440,512 bytes
- **Total:** 352,321,536 bytes (~352.32 MB).

---

## 5. Original Expert Execution Walkthrough

If Layer 1 selects Expert 1:
1. `ExpertFFN` routes the token to its internal feed-forward network.
2. The network runs the `gate_proj` (Linear).
3. `LinearExecutionGPU` requests a read for the `gate_proj` weight tensor.
4. `issueRamulator` packages the `Tensor` into a `DRAMRequest`.
5. `Device` forwards this to `dram_interface` (HBM).
6. The exact same process repeats for `up_proj` and `down_proj`.
7. All accesses incur standard HBM latency.

---

## 6. Why the Original System Was Not Enough

To model our B1 experiment (off-chip loading), the simulator needed to know:
1. Which tensors are expert weights vs. normal activations?
2. Are these expert weights currently in HBM or off-chip DDR5?
3. If they are off-chip, we must subject their memory requests to the slower DDR5 timing model, not the HBM timing model.
4. If one weight matrix of an expert is loaded, the entire expert should be loaded together.

This required introducing a dual-memory infrastructure.

---

## 7. Complete Change Log

Here is a detailed breakdown of every source file modified to implement the dual-memory infrastructure.

### `src/hardware/base.h`
**Change:**
```cpp
enum class MemoryTarget { HBM, OFFCHIP_DRAM };
using CacheKey = std::tuple<LayerType, ProcessorType, DRAMRequestType, long, MemoryTarget>;
```
**Why:** We introduced `MemoryTarget` to explicitly tag memory requests. We expanded the `CacheKey` so that the execution cache doesn't accidentally reuse an HBM cache hit for an OFFCHIP_DRAM request of the same size.

### `src/hardware/hardware_config.h`
**Change:** Added `std::string offchip_dram_cfg_path = "";` to `SystemConfig`.
**Why:** Allows configuring the second memory tier from YAML (e.g., pointing it to DDR5).

### `src/hardware/device.h` & `src/hardware/device.cpp`
**Change:** 
Added `offchip_dram_interface` and `offchip_mmap_controller`.
Added residency tracking:
```cpp
std::set<std::pair<int, int>> resident_experts;
bool is_expert_resident(int layer_id, int expert_id);
void mark_expert_resident(int layer_id, int expert_id);
```
Updated `run_ramulator` to route based on `MemoryTarget`:
```cpp
void Device::run_ramulator(DRAMRequest_Ptr dram_request, MemoryTarget target) {
  std::list<DRAMRequest::Ptr> request;
  request.push_back(dram_request);
  if (target == MemoryTarget::OFFCHIP_DRAM && offchip_dram_interface != nullptr) {
    offchip_dram_interface->HandleRequest(request, 0);
  } else {
    dram_interface->HandleRequest(request, 0);
  }
}
```
**Why:** The `Device` is the central hardware coordinator. It now initializes and owns two separate Ramulator instances. It dynamically routes `DRAMRequest` objects to either the fast HBM instance or the slow DDR5 instance based on the `MemoryTarget`.

### `src/module/tensor.h`
**Change:** Added metadata fields:
```cpp
bool is_expert_weight = false;
int expert_id = -1;
int expert_layer_id = -1;
std::vector<Tensor_Ptr> sister_weights;
MemoryTarget weight_target = MemoryTarget::HBM;
bool load_sisters = false;
```
**Why:** Tensors are passed down to the hardware level. To make hardware-level routing decisions, the `Tensor` must carry its own metadata about whether it is an expert, which expert it belongs to, and where it resides.

### `src/module/expert.cpp`
**Change:** In `ExpertFFN::ExpertFFN()`, after building the expert sub-modules, it traverses down into `gate_proj`, `up_proj`, and `down_proj` to extract their "A" tensors. It tags them with `is_expert_weight = true` and links them via the `sister_weights` vector.
**Why:** We must identify expert weights at model construction time. Linking sister weights allows us to trigger the loading of the entire expert (W1, W2, W3) simultaneously when any single matrix is accessed.

### `src/module/linear.cpp`
**Change:** In `Linear::forward()`, we added dynamic residency checking:
```cpp
A->weight_target = MemoryTarget::HBM;
A->load_sisters = false;
if (A->is_expert_weight) {
  int layer_id = sequences_metadata->cur_layer;
  if (!device->config.offchip_dram_cfg_path.empty()) {
    if (!device->is_expert_resident(layer_id, A->expert_id)) {
      A->weight_target = MemoryTarget::OFFCHIP_DRAM;
      A->load_sisters = true;
      device->mark_expert_resident(layer_id, A->expert_id);
    }
  }
}
```
**Why:** This logic executes *during the forward pass simulation*. It checks the `Device`'s residency cache. If the expert is missing, it tags the tensor for `OFFCHIP_DRAM` and marks the expert as resident for all future accesses.

### `src/hardware/linear_impl.cpp`
**Change:** `LinearExecutionGPU` now respects the `MemoryTarget` and issues sister requests:
```cpp
ExecStatus weight_status = issueRamulator(..., weight, weight->weight_target);
if (weight->load_sisters) {
  for (auto sister : weight->sister_weights) {
    ExecStatus sister_status = issueRamulator(..., sister, MemoryTarget::OFFCHIP_DRAM);
    // accumulate latency...
  }
  weight->load_sisters = false;
}
```
It also adds a massive block of telemetry code to write to `expert_load.csv`.
**Why:** Actually enacts the off-chip delay by querying Ramulator. Also records exactly how long the off-chip load took, separating `T_load` from standard computation latency.

---

## 8. Dual Memory Infrastructure

By instantiating two `DRAMInterface` objects in `Device::Device()`, we literally create two isolated memory worlds.

```
Device
 ├── dram_interface (HBM)
 │      └── MMapController (Maps to HBM geometry)
 │            └── Ramulator (HBM3 Timing)
 │
 └── offchip_dram_interface (DDR5)
        └── MMapController (Maps to DDR5 geometry)
              └── Ramulator (DDR5 Timing)
```

Both use the same `MemoryConfig` layout for address generation, but they point to different Ramulator backend config files. `issueRamulator` calls `device->run_ramulator()`, passing the `MemoryTarget` flag to decide which pointer to follow.

---

## 9. DDR5 Configuration

The off-chip interface relies on a configuration file (e.g., `dram_config_DDR5.yaml`):

```yaml
Ramulator:
  Memory:
    Standard: DDR5
    Speed: DDR5_4800B
    Channels: 2
    Ranks: 2
```

When `DRAMInterface::Create` parses this, Ramulator loads standard JEDEC DDR5-4800 timings (tCAS, tRCD, tRP, etc.).
This configuration defines the *physical constraints* of the simulated off-chip memory, which is drastically slower and has fewer channels than the HBM configuration.

---

## 10. Memory Target Abstraction

We introduced:
```cpp
enum class MemoryTarget { HBM, OFFCHIP_DRAM };
```
Instead of hardcoding `if (is_offchip)` everywhere, we tag the `Tensor` and pass `MemoryTarget` all the way down the stack. `issueRamulator` accepts `MemoryTarget`, overriding the default `HBM`. This keeps the API clean and extensible. If we wanted to add a `MemoryTarget::CXL_SSD`, we just add it to the enum and instantiate a third interface in `Device`.

---

## 11. Expert Residency

The residency tracking is entirely logical.
Located in `src/hardware/device.h`:
```cpp
std::set<std::pair<int, int>> resident_experts;
```
It stores `(layer_id, expert_id)`.

When an expert is first encountered in `Linear::forward()`, the `resident_experts` set is checked. Since it is empty, the check fails. The system marks the tensor for `OFFCHIP_DRAM` and immediately inserts `(layer_id, expert_id)` into the set.

On the next iteration, when the same expert is routed to, `resident_experts.find()` succeeds. The `weight_target` remains `HBM`. The simulation entirely skips the DDR5 penalty.

---

## 12. Expert Weight Metadata

We couldn't put metadata on the wrapper `ExpertFFN` or `Linear` classes because `LinearExecutionGPU` (which talks to Ramulator) only sees raw `Tensor` objects. 

In `src/module/expert.cpp`, we extract the innermost `A` tensors (the actual weight matrices):
```cpp
if (gate && gate->get_module("Linear")) w1 = gate->get_module("Linear")->get_tensor("A");
```
We tag `A` with `is_expert_weight = true` and `expert_id`. This allows the lowest levels of hardware simulation to recognize exactly what piece of data is flowing through the system.

---

## 13. Why W1/W2/W3 Matter

An MoE FeedForward block executes:
`Output = Down_Proj(Activation(Gate_Proj(X)) * Up_Proj(X))`

If we only loaded `Gate_Proj` (W1) from DDR5, then later hit `Up_Proj` (W3) and loaded it separately, we would underestimate memory pressure or simulate fragmented loading.

In reality, the entire expert (all three matrices) is loaded together. We implemented `sister_weights`. When W1 is accessed and triggers a miss, it sets `load_sisters = true`. `LinearExecutionGPU` sees this, issues a memory request for W1, and *immediately iterates through the sister weights (W2, W3) and issues requests for them too*. This accurately simulates the massive burst of DDR5 traffic required to fetch the full ~352 MB footprint.

---

## 14. OFF-CHIP MISS Walkthrough

**Scenario: Layer 1, Expert 1. Expert is NOT resident.**

1. `ExpertFFN` receives the token.
2. It calls `gate_proj` (Linear).
3. `Linear::forward()` executes.
4. It checks `device->is_expert_resident(1, 1)`. Returns `false`.
5. It sets `A->weight_target = MemoryTarget::OFFCHIP_DRAM`.
6. It sets `A->load_sisters = true`.
7. It calls `device->mark_expert_resident(1, 1)`.
8. `LinearExecutionGPU()` is invoked.
9. It calls `issueRamulator(..., weight, target=OFFCHIP_DRAM)`.
10. `issueRamulator` calls `device->run_ramulator(req, OFFCHIP_DRAM)`.
11. `Device` routes the request to `offchip_dram_interface`.
12. Ramulator simulates reading 117MB from DDR5. Returns long latency.
13. `LinearExecutionGPU` sees `load_sisters == true`.
14. It loops over W2 and W3, issuing requests to `OFFCHIP_DRAM`.
15. Ramulator simulates reading the remaining 234MB.
16. The total latency is aggregated as `weight_memory_cost`.
17. The telemetry logger writes a "miss" entry to `expert_load.csv`.
18. The GPU compute proceeds.

---

## 15. OFF-CHIP HIT Walkthrough

**Scenario: Layer 1, Expert 1. Expert IS resident (Iter 1+).**

1. `ExpertFFN` calls `gate_proj` (Linear).
2. `Linear::forward()` executes.
3. It checks `device->is_expert_resident(1, 1)`. Returns `true`.
4. It does NOTHING. `A->weight_target` remains `MemoryTarget::HBM`.
5. `A->load_sisters` remains `false`.
6. `LinearExecutionGPU()` is invoked.
7. It calls `issueRamulator(..., weight, target=HBM)`.
8. `Device` routes the request to the standard `dram_interface`.
9. Ramulator simulates reading from fast HBM.
10. The telemetry logger writes a "hit" entry to `expert_load.csv` with `0` off-chip latency cost.
11. The GPU compute proceeds.

---

## 16. How the Request Actually Reaches Ramulator

```
LinearExecutionGPU (src/hardware/linear_impl.cpp)
        | (Passes Tensor, MemoryTarget)
        v
issueRamulator() (src/hardware/layer_impl.cpp)
        | (Wraps Tensor into DRAMRequest)
        v
Device::run_ramulator() (src/hardware/device.cpp)
        | (Checks MemoryTarget, picks DRAMInterface_Ptr)
        v
DRAMInterface::HandleRequest() (src/hardware/device.cpp)
        | (Generates physical memory addresses from size)
        v
MMapController
        |
        v
Ramulator (DDR5 timing model)
```

---

## 17. Ramulator Request Lifecycle

1. `issueRamulator` creates a `DRAMRequest`.
2. It sizes it based on `Tensor::getSize()`.
3. `DRAMInterface::HandleRequest` breaks this massive request down into standard 64-byte or 256-byte cache line reads.
4. `MMapController` translates these chunk accesses into physical addresses (Channel X, Rank Y, Bank Z).
5. Ramulator inserts these addresses into its internal queues.
6. Based on DDR5 rules (e.g., tRCD, tCAS), Ramulator determines when each piece of data is electrically available.
7. `DRAMInterface` advances its internal simulated clock (`tick()`) until Ramulator finishes serving all chunks.
8. The final simulated clock time becomes the memory latency.
9. This latency is accumulated in `ExecStatus.memory_duration`.
10. `LinearExecutionGPU` adds this memory duration to the final execution time.

---

## 18. HBM vs DDR5 in this Simulator

| Feature | HBM (Baseline) | DDR5 (Off-chip) |
|---|---|---|
| **Location** | On GPU | Host System Motherboard |
| **Simulated Speed** | High Bandwidth, Low Latency | Lower Bandwidth, Higher Latency |
| **Object** | `dram_interface` | `offchip_dram_interface` |
| **Config** | `dram_config_HBM3_80GB.yaml` | `dram_config_DDR5.yaml` |
| **B0 Access** | All experts | None |
| **B1 Access** | Compute, Hit experts | Miss experts |

---

## 19. B0 vs B1 Code Path

The difference between Baseline 0 (B0) and Baseline 1 (B1) is purely driven by configuration.

In `b0_full.yaml`:
```yaml
system:
  optimization:
    offchip_dram_cfg_path: ""
```
Because the path is empty, `Linear::forward` skips residency checks entirely. `weight_target` always remains `HBM`.

In `b1_full.yaml`:
```yaml
system:
  optimization:
    offchip_dram_cfg_path: "./dram_config_DDR5.yaml"
```
Because the path is valid, `Device` instantiates the second Ramulator instance. `Linear::forward` begins checking residency and routing misses to `OFFCHIP_DRAM`.

---

## 20. Actual Timing / Telemetry

We implemented highly granular telemetry:

### `expert_load.csv`
Generated in `LinearExecutionGPU`. Captures:
`request_id, layer_id, expert_id, weight_tensor, size_bytes, start, end, duration, memory_source, memory_destination, cache_hit_or_miss`
**`T_load`** is extracted from the `duration` column of cold misses.

### `processor_trace.csv`
Generated via `writeProcessorTrace` in `src/module/timeboard.cpp`. Records every node in the execution graph, mapping operations to GPU/PIM, allowing us to compute **`T_compute`** (how long the GPU actually took to multiply the matrices).

By separating these, we discovered that `T_load ≈ 981µs` and `T_compute ≈ 137µs`, producing a bottleneck ratio of ~7.16x. Note that this ratio compares sequential memory fetching vs compute; it does not automatically represent exact end-to-end slowdown due to potential overlapping.

---

## 21. Important Implementation Caveats

If you are expanding this research, be intimately aware of these limitations in the current code:

1. **Logical vs Physical Residency:** The `resident_experts` set grows indefinitely. It does not enforce the 80 GB limit. The experiment tests *cold loading latency*, not *capacity constraints*.
2. **Missing Eviction:** Because capacity is infinite, there is no LRU or LFU eviction policy implemented.
3. **Bandwidth Scaling Artifact:** LLMSimulator uses an internal `memory_scale_factor` to abstract complex memory accesses. This means the DDR5 bandwidth might appear artificially inflated (~359 GB/s) in trace logs. Do not confuse simulated effective throughput with literal hardware DDR5 throughput.
4. **Layer 0 DAG Artifact:** Due to how the simulator constructs its DAG, Layer 0 experts are queried during initialization. This pre-warms the residency cache, resulting in only 248 cold misses (from layers 1-31) instead of the expected 256.

---

## 22. Complete Source-Code Map

| Component | File | Class/Function | Responsibility |
|---|---|---|---|
| **Device Mgr** | `device.cpp` | `Device::Device` | Instantiates HBM and DDR5 Ramulator instances. |
| **Residency Tracker** | `device.h` | `resident_experts` | Stores `(layer, expert)` pairs for residency checks. |
| **Metadata Tagging** | `expert.cpp` | `ExpertFFN::ExpertFFN` | Annotates `A` tensors as expert weights and links sisters. |
| **Runtime Router** | `linear.cpp` | `Linear::forward` | Checks residency; sets `MemoryTarget::OFFCHIP_DRAM` on miss. |
| **Execution/Telemetry**| `linear_impl.cpp`| `LinearExecutionGPU` | Executes Ramulator requests, triggers sisters, logs `T_load`. |
| **DRAM Gateway** | `layer_impl.cpp` | `issueRamulator` | Generates `DRAMRequest` and routes based on `MemoryTarget`. |
| **Config Loader** | `hardware_config.h`| `SystemConfig` | Holds `offchip_dram_cfg_path`. |
| **Compute Profiler** | `timeboard.cpp` | `writeProcessorTrace` | Recursively walks the execution graph to log `T_compute`. |

---

## 23. "Follow This in the Debugger"

If you want to trace an off-chip miss yourself, set these breakpoints:

1. **Breakpoint 1: `src/module/linear.cpp` inside `Linear::forward`**
   - Condition: `A->is_expert_weight == true`
   - Inspect: `sequences_metadata->cur_layer` and `A->expert_id`.
   - Step over `device->is_expert_resident`. Watch `weight_target` change to `MemoryTarget::OFFCHIP_DRAM`.
2. **Breakpoint 2: `src/hardware/linear_impl.cpp` at `LinearExecutionGPU`**
   - Inspect: `weight->weight_target`. It should be 1 (`OFFCHIP_DRAM`).
   - Step into `issueRamulator(..., weight, weight->weight_target)`.
3. **Breakpoint 3: `src/hardware/device.cpp` at `Device::run_ramulator`**
   - Inspect: `target`. See the code branch into `offchip_dram_interface->HandleRequest()`.
4. **Breakpoint 4: Back in `LinearExecutionGPU`**
   - Watch the `load_sisters` loop fire, issuing two more requests to OFFCHIP_DRAM for the remaining weight matrices.
   - Watch the telemetry block write the CSV row.

---

## 24. "If I Wanted to Modify This..."

- **Add an Eviction Policy:** Modify `src/hardware/device.h`. Replace `std::set resident_experts` with an LRU queue that enforces a strict byte limit. When full, pop the oldest expert.
- **Change DDR5 Timing:** Edit `dram_config_DDR5.yaml`. No C++ changes required.
- **Add a 3rd Tier (CXL SSD):** Add `CXL_SSD` to `MemoryTarget` in `base.h`. Instantiate a third interface in `Device::Device`. Modify `Linear::forward` to handle a multi-level cache lookup.
- **Implement Prefetching:** You would intercept the token sequence in `src/scheduler/sequence.cpp` (where routing probabilities are known), calculate upcoming experts, and manually call `issueRamulator` ahead of time to overlap memory latency with compute.

---

## 25. Complete End-to-End Example

**Scenario: Token reaches Layer 1, Expert 1 (FP16, 352MB total footprint).**

1. **Router:** Assigns token to Expert 1.
2. **ExpertFFN:** Token enters FeedForward network.
3. **Weight Tensor:** `gate_proj` (W1) "A" tensor carries metadata (`is_expert_weight=true`, `expert_id=1`, `sister_weights={W2, W3}`).
4. **Residency Check:** `Linear::forward` calls `is_expert_resident(1, 1)`. 
5. **MISS:** Cache is cold. Returns false.
6. **OFFCHIP_DRAM:** Tensor is tagged `weight_target = OFFCHIP_DRAM`, `load_sisters = true`. Marked resident for the future.
7. **DRAMRequest:** `LinearExecutionGPU` asks `issueRamulator` to process W1.
8. **Device:** Routes W1 to `offchip_dram_interface`.
9. **Ramulator DDR5:** Simulates memory delay.
10. **Sisters Loaded:** `LinearExecutionGPU` sees `load_sisters=true` and immediately sends W2 and W3 to DDR5.
11. **Telemetry:** Writes "miss" row with total latency to `expert_load.csv`.
12. **GPU Execution:** Computes matrix multiplication. Writes to `processor_trace.csv`.

**Next Iteration:**
Token reaches Layer 1, Expert 1.
1. `Linear::forward` checks `is_expert_resident(1, 1)`.
2. **HIT:** Returns true.
3. **HBM:** Tensor retains default `MemoryTarget::HBM`.
4. **GPU Execution:** Runs instantly out of HBM without triggering Ramulator DDR5.

---

## 26. Final Mental Model

**LLMSimulator** computes neural networks. It passes **Tensors** to the **Device**. 
To execute operations, the **Device** generates **DRAMRequests**. 
These requests flow through the **DRAMInterface** and **MMapController** into **Ramulator**, which simulates the real physical latency of memory hardware.

Our modifications split this pipeline. We added a **second memory interface** pointing to a DDR5 Ramulator instance. We injected **metadata into the expert weight tensors** and added a **logical residency tracker**. 

During simulation, if an expert tensor is accessed and is not resident, it is routed to the DDR5 simulator penalty box before being marked resident in HBM. This architecture allows us to meticulously capture the exact latency cost (**T_load**) of shuffling large MoE weights across slow interconnects while cleanly separating it from raw GPU compute time (**T_compute**).
