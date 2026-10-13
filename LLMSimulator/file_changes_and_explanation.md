# Source Code Changes and Comprehensive Architectural Explanation

> **Document Classification:** Technical Architecture & Code Implementation Guide  
> **Target File:** `reproduction_A100/with_off_chip_dram/file_changes_and_explanation.md`  
> **Source Reference:** *Group 13 — Computer Architecture Mid Evaluation Report: The MoE Memory Wall — Predicting & Prefetching Experts for Faster LLM Inference*  
> **Target System:** LLMSimulator with Dual-Memory Subsystem (HBM3 + Off-Chip DDR5 via Ramulator 2)

---

## 1. Executive Summary & Problem Motivation

In modern Mixture-of-Experts (MoE) architectures such as **DeepSeek-V3** (60 layers: 3 dense, 57 MoE; 256 experts per layer, Top-8 routing per token), the aggregate parameter volume drastically exceeds on-package GPU memory (HBM):
- Each expert contains three projection matrices ($W_1$ Gate, $W_2$ Down, $W_3$ Up) of dimension $7168 \times 2048$.
- In 16-bit precision (FP16/BF16), each expert occupies:
  $$\text{Size}_{\text{expert}} = 3 \times 7168 \times 2048 \times 2\text{ bytes} = 88,080,384\text{ bytes} \approx 88\text{ MB}$$
- Across 57 MoE layers with 256 experts each, the total expert weight footprint is:
  $$\text{Total Footprint} = 57 \times 256 \times 88\text{ MB} = 14,592\text{ experts} \times 88\text{ MB} \approx 1.285\text{ TB}$$

Because an NVIDIA A100 GPU package provides only **80 GB of HBM**, it is physically impossible to hold 1.285 TB of expert weights locally. To overcome this **MoE Memory Wall**, the simulator was re-architected to introduce a disaggregated, two-tier memory hierarchy:
1. **Tier 1 (Local HBM3):** 80 GB fast on-chip memory managed as an LRU cache (holding up to ~908 experts).
2. **Tier 2 (Secondary Off-Chip DDR5 DRAM):** 2.75 TB backing store backed by Ramulator 2, storing all 14,592 expert weights.

When a router selects an expert absent from HBM, its weights must be fetched across the memory bus from off-chip DDR5 before compute can begin. The following sections quote every source code modification documented in the evaluation report, explain the architectural **WHY** behind each, and conclude with the complete end-to-end execution flow.

---

## 2. Quoted Source Code Changes & In-Depth Technical "WHY"

---

### Part 1: Memory Target Classification Enum
**File:** [`src/hardware/base.h`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/base.h)

#### Quoted Source Code
```cpp
//src/hardware/base.h
enum class MemoryTarget {
    HBM,
    OFFCHIP_DRAM
};
```

#### Why This Change Was Made
* **Problem Solved:** Previously, the simulator had only a single memory abstraction (`dram_interface`), assuming all weights and activations resided uniformly within GPU HBM. There was no mechanism to distinguish between an on-chip HBM access and an off-chip DDR5 bus transaction.
* **Architectural Purpose:** `MemoryTarget` introduces an explicit routing discriminator. Every tensor memory request and Ramulator transaction can now be tagged as either `HBM` or `OFFCHIP_DRAM`. This enables the hardware scheduler, memory mappers, and timing boards to dispatch requests to the appropriate memory controller and apply distinct latency models.
* **Failure Mode Without It:** Without this enum, any memory transaction would default to HBM, making it impossible to simulate the transfer latency, bus congestion, and energy costs of fetching cold experts from secondary storage.

---

### Part 2: Scaled Off-Chip DDR5 Geometry Configuration
**File:** [`src/dram/memory_config.h`](file:///home/mohman/ca_project/LLMSimulator/src/dram/memory_config.h)

#### Quoted Source Code
```cpp
//src/dram/memory_config.h
static MemoryConfig ddr5_32gb_x16_2ch = MemoryConfig(
    1,          // num_cube
    1,          // num_logic_cube
    2,          // num_channel
    2,          // num_rank
    4,          // num_bankgroup
    4,          // num_bank
    10485760,   // num_row (2^17)*80 -> increased to simulate enough capacity for MoE
    64,         // num_col (2^10 / 16)
    64,         // granul (bytes per request)
    MemoryConfig::MemoryType::DDR5 // mem_type
);
```

#### Why This Change Was Made
* **Problem Solved:** Standard Ramulator 2 DDR5 presets model 32 GB or 64 GB physical DIMMs. If 1.285 TB of expert weights were mapped into a standard 32 GB address space, physical row/rank address overflows would occur immediately during linear address calculation in `MMapController`.
* **Architectural Purpose:** The row count was systematically scaled to $10,485,760$ rows ($2^{17} \times 80$). Calculating total capacity:
  $$\text{Capacity} = 2\text{ channels} \times 2\text{ ranks} \times 4\text{ bankgroups} \times 4\text{ banks} \times 10,485,760\text{ rows} \times 64\text{ cols} \times 64\text{ bytes} \approx 2.75\text{ TB}$$
  This provides 2.75 TB of physical DDR5 address space, allowing all 14,592 DeepSeek-V3 experts ($1.285\text{ TB}$) to fit cleanly within 46.76% of DDR5 rows without aliasing or overflow.
* **Failure Mode Without It:** The simulation would crash with out-of-bounds address mapping assertion failures in Ramulator 2 when attempting to register expert tensors exceeding 32 GB.

---

### Part 3: Secondary DRAM Interface & Controller Initialization
**File:** [`src/hardware/device.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp)

#### Quoted Source Code
```cpp
//src/hardware/device.cpp
std::string offchip_cfg = config.offchip_dram_cfg_path;
if (offchip_cfg.empty()) {
    offchip_cfg = dram_cfg_path;
}

//Memory timing scale and memory organization initialization
double offchip_scale_factor = memory_scale_factor;
MemoryConfig offchip_memory_config = memory_config;
if (offchip_cfg.find("DDR5") != std::string::npos) {
    offchip_scale_factor = 0.625;
    offchip_memory_config = ddr5_32gb_x16_2ch;
}

//Second Interface and address mapper construction
offchip_dram_interface =
    DRAMInterface::Create(
        offchip_cfg,
        offchip_scale_factor
    );
offchip_mmap_controller =
    MMapController::Create(
        offchip_memory_config
    );
```

#### Why This Change Was Made
* **Problem Solved:** An A100 GPU Device instance historically owned only a single `dram_interface` pointer tied to HBM timing. It could not interact with an external memory system operating at a different clock frequency or protocol.
* **Architectural Purpose:** 
  1. Instantiates a dedicated `offchip_dram_interface` backed by Ramulator 2 running `dram_config_DDR5_retry.yaml`.
  2. Sets `offchip_scale_factor = 0.625`, precisely matching the clock cycle ratio between DDR5-3200 ($1.6\text{ GHz} \Rightarrow 0.625\text{ ns}$ cycle) and GPU simulation base time units.
  3. Creates `offchip_mmap_controller` using the expanded `ddr5_32gb_x16_2ch` geometry so logical byte addresses translate into valid DDR5 (channel, rank, bankgroup, bank, row, column) tuples.
* **Failure Mode Without It:** Secondary DRAM could not be simulated; all accesses would either fail with null pointer dereferences or incorrectly execute with HBM timing characteristics (2 TB/s bandwidth instead of DDR5's ~50 GB/s bandwidth).

---

### Part 4: LRU Cache State Management & Eviction
**Files:** [`src/hardware/device.h`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.h), [`src/hardware/device.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp)

#### Quoted Source Code (Cache State)
```cpp
//Cache State
std::list<std::tuple<int, int, long long>> lru_expert_list;
std::unordered_map<
    std::pair<int, int>,
    std::list<std::tuple<int, int, long long>>::iterator,
    PairHash> resident_experts;

long long cache_capacity_bytes =
    std::getenv("DEBUG_CACHE_CAPACITY")
    ? std::stoll(std::getenv("DEBUG_CACHE_CAPACITY"))
    : 80000000000LL;

long long resident_bytes = 0;

//Data Structure tracks
//Expert ID, Expert Size, Current Residency, LRU Order, Total Resident Bytes,
//Configured Cache capacity
```

#### Quoted Source Code (Eviction Logic)
```cpp
//New Expert Addition Check
while (resident_bytes + expert_size > cache_capacity_bytes &&
       !lru_expert_list.empty()) {
    auto lru_expert = lru_expert_list.front();
    lru_expert_list.pop_front();
    std::pair<int, int> evict_key = {
        std::get<0>(lru_expert),
        std::get<1>(lru_expert)
    };
    resident_experts.erase(evict_key);
    resident_bytes -= std::get<2>(lru_expert);
}
```

#### Why This Change Was Made
* **Problem Solved:** Without capacity tracking and replacement, HBM would either assume infinite capacity (storing all 1.285 TB and hiding cold misses) or remain completely empty.
* **Architectural Purpose:**
  1. `resident_experts` hash map indexed by composite key `(layer_id, expert_id)` delivers $O(1)$ residency lookup time.
  2. `lru_expert_list` doubly linked list maintains strict temporal access ordering: the front is the Least Recently Used (LRU) candidate, and the back is Most Recently Used (MRU).
  3. `cache_capacity_bytes` enforces the hardware 80 GB limit ($80,000,000,000\text{ bytes}$).
  4. The `while` loop proactively evicts LRU experts from HBM until enough contiguous capacity is available to host the incoming $88\text{ MB}$ expert.
* **Failure Mode Without It:** The memory budget would be violated, rendering cache residency studies and prefetching simulations scientifically invalid.

---

### Part 5: Expert Residency Check, Cache Hit Path & Cache Trace
**Files:** [`src/hardware/device.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp), [`src/module/linear.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/module/linear.cpp)

#### Quoted Source Code (Check Residency & Logging Setup)
```cpp
//src/hardware/device.cpp
bool Device::is_expert_resident(int layer_id, int expert_id) {
    static bool init = false;
    static std::ofstream csv;
    if (!init) {
        const char* trace_dir_env = std::getenv("TRACE_DIR");
        std::string path = "expert_cache.csv";
        if (trace_dir_env)
            path = std::string(trace_dir_env) + "/expert_cache.csv";
        csv.open(path, std::ios::app);
        csv.seekp(0, std::ios::end);
        if (csv.tellp() == 0) {
            csv << "layer_id,expert_id,action,resident_bytes\n";
        }
        init = true;
    }
```

#### Quoted Source Code (Cache Hit Handling)
```cpp
    if (it != resident_experts.end()) {
        // Cache HIT: Move to MRU position (back of the list)
        auto node = *(it->second);
        lru_expert_list.erase(it->second);
        lru_expert_list.push_back(node);
        it->second = std::prev(lru_expert_list.end());
        csv << layer_id << "," << expert_id
            << ",HIT," << resident_bytes << "\n";
        csv.flush();
        return true;
    }
```

#### Quoted Source Code (Linear Forward Residency Interception)
```cpp
//src/module/linear.cpp
A->weight_target = MemoryTarget::HBM;
A->load_sisters = false;

if (A->is_expert_weight && input->getSize() > 0) {
    int layer_id = sequences_metadata->cur_layer;
    A->expert_layer_id = layer_id;
    if (!device->config.offchip_dram_cfg_path.empty()) {
        if (!device->is_expert_resident(layer_id, A->expert_id)) {
            A->weight_target = MemoryTarget::OFFCHIP_DRAM;
            A->load_sisters = true;
        }
    }
}
```

#### Why This Change Was Made
* **Problem Solved:** When `Linear::forward()` executes, it must dynamically decide whether the current expert weight tensor needs to be transferred across the DDR5 bus or can be accessed directly in on-package HBM.
* **Architectural Purpose:**
  1. If `is_expert_resident()` returns `true` (Cache Hit), the node in `lru_expert_list` is spliced to the back (MRU position) in $O(1)$ time, refreshing its lifespan. A `HIT` event is appended to `expert_cache.csv`.
  2. If `false` (Cache Miss), `Linear::forward()` intercepts the request:
     - Sets `weight_target = MemoryTarget::OFFCHIP_DRAM`.
     - Flags `load_sisters = true`, instructing the hardware engine to fetch all three projection matrices ($W_1, W_2, W_3$) together.
  3. Uses `sequences_metadata->cur_layer` to assign the proper layer context, ensuring experts with identical IDs in different layers do not collide.
* **Failure Mode Without It:** The execution engine would fetch matrices independently on three separate sequential misses, tripling the control overhead and serialization delay.

---

### Part 6: Sister Weight Batching & Request Aggregation
**File:** [`src/hardware/linear_impl.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/linear_impl.cpp)

#### Quoted Source Code
```cpp
//src/hardware/linear_impl.cpp
std::vector<Tensor_Ptr> batch_tensors;
batch_tensors.push_back(weight);

if (weight->load_sisters) {
    for (auto sister : weight->sister_weights) {
        sister->expert_layer_id = weight->expert_layer_id;
        batch_tensors.push_back(sister);
    }
    weight->load_sisters = false;
}

ExecStatus weight_status =
    issueRamulatorBatch(
        device,
        LayerType::LINEAR,
        ProcessorType::GPU,
        DRAMRequestType::kRead,
        PIMOperandType::kDRAM,
        batch_tensors,
        weight->weight_target);
```

#### Why This Change Was Made
* **Problem Solved:** An expert FFN comprises Gate ($W_1$), Up ($W_3$), and Down ($W_2$) projection weights. Issuing three independent DRAM requests forces three separate memory round trips, queue stalls, and status polling operations.
* **Architectural Purpose:**
  1. Sister weight linking (`sister_weights`) groups all three projection tensors into a single unified batch vector (`batch_tensors`) totaling 88 MB.
  2. `issueRamulatorBatch` submits the entire 88 MB package in one single burst to DDR5 Ramulator 2, maximizing memory bus burst efficiency and bank parallelism.
  3. Resets `weight->load_sisters = false` immediately, preventing redundant loads during subsequent matrix multiplication stages of the same expert.
* **Failure Mode Without It:** DDR5 transfer scheduling would serialize into three separate requests, introducing artificial queuing delays and distorting memory bus contention profiles.

---

### Part 7: Memory Interface Dispatch & DDR5 Cache Bypassing
**Files:** [`src/hardware/layer_impl.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/layer_impl.cpp), [`src/hardware/device.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp)

#### Quoted Source Code (Cache Bypass Key in layer_impl.cpp)
```cpp
//src/hardware/layer_impl.cpp
ExecStatus issueRamulatorBatch(
    Device_Ptr device,
    LayerType layer_type,
    ProcessorType processor_type,
    DRAMRequestType dram_request_type,
    PIMOperandType pim_operand_type,
    const std::vector<Tensor_Ptr>& tensors,
    MemoryTarget target)

//Calculate aggregate size
long total_size = 0;
for (auto tensor : tensors) {
    total_size += tensor->getSize();
}

//Create execution key
CacheKey key = std::make_tuple(
    layer_type,
    processor_type,
    dram_request_type,
    total_size,
    target
);

//Bypass or not
bool bypass_cache =
    (target == MemoryTarget::OFFCHIP_DRAM);
```

#### Quoted Source Code (Off-Chip Memory Object Creation in layer_impl.cpp)
```cpp
//src/hardware/layer_impl.cpp
std::list<DRAMRequest::Ptr> requests;
for (auto tensor : tensors) {
    DRAMRequest::Ptr dram_request =
        DRAMRequest::Create(dram_request_type);
    if (target == MemoryTarget::OFFCHIP_DRAM &&
        device->offchip_mmap_controller) {
        auto offchip_mem_obj = MemoryObject::Create(
            tensor->getMMap(),
            tensor->getMemoryObject()->getLogicAddr(),
            tensor->getSize(),
            device->offchip_mmap_controller
        );
        dram_request->AddOperand(
            offchip_mem_obj,
            pim_operand_type
        );
    } else {
        dram_request->AddOperand(
            tensor->getMemoryObject(),
            pim_operand_type
        );
    }
    requests.push_back(dram_request);
}
```

#### Quoted Source Code (Dispatch in device.cpp)
```cpp
//src/hardware/device.cpp
void Device::run_ramulator_batch(
    const std::list<DRAMRequest_Ptr>& requests,
    MemoryTarget target) {
    if (target == MemoryTarget::OFFCHIP_DRAM &&
        offchip_dram_interface != nullptr) {
        offchip_dram_interface->HandleRequest(requests, 0);
    } else {
        dram_interface->HandleRequest(requests, 0);
    }
}
```

#### Quoted Source Code (Status Retrieval in layer_impl.cpp)
```cpp
//src/hardware/layer_impl.cpp
//Submit Request List
device->run_ramulator_batch(requests, target);

//Retrieve Execution Status from selected memory interface and return status
if (target == MemoryTarget::OFFCHIP_DRAM &&
    device->offchip_dram_interface) {
    exec_status =
        device->offchip_dram_interface->getExecStatus();
} else {
    exec_status =
        device->dram_interface->getExecStatus();
}
return exec_status;
```

#### Why This Change Was Made
* **Problem Solved:** LLMSimulator includes an internal analytical acceleration table (`CacheKey`) that caches simulated DRAM latency for recurring tensor shapes to speed up simulation. Additionally, memory requests must use the correct physical address mapping scheme.
* **Architectural Purpose:**
  1. `bypass_cache = (target == MemoryTarget::OFFCHIP_DRAM)` forces all off-chip DDR5 requests to bypass analytical shortcut tables and execute cycle-accurate simulation in Ramulator 2. Because DDR5 bank conflicts and row buffer hit/miss dynamics vary dynamically with bus traffic, analytical caching would produce inaccurate flat delays.
  2. Attaches `offchip_mmap_controller` to the request's `MemoryObject`, ensuring the address translation follows DDR5 geometry rather than HBM.
  3. `run_ramulator_batch()` dispatches the request list directly to `offchip_dram_interface->HandleRequest()`, and retrieves timing results from `offchip_dram_interface->getExecStatus()`.
* **Failure Mode Without It:** DDR5 requests would either hit synthetic HBM analytical cache entries (producing instant 0 ns or HBM-speed returns) or corrupt HBM memory controller internal state.

---

### Part 8: Post-Loading Residency Registration & Latency Accounting
**File:** [`src/hardware/linear_impl.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/linear_impl.cpp)

#### Quoted Source Code
```cpp
//src/hardware/linear_impl.cpp
time_ns weight_memory_cost = weight_status.memory_duration;

if (weight->is_expert_weight) {
    if (weight->weight_target == MemoryTarget::OFFCHIP_DRAM) {
        long long total_expert_size = 0;
        for (auto t : batch_tensors) {
            total_expert_size += t->getSize();
        }
        device->mark_expert_resident(
            weight->expert_layer_id,
            weight->expert_id,
            total_expert_size);
    }
}
```

#### Why This Change Was Made
* **Problem Solved:** Once Ramulator completes the physical transfer of the 88 MB batch across the DDR5 bus, the simulator must account for the transfer duration and transition the expert's state to "resident in HBM."
* **Architectural Purpose:**
  1. Captures `weight_status.memory_duration` (the ~3.9 ms simulated transfer time calculated by Ramulator 2) and accumulates it into the device timeline and operator execution metrics.
  2. Calls `device->mark_expert_resident(layer_id, expert_id, total_expert_size)`:
     - Triggers LRU eviction if needed to free up 88 MB.
     - Adds `(layer_id, expert_id)` to `resident_experts`.
     - Places the expert at the MRU position in `lru_expert_list`.
     - Appends an eviction/insert record to `expert_cache.csv`.
  3. Logs the completed event to `expert_load.csv` with exact start timestamp, duration (~3,901,000 ns), and transferred byte volume.
* **Failure Mode Without It:** The expert would never be marked as resident, causing every subsequent access to re-trigger an off-chip DDR5 fetch (100% cold miss rate), while load duration would fail to register on the device timeboard.

---

## 3. High-Level Flow Chart of MoE Dual-Memory Execution

The diagram below synthesizes the complete architectural flow across all modified components:

```
                      ┌────────────────────────────────────────┐
                      │        Token Generation Request        │
                      └────────────────────────────────────────┘
                                           │
                                           ▼
                      ┌────────────────────────────────────────┐
                      │      MoE Routing Layer (Top-K=8)       │
                      │  Records: traces/expert_routing.csv    │
                      └────────────────────────────────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        │ For each selected Expert (0..7)     │
                        ▼                                     ▼
 ┌──────────────────────────────────────────────┐  ┌──────────────────────────────────────────┐
 │         Check HBM Residency Status           │  │      Check HBM Residency Status          │
 │       Device::is_expert_resident()           │  │      Device::is_expert_resident()        │
 └──────────────────────────────────────────────┘  └──────────────────────────────────────────┘
                        │                                                     │
             [YES: Cache Hit]                                      [NO: Cache Miss]
                        │                                                     │
                        ▼                                                     ▼
 ┌──────────────────────────────────────────────┐  ┌──────────────────────────────────────────┐
 │ 1. Move to MRU in lru_expert_list            │  │ 1. Tag weight_target = OFFCHIP_DRAM      │
 │ 2. Log 'HIT' to traces/expert_cache.csv      │  │ 2. Set load_sisters = true               │
 │ 3. Target: MemoryTarget::HBM                 │  │ 3. Group W1, W2, W3 into batch_tensors   │
 └──────────────────────────────────────────────┘  └──────────────────────────────────────────┘
                        │                                                     │
                        │                                                     ▼
                        │                          ┌──────────────────────────────────────────┐
                        │                          │ Run Ramulator 2 DDR5 Batch Simulation    │
                        │                          │   • 88 MB transfer across DDR5 bus       │
                        │                          │   • Simulated transfer cost: ~3.90 ms    │
                        │                          │   • Bypass analytical latency cache      │
                        │                          └──────────────────────────────────────────┘
                        │                                                     │
                        │                                                     ▼
                        │                          ┌──────────────────────────────────────────┐
                        │                          │ Evict LRU Experts from HBM if Full       │
                        │                          │   • Update lru_expert_list & map         │
                        │                          │   • Log eviction to expert_cache.csv     │
                        │                          └──────────────────────────────────────────┘
                        │                                                     │
                        │                                                     ▼
                        │                          ┌──────────────────────────────────────────┐
                        │                          │ Mark Expert Resident in HBM              │
                        │                          │   • Device::mark_expert_resident()       │
                        │                          │   • Log duration to expert_load.csv      │
                        │                          └──────────────────────────────────────────┘
                        │                                                     │
                        └──────────────────┬──────────────────────────────────┘
                                           │
                                           ▼
                      ┌────────────────────────────────────────┐
                      │   Execute Expert FFN Compute on GPU    │
                      │     (Gate, Up, Down Projections)       │
                      │       Compute Duration: ~1.38 ms       │
                      └────────────────────────────────────────┘
                                           │
                                           ▼
                      ┌────────────────────────────────────────┐
                      │  Combine Expert Outputs & Next Layer   │
                      └────────────────────────────────────────┘
```

---

## 4. Complete End-to-End Execution Flow (Based on Report)

The evaluation report synthesizes the complete MoE request lifecycle into **9 sequential stages**:

### Step 1: Sequence Scheduling & Routing Decision Recording
* **Invoked in:** [`src/scheduler/sequence.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp)
* **Action:** The request scheduler advances the current token and queries the MoE gating router. For DeepSeek-V3, Top-8 experts are chosen out of 256 candidates per MoE stage.
* **Telemetry Output:** The selected `(token_id, stage_id, expert_id)` tuples are immediately streamed to [`expert_routing.csv`](file:///home/mohman/ca_project/LLMSimulator/traces/expert_routing.csv).

### Step 2: Expert Construction & Sister Linking
* **Invoked in:** [`src/module/expert.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/module/expert.cpp)
* **Action:** `ExpertFFN` instantiates the three projection layers ($W_1$ Gate, $W_2$ Down, $W_3$ Up).
* **Code Mechanism:** The three weight tensors are cross-referenced via `sister_weights` pointers, establishing an 88 MB multi-tensor bundle for atomic loading.

### Step 3: Residency Verification in Linear Layer
* **Invoked in:** [`src/module/linear.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/module/linear.cpp)
* **Action:** Prior to invoking GPU execution, `Linear::forward()` queries `Device::is_expert_resident(layer_id, expert_id)`.
* **Branching:**
  - **If Resident (Cache Hit):** The expert is refreshed to the MRU position in `lru_expert_list`. `expert_cache.csv` records a `HIT` event. Execution proceeds directly to compute via HBM.
  - **If Non-Resident (Cache Miss):** The weight tensor is flagged with `weight_target = MemoryTarget::OFFCHIP_DRAM` and `load_sisters = true`.

### Step 4: Sister Weight Batch Assembly
* **Invoked in:** [`src/hardware/linear_impl.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/linear_impl.cpp)
* **Action:** `LinearExecutionGPU` detects `weight->load_sisters == true`. It aggregates $W_1$, $W_2$, and $W_3$ into `batch_tensors` (88 MB) and calls `issueRamulatorBatch`.

### Step 5: Interface Routing & Analytical Cache Bypass
* **Invoked in:** [`src/hardware/layer_impl.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/layer_impl.cpp)
* **Action:** `issueRamulatorBatch` checks `target == MemoryTarget::OFFCHIP_DRAM`:
  - Sets `bypass_cache = true`, disabling analytical shortcut interpolation.
  - Translates tensor logical addresses through `offchip_mmap_controller` using the expanded DDR5 geometry ($10,485,760$ rows).
  - Submits the batch via `device->run_ramulator_batch(requests, target)`.

### Step 6: Cycle-Accurate Ramulator 2 DDR5 Simulation
* **Invoked in:** [`src/hardware/device.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp)
* **Action:** `Device::run_ramulator_batch()` routes the request directly to `offchip_dram_interface->HandleRequest()`. Ramulator 2 simulates DRAM bankgroup activation, read burst delays, channel contention, and bus turnaround times for the 88 MB payload, returning an exact `ExecStatus::memory_duration` (median ~3.901 ms).

### Step 7: HBM Capacity Management & LRU Eviction
* **Invoked in:** [`src/hardware/device.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp)
* **Action:** Before registering the newly loaded expert in HBM, `mark_expert_resident()` evaluates:
  $$\text{resident\_bytes} + 88\text{ MB} > 80\text{ GB}$$
  If true, it pops experts from the front of `lru_expert_list` (LRU candidates), erases them from `resident_experts`, subtracts their byte footprint, and logs eviction events to `expert_cache.csv` until 88 MB is free.

### Step 8: Registration & Load Telemetry
* **Invoked in:** [`src/hardware/linear_impl.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/linear_impl.cpp)
* **Action:** The incoming expert is registered in `resident_experts` and appended to the back of `lru_expert_list` (MRU). A detailed completion record is flushed to [`expert_load.csv`](file:///home/mohman/ca_project/LLMSimulator/traces/expert_load.csv), recording:
  - `expert_layer_id`, `expert_id`
  - `start_time_ns`, `end_time_ns`, `load_duration_ns` (~3.9 ms)
  - `bytes_transferred` (88,080,384 bytes)
  - `hit_status` (`MISS`)

### Step 9: Expert FFN Execution & Token Output
* **Invoked in:** GPU Compute Subsystem
* **Action:** With weights now residing in HBM, the GPU executes Gate, Up, and Down GEMM operations (~1.38 ms). The computed activations are reduced across all Top-8 experts, passed to the residual connection, and propagated to the next decoder stage.

---

## 5. Architectural Summary Table of Modified Files

| Source File | Primary Function / Scope | Key Modification | Architectural Role |
| :--- | :--- | :--- | :--- |
| [`src/hardware/base.h`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/base.h) | Global Definitions | Added `enum class MemoryTarget { HBM, OFFCHIP_DRAM };` | Categorizes memory requests for multi-tier routing |
| [`src/dram/memory_config.h`](file:///home/mohman/ca_project/LLMSimulator/src/dram/memory_config.h) | DRAM Configurations | Defined `ddr5_32gb_x16_2ch` with $10,485,760$ rows | Expands DDR5 capacity to 2.75 TB to hold all 14,592 experts |
| [`src/hardware/device.h`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.h) | `Device` Header | Declared `offchip_dram_interface`, `resident_experts`, `lru_expert_list` | Adds secondary memory interface and LRU cache state tracking |
| [`src/hardware/device.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/device.cpp) | `Device` Implementation | Initialized `offchip_dram_interface`; implemented `is_expert_resident()` & `mark_expert_resident()` | Dual-DRAM lifecycle management and $O(1)$ LRU eviction |
| [`src/module/tensor.h`](file:///home/mohman/ca_project/LLMSimulator/src/module/tensor.h) | `Tensor` Class | Added `is_expert_weight`, `expert_id`, `weight_target`, `sister_weights`, `load_sisters` | Tensors carry MoE metadata and multi-tier memory target tags |
| [`src/module/expert.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/module/expert.cpp) | `ExpertFFN` Class | Linked $W_1, W_2, W_3$ projection tensors via `sister_weights` | Enables single-burst 88 MB batch loading of full expert parameters |
| [`src/module/linear.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/module/linear.cpp) | `Linear::forward` | Intercepts weight access; redirects to `OFFCHIP_DRAM` on cache miss | Connects forward compute pass to HBM cache residency check |
| [`src/hardware/linear_impl.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/linear_impl.cpp) | GPU Linear Execution | Packs `batch_tensors`; captures `weight_status.memory_duration`; marks resident | Batches sister transfers and accounts for DDR5 latency in timeline |
| [`src/hardware/layer_impl.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/hardware/layer_impl.cpp) | Hardware Layer Driver | Sets `bypass_cache = true`; binds `offchip_mmap_controller`; dispatches batch | Enforces cycle-accurate Ramulator 2 DDR5 simulation without shortcuts |
| [`src/scheduler/sequence.cpp`](file:///home/mohman/ca_project/LLMSimulator/src/scheduler/sequence.cpp) | Scheduler | Logs routing assignments to `expert_routing.csv` | Provides ground-truth trace of expert activation sequence |
