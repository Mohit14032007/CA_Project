# LLMSimulator Repository Manifest

## 1. Source Control
- **Git Commit:** `419252761fbdb95b789778a02256d458a5537ec7`
- **Branch:** `main`

## 2. Architecture & Build
- **Build System:** CMake + Make
- **Executable:** `./sim`
- **Config Loading:** YAML-based configuration (via YAML-CPP) parsed into the `config` singleton.
- **Model Configuration System:** Defined in `src/model/model_config.h` (e.g., Mixtral uses `ModelConfig::get_mixtral_config()`). Configured via `model_name` in YAML.

## 3. Hardware Architecture
- **Hardware Configuration System:** Defined in `src/hardware/hardware_config.h`, which contains static presets (e.g., `A100`, `H100`). Selected via `gpu_gen` in YAML.
- **DRAM/HBM Configuration Files:** Extensively specified through constants in the codebase and parameters passed in `SystemConfig`.
- **Processor Type Implementation:** Uses `ProcessorType` enum (`GPU`, `LOGIC`, etc.) evaluated in `TopModuleGraph::simulate()`.
- **Logic-PIM Implementation:** Simulated using independent memory bandwidth variables (`logic_memory_bandwidth = memory_bandwidth * logic_x`) and specialized Ops/Byte (`logic_op_b`).

## 4. Parallelism & Routing
- **TP/DP Implementation:** Tensor Parallelism (`ne_tp_dg`, `TP`) naturally shards non-expert layers across devices. DP is supported but currently unused (DP=1).
- **Expert Tensor Parallelism:** Shards individual experts across devices using `e_tp_dg` parameter (Expert Tensor degree).
- **PE Implementation:** A Processing Element dynamic load balancer (enabled via `parallel_execution`) actively compares execution cost functions to reassign bottlenecks dynamically during execution.
- **Routing Instrumentation:** Output to `expert_routing.csv` in append-only mode utilizing a separate asynchronous trace logging utility to preserve core timeline integrity.

## 5. Metrics & Output
- **Statistics System:** Handled via custom structured CSV reporting loop at the end of execution.
- **CSV Generation:** Found in `Cluster::exportPerformance()` and `Cluster::runIterationMixed()`.
- **TimeBoard:** Tracks operational milestones per device, providing hierarchical node metrics.
- **Gantt/Export System:** `exportGantt` writes `processor_trace.csv` for fine-grained per-layer telemetry on a per-device basis.
- **Processor Instrumentation:** Fully device-aware, appending sequentially into `processor_trace.csv` representing each individual device's valid execution timeline.
