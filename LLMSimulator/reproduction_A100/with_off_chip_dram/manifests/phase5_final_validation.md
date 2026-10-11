# Phase 5 Final Validation Manifest

## Run Configuration
- **Workload Trace:** `mixtral_synthesis_32_2_GPU_N1_D1_TP1_DP1_maxbatch1_maxprocess524288_iter5_skew0_precision_byte2_parallel_execution0_ramul_decode`
- **Config Path:** `reproduction_A100/with_off_chip_dram/configs/b0_phase5.yaml`
- **Model Parameters:** Unchanged (Verified Mixtral Architecture)
- **Routing Semantics:** Unchanged
- **Logic-PIM:** Disabled
- **ET:** 1
- **GPUs:** 1
- **Phase 4 Synthetic Request:** Disabled

## Exact Command
```bash
mkdir -p reproduction_A100/with_off_chip_dram/results/b0_phase5/logs && \
cd build && make -j$(nproc) && cd .. && \
./build/run reproduction_A100/with_off_chip_dram/configs/b0_phase5.yaml > reproduction_A100/with_off_chip_dram/results/b0_phase5/logs/run.log 2>&1
```

## Artifact Paths
- **Official Log Path:** `reproduction_A100/with_off_chip_dram/results/b0_phase5/logs/run.log`
- **Telemetry Path:** `reproduction_A100/with_off_chip_dram/results/b0_phase5/csv/expert_load.csv`

## Telemetry Sanity Check (Layer 1, Expert 1)
- **Expert Weight Accounting:** `352,321,536` bytes. This accounts for:
    - `W1 (gate_proj)`: 117,440,512 bytes
    - `W3 (up_proj)`: 117,440,512 bytes
    - `W2 (down_proj)`: 117,440,512 bytes
- **Representative MISS (Iteration 0):**
    - `hit_or_miss`: miss
    - `memory_source`: OFFCHIP_DRAM
    - `memory_destination`: HBM
    - `measured DDR5 load duration`: `981,447 ns` (0.98 ms)
- **Representative HIT (Iteration 1+):**
    - `hit_or_miss`: hit
    - `memory_source`: HBM
    - `memory_destination`: HBM
    - `additional load duration`: `0 ns`

## Validation Status
- **CHECK 1:** `run.log` exists inside `b0_phase5/logs/` -> **PASS**
- **CHECK 2:** `run.log` is generated from the actual simulator execution -> **PASS**
- **CHECK 3:** `run.log` contains Phase-5 `stdout`/`stderr` -> **PASS**
- **CHECK 4:** Phase-4 synthetic message does NOT appear in the production run -> **PASS**
- **CHECK 5:** `expert_load.csv` contains `OFFCHIP_DRAM` misses -> **PASS**
- **CHECK 6:** `OFFCHIP_DRAM` misses have NONZERO simulated load duration -> **PASS**
- **CHECK 7:** `HBM` hits have zero additional off-chip load duration -> **PASS**
- **CHECK 8:** Iteration 0 demonstrates misses -> **PASS**
- **CHECK 9:** Later iterations demonstrate hits -> **PASS**
- **CHECK 10:** Same `(layer_id, expert_id)` transitions correctly `MISS -> HIT` -> **PASS**
- **CHECK 11:** All required expert weight tensors are represented -> **PASS**
- **CHECK 12:** Expert execution follows required memory load completion -> **PASS**
- **CHECK 13:** No duplicate DDR5 load occurs for resident experts -> **PASS**
- **CHECK 14:** No change to routing semantics occurred -> **PASS**
- **CHECK 15:** No change to verified Mixtral model parameters occurred -> **PASS**
- **CHECK 16:** No Logic-PIM -> **PASS**
- **CHECK 17:** ET = 1 -> **PASS**
- **CHECK 18:** Exactly one GPU -> **PASS**
