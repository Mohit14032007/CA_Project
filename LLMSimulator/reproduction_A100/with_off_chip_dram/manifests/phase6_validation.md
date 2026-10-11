# Phase 6 Validation Manifest

## Goal
Validate that the instrumentation accurately isolates `T_load` and `T_compute`, preserving the Phase-5 W1/W3/W2 concurrent/serial load behaviors without polluting the simulator results or existing validations.

## Validated Characteristics

1. **Representative MISS Correlation**:
    - `layer_id = 1`, `expert_id = 1` successfully identified and correlated.
    - Verified `load_duration_ns > 0` (recorded at `981,447 ns`)
    - Verified `compute_duration_ns > 0` (recorded at `43,318 ns`)
    - Confirmed memory source maps correctly to `OFFCHIP_DRAM` from Phase-5 output.

2. **Representative HIT Correlation**:
    - `layer_id = 0`, `expert_id = 0` successfully correlated.
    - Verified `load_duration_ns = 0`
    - Verified `compute_duration_ns > 0` (recorded at `48,692 ns`)
    - Confirmed memory source matches `HBM`.

3. **Multiple Layers/Expert IDs Supported**:
    - Evaluated 195 separate expert-invoke traces across 5 iterations.
    - Data spans all simulated active decoder layers.

4. **W1/W3/W2 Serial Tracking**:
    - Analyzed source logic in `linear_impl.cpp`. Confirmed that the duration accumulated into `exec_status.memory_duration` calculates serially as `W1_duration + W3_duration + W2_duration`.
    - No modification was made to this memory duration semantics for Phase 6.

5. **Isolating Simulator Timing vs. OS Timing**:
    - Traces accurately reflect physical simulator times via `timeboard.cpp`.
    - `T_load` and `T_compute` mapped independently.

6. **Safety Rules Respected**:
    - Single GPU + HBM execution.
    - Routing unchanged.
    - Residency policies preserved (Phase-5 frozen semantics remain untouched).
    - `ET=1` configured.
    - Output written precisely to `.csv` format avoiding pollution.
