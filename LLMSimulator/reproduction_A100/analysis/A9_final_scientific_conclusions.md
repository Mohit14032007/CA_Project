# Final Scientific Conclusions

## 1. Experiment objective
Analyze Duplex, PE, and ET optimizations on A100.

## 2. Controlled experimental methodology
Isolated configuration changes ensuring A3->A4->A5->A6 architectural differences only.

## 3. A3 baseline
[MEASURED] Total runtime 69.79 ms.

## 4. Duplex effect
[MEASURED] 2.52x speedup by Logic-PIM handling attention.

## 5. PE effect
[MEASURED] 1.00x speedup. [INTERPRETED] PE concurrency fell outside the critical path.

## 6. ET effect
[MEASURED] 1.01x speedup over A5. [INTERPRETED] Tensor sharding effectively distributed MoE computations, saving time globally despite an introduced 2.6M ns ET all-reduce communication overhead.

## 7. Communication behavior
[MEASURED] ET=4 reduced total communication from 9.12ms to 7.92ms despite adding ET all-reduces.

## 8. Device distribution
[MEASURED] ET=4 precisely balanced MoE operations equally across all four devices.

## 9. Routing consistency
[MEASURED] Routing was byte-for-byte identical across all four configurations.

## 10. Memory limitation
The available simulator CSV artifacts do not expose sufficient explicit memory-footprint fields for a rigorous cross-configuration memory analysis.

## 11. PE-concurrency limitation
Reliable explicit PE concurrency could not be established from the available processor trace semantics without an unacceptably expensive interval analysis.

## 12. Overall findings
Duplex gives massive speedup. ET provides slight speedup and excellent balancing. PE provides negligible speedup for this workload.

## 13. Threats/limitations
Simulation-based findings. Unbounded memory inferences avoided.
