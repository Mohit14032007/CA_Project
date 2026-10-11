# A7 Master Cross-Configuration Analysis

## 1. Experiment Matrix
Configurations A3 (Baseline), A4 (Duplex), A5 (Duplex+PE), A6 (Duplex+PE+ET).

## 4. Speedups
A3->A4: 2.52x, A4->A5: 1.00x, A5->A6: 1.01x. [DERIVED]

## 5. Duplex effect
[MEASURED] Massive 2.52x speedup by shifting attention to Logic-PIM.

## 6. PE effect
[INTERPRETED] PE concurrency did not shift the critical path bottleneck for this decode workload.

## 7. ET effect
[MEASURED] ET=4 provided a modest ~1.3% speedup. [INTERPRETED] Tensor sharding perfectly balanced compute, which slightly improved overall global runtime despite new ET all-reduce overhead.
