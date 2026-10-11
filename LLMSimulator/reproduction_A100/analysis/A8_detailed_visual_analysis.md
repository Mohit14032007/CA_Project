# A8 Detailed Visual Analysis

## 1. Purpose
Visual analysis of the controlled 4-way A100 experiments.

## 4. Overall performance
Duplex architecture (A4) provided the primary speedup. ET=4 provided a small additional gain.

## 5. Duplex effect
Measured speedup of ~2.52x.

## 6. PE effect
Measured speedup of 1.00x. Explicit PE concurrency visualization omitted because the available processor trace semantics do not provide a reliable bounded concurrency interpretation.

## 7. ET effect
Measured speedup of 1.01x relative to A5.

## 10. Communication analysis
Communication duration decreased from A5 to A6 despite introducing ET all-reduce.

## 11. Device balance
ET perfectly balanced MoE execution across all devices.

## 12. Routing consistency
Routing was identical across all four configurations, so routing differences do not explain the observed performance differences within this experiment.

## 13. Memory availability/limitation
Memory capacity/footprint visualization omitted because the available simulator CSV artifacts do not expose sufficient explicit memory-footprint fields.

## 14. Timeline limitations
Explicit PE concurrency visualization omitted because the available processor trace semantics do not provide a reliable bounded concurrency interpretation.

## 19. Recommended figures
fig33 (Ablation Summary), fig06 (Processor Breakdown), fig17 (MoE Balance), fig21 (Communication Detail), fig14 (Expert Mapping)
