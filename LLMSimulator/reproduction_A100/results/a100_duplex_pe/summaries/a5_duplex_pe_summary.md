# A5 Duplex + PE Summary

## 1. Configuration & 2. Workload
- **Config**: a100_duplex_pe.yaml (GPU+LOGIC, PE=ON)
- **Workload**: Decode-only (input=2048, batch=32, iter=5)

## 10. Authoritative Runtime
- Measured Latency (CSV): 27660707.242374 ns
- Traced Iteration Duration: 5562740.0 ns

## 13. Interpretation
PE modifies execution scheduling by allowing overlapping operations. Reliable PE concurrency could not be established from the available processor trace semantics. PE did not significantly change the overall measured wall-clock performance relative to A4.
