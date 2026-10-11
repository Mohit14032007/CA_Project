# A6 Duplex + PE + ET Summary

## 1. Configuration
- a100_duplex_pe_et.yaml (GPU+LOGIC, PE=ON, ET=4)
## 10. Authoritative Runtime
- Measured Latency (CSV): 27288411.376557 ns

## 13. Interpretation
ET=4 introduces `moe_all_reduce_for_e_tp` communication. The computational distribution improved overall latency despite communication overhead.
Reliable explicit PE concurrency could not be established from the available processor trace semantics.
