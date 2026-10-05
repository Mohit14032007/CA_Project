# Final Routing Validation

Data source: `expert_routing.csv` from all four target configurations.

## Summary

- **Total traces verified:** 4 (`gpu_baseline`, `duplex`, `duplex_pe`, `duplex_pe_et`)
- **Total records per trace:** 10,240
- **Requests:** 32 (IDs 2624 through 2655)
- **Token positions:** 2142, 2143, 2144, 2145, 2146
- **Layers:** 32 (Layers 0-31)
- **Top-K Ranks:** 1 and 2
- **Experts:** Valid IDs 0 through 7

## Conclusion
Routing comparison:
**GPU == Duplex == Duplex+PE == Duplex+PE+ET**

The token routing traces are 100.0% identical across all configuration modes. This confirms the simulator successfully preserved the strict algorithmic correctness of the MoE routing regardless of how the operations were scheduled, parallelized, or executed across diverse hardware units.
