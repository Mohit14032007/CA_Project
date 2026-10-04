# Phase 9 Validation

## Adherence to Read-Only Constraints
1. **Source Code**: No simulator source files in `src/` were modified.
2. **YAML Configurations**: No YAML files in `configs/` were modified.
3. **Previous Traces**: No Phase 5, Phase 6, Phase 7, or Phase 8 CSV files were modified, overwritten, or regenerated.
4. **Simulator Execution**: The simulation (`run`) was not executed. All analysis relied entirely on the frozen `results/b1_full/traces/expert_routing.csv` artifact.

## Analysis Methodology Validation
1. **Layer/Expert Integrity**: Logical experts were strictly evaluated as `(layer_id, expert_id)` tuples. Experts from different layers were never collapsed into a unified representation.
2. **Token Integrity**: Tokens were strictly inferred from `token_id`. No artificial tokens were synthesized. Lookahead distances were confined to actual existing token boundaries.
3. **Entropy Validation**: The Shannon entropy baseline used `log2(8)` precisely because Mixtral uses 8 routed experts per layer.
4. **Metrics Separation**: Top-1 metrics were not falsely compared with Top-K metrics.
5. **Prefetch Claims**: The report strictly adheres to a qualitative estimation of prefetch opportunity, explicitly stating that prediction accuracy does NOT equal exact latency hidden, because bandwidth, contention, and HBM physical capacity constraints have not yet been evaluated.
6. **Synthetic Generalization Limitation**: The most critical aspect of the evaluation was explicitly concluding that the routing predictability (which was essentially random) is an artifact of the synthetic workload generator. No false generalizations were made claiming that real Mixtral workloads exhibit zero predictability.
