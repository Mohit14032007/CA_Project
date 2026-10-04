# Phase 9 Predictability Analysis Manifest

## Included Artifacts
This phase produced quantitative analyses and reports assessing the temporal predictability of expert routing within the Phase 7 `expert_routing.csv` trace.

### Analyses
- `results/phase9_predictability/analysis/expert_frequency.csv`: Calculates frequency distribution across 8 experts per layer.
- `results/phase9_predictability/analysis/temporal_transition_matrix.csv`: Examines consecutive expert selections.
- `results/phase9_predictability/analysis/expert_reuse_distance.csv`: Calculates average distance (in tokens) until an expert is re-selected.
- `results/phase9_predictability/analysis/topk_overlap.csv`: Tracks the set intersection of top-K experts across consecutive tokens.
- `results/phase9_predictability/analysis/expert_cooccurrence.csv`: Identifies pairs of experts commonly chosen together.
- `results/phase9_predictability/analysis/routing_entropy.csv`: Measures Shannon entropy of expert selection per layer.
- `results/phase9_predictability/analysis/predictor_results.csv`: Evaluates simple predictive strategies (Previous Token, Most Frequent) against a random baseline.
- `results/phase9_predictability/analysis/predictability_vs_lookahead.csv`: Predictability accuracy across 1-token and 2-token lookahead windows.

### Reports
- `results/phase9_predictability/phase9_predictability_report.md`: The definitive scientific summary of Phase 9.

### Plots
- `results/phase9_predictability/plots/expert_frequency_distribution.png`: Visualization of the frequency skew across layers.
