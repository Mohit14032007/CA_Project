# Phase 9 Plot Manifest

## Plot 1
- **Filename**: `01_expert_frequency.png`
- **Source CSV**: `results/phase9_predictability/analysis/expert_frequency.csv`
- **Analysis Script**: `scratch/generate_presentation_plots.py`
- **Question**: Is routing uniform, or is there a dominant expert?
- **Main Trend**: Moderate skew exists; the most frequent expert captures ~30% of access.
- **Conclusion**: There is a static bias in the synthetic workload.
- **Supports Prefetching?**: No, static skew alone cannot drive a dynamic temporal prefetcher.

## Plot 2
- **Filename**: `02_layer_expert_heatmap.png`
- **Source CSV**: `results/phase9_predictability/analysis/expert_frequency.csv`
- **Analysis Script**: `scratch/generate_presentation_plots.py`
- **Question**: Are some experts systematically preferred in particular layers?
- **Main Trend**: The skew pattern is completely uniform across all 32 layers.
- **Conclusion**: The synthetic generator applies the identical statistical distribution to every depth layer.
- **Supports Prefetching?**: No.

## Plot 3
- **Filename**: `03_temporal_transition_heatmap.png`
- **Source CSV**: `results/b1_full/traces/expert_routing.csv` (aggregated during plotting)
- **Analysis Script**: `scratch/generate_presentation_plots.py`
- **Question**: Does knowing the current expert tell us which expert will appear next?
- **Main Trend**: Transition probabilities align strictly with the static marginal frequencies.
- **Conclusion**: Tokens transition independently without sequential structural logic.
- **Supports Prefetching?**: No.

## Plot 4
- **Filename**: `04_topk_overlap_distribution.png`
- **Source CSV**: `results/phase9_predictability/analysis/topk_overlap.csv`
- **Analysis Script**: `scratch/generate_presentation_plots.py`
- **Question**: How often do consecutive tokens select overlapping expert sets?
- **Main Trend**: Mean overlap is ~0.496 experts, matching random expectations for the skewed distribution.
- **Conclusion**: Sequential sets act as independent samples.
- **Supports Prefetching?**: No.

## Plot 5
- **Filename**: `05_routing_entropy.png`
- **Source CSV**: `results/phase9_predictability/analysis/routing_entropy.csv`
- **Analysis Script**: `scratch/generate_presentation_plots.py`
- **Question**: Is expert selection heavily concentrated or broadly spread?
- **Main Trend**: Entropy is consistently ~2.8 bits per layer (max 3.0 bits).
- **Conclusion**: Routing decisions remain highly variable despite the slight static skew.
- **Supports Prefetching?**: No.

## Plot 6
- **Filename**: `06_predictor_comparison.png`
- **Source CSV**: `results/phase9_predictability/analysis/predictor_results.csv`
- **Analysis Script**: `scratch/generate_presentation_plots.py`
- **Question**: Can simple history-based predictors outperform random guessing?
- **Main Trend**: A previous-token predictor achieves 24.8% Top-K recall vs. 25.0% for random guessing.
- **Conclusion**: Historical information is useless for predicting this synthetic trace.
- **Supports Prefetching?**: No, it fundamentally advises against prefetching on this trace.

## Plot 7
- **Filename**: `07_predictability_vs_lookahead.png`
- **Source CSV**: `results/phase9_predictability/analysis/predictability_vs_lookahead.csv`
- **Analysis Script**: `scratch/generate_presentation_plots.py`
- **Question**: Does prediction become worse as we predict further into the future?
- **Main Trend**: Accuracy is flat and matches random guessing at all lookahead horizons.
- **Conclusion**: The trace is uniformly uncorrelated.
- **Supports Prefetching?**: No.

## Plot 8
- **Filename**: `08_expert_reuse_distance.png`
- **Source CSV**: `results/phase9_predictability/analysis/expert_reuse_distance.csv`
- **Analysis Script**: `scratch/generate_presentation_plots.py`
- **Question**: How quickly are logical experts reused?
- **Main Trend**: Wide distribution correlated primarily with the baseline expert frequency.
- **Conclusion**: The interval between logical accesses is governed by random sampling probability.
- **Supports Prefetching?**: No.
