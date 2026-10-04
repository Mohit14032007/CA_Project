# Phase 9: Expert Access Predictability Report

## 1. Objective
The goal of this phase is to analyze the `expert_routing.csv` trace generated in Phase 7 to determine whether the expert-routing sequence contains sufficient temporal or spatial structure to make expert accesses predictable. This analysis is critical to determine whether implementing a prefetch mechanism is scientifically justified to hide the ~981 µs off-chip memory latency.

## 2. Input Trace
The frozen Phase 7 trace `results/b1_full/traces/expert_routing.csv` was used as the definitive source.
- **Record Count**: 20,480 routing decisions
- **Iterations (Tokens)**: 5 consecutive tokens
- **Concurrent Sequences (Batch Size)**: 32 (`request_id`s)
- **Layers**: 32 MoE layers
- **Experts**: 8 experts per layer
- **Top-K**: 2 experts selected per token per layer

## 3. Expert Frequency and Entropy
- **Routing Skew**: The frequency distribution shows moderate skew rather than uniform distribution. This is expected due to the synthetic workload generator's skewness parameter (`w ~ 1/k^s`).
- **Maximum Probability**: The most frequent expert typically achieves ~30% access probability, compared to a baseline of 12.5% for uniform random selection among 8 experts.

## 4. Temporal Predictability
Temporal correlation was evaluated by comparing the top-2 experts selected for token $t$ with those selected for token $t+1$ at the same layer for the same sequence.
- **Top-K Recall (1-step lookahead)**: ~24.8%
- **Exact Set Match**: ~3.7%

**Conclusion on Temporal Predictability**: The observed top-K recall (24.8%) is mathematically indistinguishable from random selection (25% for 2 out of 8 experts). The exact set match (3.7%) also mirrors the random baseline (1/28 ≈ 3.57%). There is **no meaningful temporal correlation** in the trace sequence.

## 5. Simple Predictor Performance

| Predictor | Lookahead | Top-1 Accuracy | Top-k Recall (k=2) | Exact Set Accuracy |
|-----------|-----------|----------------|--------------------|--------------------|
| Most Frequent | 1 token | ~29.6% | N/A | N/A |
| Previous Token | 1 token | N/A | ~24.8% | ~3.7% |
| Previous Token | 2 tokens | N/A | ~24.5% | ~3.9% |
| *Random Baseline* | *N/A* | *12.5%* | *25.0%* | *~3.57%* |

## 6. Prefetch Opportunity
Because temporal predictability is indistinguishable from random noise, attempting to prefetch experts based on the previous token's routing decisions would result in an overwhelming number of incorrect fetches (cache pollution and wasted memory bandwidth). 
- A static predictor fetching the "most frequent" expert would guess correctly ~30% of the time, but would miss the other 70%, still incurring massive 981 µs latency stalls.

## 7. Synthetic Workload Limitation
**CRITICAL FINDING**: The routing unpredictability observed here is an artifact of the `data: synthesis` configuration used in Phase 7. Inspecting `src/scheduler/sequence.cpp` confirms that the synthetic generator picks experts independently for each token based purely on a static probability distribution (skew). It fundamentally lacks the semantic continuity, context-dependence, and long-range dependencies inherent in real LLM inference workloads (e.g., real user prompts).

## 8. Scientific Conclusion
**"Does the current workload provide enough expert-access predictability to justify implementing a prefetcher?"**
**NO.** The synthetic workload's routing is completely independent per token (effectively random, with a slight static frequency skew). Implementing a prefetcher for this specific trace would yield zero scientific insight into prefetching performance, as the prefetcher would invariably fail to predict the random sequence.

## 9. Recommendation for Phase 10
Do NOT implement prefetching yet. 
Before evaluating eviction or prefetching mechanisms, the simulator MUST be transitioned from a synthetic random-routing workload to a trace derived from a real LLM inference workload (e.g., a real Mixtral trace). Only a real trace possesses the inherent semantic temporal correlation necessary to evaluate predictive prefetching strategies. Phase 10 should focus entirely on integrating real workload traces into the simulator.
