# Phase G: Expert Routing Analysis Summary

This analysis evaluates the token-level expert routing decisions recorded during the pure-decode workload (Mixtral 47B, 32 requests, 5 iterations/tokens).

## 1. Trace Integrity
- **Traces Validated:** 4 configurations (`gpu_baseline`, `duplex`, `duplex_pe`, `duplex_pe_et`)
- **Number of Requests:** 32
- **Token Positions:** 2142 to 2146 (5 consecutive token generation steps per request)
- **Layers:** 32 (Layers 0 through 31)
- **Total Routing Rows:** 10,240 records per configuration (32 requests × 5 tokens × 32 layers × 2 topk)
- **Trace completeness:** Verified. Every token/layer combination correctly generated exactly two records (Top-1 and Top-2 ranks). No malformed rows detected.

## 2. Cross-Configuration Consistency
**Identical routing across configurations:** **YES (100%)**
A direct row-by-row join (keyed on `request_id`, `token_id`, `layer_id`, `topk_rank`) proved that the `expert_id` selections were **100.0% identical** across all four configurations.
*Conclusion:* Hardware/execution topology changes (Logic-PIM, PE dynamic balancing, ET sharding) did NOT alter the observed router decisions for this workload. The routing algorithm behaves entirely independently of the execution architecture.

## 3. Expert Frequency
Expert selection is highly balanced across the 8 available experts.
- Max total selections (top-1 + top-2): 1,329
- Min total selections: 1,239
- Mean total selections: 1,280

## 4. Expert Imbalance
- **Standard Deviation:** ~35.75
- **Coefficient of Variation (CV):** ~0.028 (Extremely low)
- **Max/Mean Ratio:** 1.038
*Conclusion:* At a global aggregate level, the router distributes load almost perfectly uniformly. This likely reflects Mixtral's routing auxiliary loss heavily penalizing unbalanced states.

## 5. Temporal & Spatial Correlation
We analyzed transitions in top-1 expert selection:
- **Temporal Correlation:** Predicting token $t+1$ uses the *same* expert as token $t$ (same request, same layer).
- **Spatial Correlation:** Predicting layer $L+1$ uses the *same* expert as layer $L$ (same request, same token).

**Temporal Top-k Overlap:**
- Measured Jaccard overlap between the top-2 sets of token $t$ and $t+1$. 
- Output saved to `temporal_topk_overlap.csv`.

**Spatial Top-k Overlap:**
- Measured Jaccard overlap between the top-2 sets of layer $L$ and $L+1$.
- Output saved to `spatial_topk_overlap.csv`.

## 6. Prediction Accuracy
We tested naive prediction models against a global baseline (guessing the single most globally frequent expert).

| Strategy | Top-1 Accuracy | Baseline (Most Freq) Accuracy |
|---|---|---|
| **Temporal** (Same as $t-1$) | 17.80% | 25.95% |
| **Spatial** (Same as $L-1$) | 17.88% | 25.81% |

*Conclusion:* Predicting that a token uses the same top-1 expert as its predecessor (temporally or spatially) performs **worse** than statically guessing the globally dominant expert. While 17.8% is better than purely uniform random (12.5% or 1/8), it remains too low to build effective speculative execution or caching heuristics on simple spatial/temporal locality.

## 7. Routing Entropy
- **Global Entropy:** 2.999 bits (Theoretical maximum for 8 uniform experts is $\log_2(8) = 3.0$ bits).
- **Per-Layer Entropy:** Varies minutely from 2.94 bits (Layer 1) to 2.99 bits (e.g., Layer 2, Layer 4). 
*Conclusion:* Expert selection behaves nearly identically to a uniform random distribution from an entropy perspective.

## 8. Layer-Wise Patterns
Summarized in `layer_routing_summary.csv`. Every MoE layer maintains high entropy (>2.94 bits) and tight load balance (Max/Mean ratios rarely exceed 1.05). There are no "specialized" layers where a single expert dominates >50% of the traffic.

## 9. Request-Wise Patterns
Summarized in `request_routing_summary.csv`. Each of the 32 requests perfectly generated 5 tokens across 32 layers. The entropy at the individual request level remains robustly high.

## 10. Main Observations
1. **Execution vs. Routing Decoupling:** Modifying the processor execution paradigm (PE, ET) has zero impact on what the router computes, validating that the simulator strictly preserves algorithmic integrity.
2. **Aggressive Load Balancing:** The model inherently load-balances across all 8 experts exceptionally well (Max/Mean = 1.038). This implies that dynamic, coarse-grained load balancers (like PE) are highly effective because they don't have to deal with massive transient skew.
3. **Low Locality:** Both temporal and spatial transitions are near-uniform. Software techniques that rely on routing predictability (e.g., expert prefetching based on previous tokens or layers) will perform poorly on this architecture.

## 11. Limitations
- The sample size (32 requests × 5 iterations) is relatively small for deeply statistical transition matrices.
- The workload is purely decode. Prefill workloads (which route thousands of tokens simultaneously) might exhibit different spatial batching properties not captured here.
- The baseline prediction accuracy (25.9%) suggests a slight skew towards specific experts that could be highly sequence-dependent.

## Phase G Status
- **Phase G PASS/FAIL:** **PASS**
- **Ready for next phase:** **YES**
