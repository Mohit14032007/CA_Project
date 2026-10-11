#!/usr/bin/env python3
import sys
import os
import argparse
import json
import pandas as pd
import numpy as np

def analyze_sequential_results(exp_dir=None, traces_dir=None, processed_dir=None):
    if exp_dir is None and traces_dir is None:
        raise ValueError("Either exp_dir or traces_dir must be provided.")

    if processed_dir is None:
        if exp_dir is not None:
            processed_dir = os.path.join(exp_dir, "processed")
        else:
            processed_dir = os.path.normpath(os.path.join(traces_dir, "..", "processed"))

    if traces_dir is None:
        traces_dir = os.path.join(exp_dir, "traces")
        if not os.path.exists(traces_dir):
            traces_dir = os.path.join(exp_dir, "raw")

    os.makedirs(processed_dir, exist_ok=True)

    routing_path = os.path.join(traces_dir, "expert_routing.csv")
    load_path = os.path.join(traces_dir, "expert_load.csv")
    cache_path = os.path.join(traces_dir, "expert_cache.csv")

    target_label = exp_dir if exp_dir else traces_dir
    print(f"[INFO] Running multi-token sequential analysis for {target_label}...")

    # 1. Temporal Routing Transition Analysis
    temporal_summary = {}
    if os.path.exists(routing_path) and os.path.getsize(routing_path) > 0:
        df_routing = pd.read_csv(routing_path)
        tokens = sorted(df_routing["token_id"].unique())
        layers = sorted(df_routing["layer_id"].unique())
        
        temporal_summary["total_tokens"] = len(tokens)
        temporal_summary["tokens"] = [int(t) for t in tokens]
        
        # Compute Jaccard similarity between adjacent tokens per layer
        jaccard_per_step = []
        if len(tokens) > 1:
            for l in layers:
                df_l = df_routing[df_routing["layer_id"] == l]
                for i in range(len(tokens) - 1):
                    t0 = tokens[i]
                    t1 = tokens[i + 1]
                    s0 = set(df_l[df_l["token_id"] == t0]["expert_id"])
                    s1 = set(df_l[df_l["token_id"] == t1]["expert_id"])
                    
                    inter = len(s0.intersection(s1))
                    union = len(s0.union(s1))
                    jaccard = inter / union if union > 0 else 0.0
                    jaccard_per_step.append({
                        "layer_id": int(l),
                        "token_pair": f"{t0}->{t1}",
                        "jaccard_similarity": jaccard,
                        "overlap_count": inter
                    })
        
        if jaccard_per_step:
            df_jaccard = pd.DataFrame(jaccard_per_step)
            df_jaccard.to_csv(os.path.join(processed_dir, "temporal_routing_jaccard.csv"), index=False)
            temporal_summary["mean_adjacent_token_jaccard"] = float(df_jaccard["jaccard_similarity"].mean())
            print(f"  [TEMPORAL] Mean Jaccard similarity between adjacent tokens: {temporal_summary['mean_adjacent_token_jaccard']:.4f}")
        else:
            temporal_summary["mean_adjacent_token_jaccard"] = 0.0

    # 2. Cache Eviction & Occupancy Analysis
    cache_analysis = {}
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        df_cache = pd.read_csv(cache_path)
        evictions = df_cache[df_cache["action"] == "EVICT"]
        cache_analysis["total_evictions"] = int(len(evictions))
        cache_analysis["evicted_experts_count"] = int(evictions["expert_id"].nunique()) if not evictions.empty else 0
        cache_analysis["peak_resident_bytes"] = int(df_cache["resident_bytes"].max())
        cache_analysis["final_resident_bytes"] = int(df_cache["resident_bytes"].iloc[-1])
        print(f"  [CACHE] Eviction total: {cache_analysis['total_evictions']}, Peak bytes: {cache_analysis['peak_resident_bytes']} B")

    # 3. Save comprehensive analysis summary
    analysis_results = {
        "temporal_routing": temporal_summary,
        "cache_analysis": cache_analysis
    }
    with open(os.path.join(processed_dir, "advanced_analysis.json"), "w") as f:
        json.dump(analysis_results, f, indent=2)

    print(f"[SUCCESS] Advanced sequential analysis saved to {processed_dir}/advanced_analysis.json.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-token sequential decoding analysis.")
    parser.add_argument("--exp-dir", type=str, default=None, help="Experiment directory.")
    parser.add_argument("--traces-dir", type=str, default=None, help="Traces directory.")
    parser.add_argument("--processed-dir", type=str, default=None, help="Processed directory.")
    args = parser.parse_args()

    analyze_sequential_results(exp_dir=args.exp_dir, traces_dir=args.traces_dir, processed_dir=args.processed_dir)
