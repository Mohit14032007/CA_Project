#!/usr/bin/env python3
import sys
import os
import json
import pandas as pd
import numpy as np

def normalize_and_validate_schema(df, filename, required_cols, col_mapping):
    """
    Normalizes known column name variants to canonical names and validates schema.
    Raises ValueError with detailed error message if required columns are missing.
    """
    for col in list(df.columns):
        if col in col_mapping:
            df.rename(columns={col: col_mapping[col]}, inplace=True)
            
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Schema error in file '{filename}': Missing required column(s): {missing}.\n"
            f"Available columns: {list(df.columns)}"
        )
    return df

def analyze_experiment(exp_dir, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(exp_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)

    load_path = os.path.join(exp_dir, "raw", "expert_load.csv")
    if not os.path.exists(load_path):
        load_path = os.path.join(exp_dir, "expert_load.csv")
        
    cache_path = os.path.join(exp_dir, "raw", "expert_cache.csv")
    if not os.path.exists(cache_path):
        cache_path = os.path.join(exp_dir, "expert_cache.csv")

    routing_path = os.path.join(exp_dir, "raw", "expert_routing.csv")
    if not os.path.exists(routing_path):
        routing_path = os.path.join(exp_dir, "expert_routing.csv")

    status_path = os.path.join(exp_dir, "logs", "ds5c10_completion_status.txt")
    if not os.path.exists(status_path):
        status_path = os.path.join(exp_dir, "ds5c8_completion_status.txt")

    print(f"[INFO] Analyzing experiment at {exp_dir}...")

    # 1. Routing Analysis
    routing_summary = {}
    if os.path.exists(routing_path):
        df_routing = pd.read_csv(routing_path)
        df_routing = normalize_and_validate_schema(
            df_routing, routing_path,
            required_cols=["request_id", "token_id", "layer_id", "expert_id"],
            col_mapping={"expert": "expert_id", "layer": "layer_id", "token": "token_id"}
        )
        
        routing_summary["total_routing_events"] = int(len(df_routing))
        routing_summary["unique_experts_routed"] = int(df_routing["expert_id"].nunique())
        routing_summary["layers_routed"] = sorted([int(x) for x in df_routing["layer_id"].unique()])
        routing_summary["tokens_routed"] = sorted([int(x) for x in df_routing["token_id"].unique()])
        
        # Expert frequency table
        freq_df = df_routing["expert_id"].value_counts().reset_index()
        freq_df.columns = ["expert_id", "selection_count"]
        freq_df.to_csv(os.path.join(output_dir, "expert_frequency.csv"), index=False)
        
        # Per layer routing table
        layer_routing = df_routing.groupby("layer_id")["expert_id"].agg(["count", "nunique"]).reset_index()
        layer_routing.columns = ["layer_id", "total_selections", "unique_experts"]
        layer_routing.to_csv(os.path.join(output_dir, "per_layer_routing.csv"), index=False)
        
        print(f"  [ROUTING] Read {len(df_routing)} routing events across layers {routing_summary['layers_routed']}.")
    else:
        print(f"  [ROUTING] Warning: File not found at {routing_path}")

    # 2. Load Telemetry Analysis
    load_summary = {}
    if os.path.exists(load_path):
        df_load = pd.read_csv(load_path)
        df_load = normalize_and_validate_schema(
            df_load, load_path,
            required_cols=[
                "request_id", "layer_id", "expert_id", "weight_tensor",
                "size_bytes", "duration", "memory_source", "cache_hit_or_miss"
            ],
            col_mapping={
                "tensor_name": "weight_tensor",
                "weight_name": "weight_tensor",
                "tensor": "weight_tensor",
                "size": "size_bytes",
                "hit_or_miss": "cache_hit_or_miss",
                "hit_miss": "cache_hit_or_miss"
            }
        )
        
        load_summary["total_load_rows"] = int(len(df_load))
        load_summary["unique_request_ids"] = int(df_load["request_id"].nunique())
        
        # Separate individual tensor rows from aggregates
        tensor_rows = df_load[df_load["weight_tensor"] != "all_weights"]
        agg_rows = df_load[df_load["weight_tensor"] == "all_weights"]
        
        load_summary["tensor_request_rows"] = int(len(tensor_rows))
        load_summary["aggregate_rows"] = int(len(agg_rows))
        
        # Off-chip vs HBM breakdown
        miss_batches = df_load[(df_load["weight_tensor"] == "all_weights") & (df_load["cache_hit_or_miss"] == "miss")]
        hit_batches = df_load[(df_load["weight_tensor"] == "all_weights") & (df_load["cache_hit_or_miss"] == "hit")]
        
        load_summary["offchip_ddr5_batches"] = int(len(miss_batches))
        load_summary["hbm_hit_batches"] = int(len(hit_batches))
        load_summary["total_offchip_bytes"] = int(miss_batches["size_bytes"].sum())
        load_summary["total_simulated_ddr5_duration_ns"] = float(miss_batches["duration"].sum())
        load_summary["total_simulated_ddr5_duration_ms"] = float(miss_batches["duration"].sum() / 1e6)
        
        # Per-layer load metrics
        layer_load = df_load[df_load["weight_tensor"] == "all_weights"].groupby(["layer_id", "cache_hit_or_miss"]).agg(
            batch_count=("request_id", "count"),
            total_bytes=("size_bytes", "sum"),
            total_duration_ns=("duration", "sum")
        ).reset_index()
        layer_load.to_csv(os.path.join(output_dir, "per_layer_load_summary.csv"), index=False)

        # Per-expert duration distribution
        if len(miss_batches) > 0:
            dur_stats = miss_batches["duration"].describe().to_dict()
            load_summary["duration_stats_ns"] = {k: float(v) for k, v in dur_stats.items()}

        print(f"  [LOAD] Read {len(df_load)} load rows:")
        print(f"         - Individual tensor rows: {len(tensor_rows)}")
        print(f"         - 'all_weights' aggregate rows: {len(agg_rows)}")
        print(f"         - Off-chip DDR5 batches (misses): {len(miss_batches)}")
        print(f"         - Warm HBM hits: {len(hit_batches)}")
    else:
        print(f"  [LOAD] Warning: File not found at {load_path}")

    # 3. Cache Telemetry Analysis
    cache_summary = {}
    if os.path.exists(cache_path):
        df_cache = pd.read_csv(cache_path)
        df_cache = normalize_and_validate_schema(
            df_cache, cache_path,
            required_cols=["layer_id", "expert_id", "action", "resident_bytes"],
            col_mapping={"event": "action", "status": "action"}
        )
        
        cache_summary["total_cache_events"] = int(len(df_cache))
        action_counts = df_cache["action"].value_counts().to_dict()
        cache_summary["actions"] = {str(k): int(v) for k, v in action_counts.items()}
        cache_summary["final_resident_bytes"] = int(df_cache["resident_bytes"].iloc[-1]) if len(df_cache) > 0 else 0
        cache_summary["peak_resident_bytes"] = int(df_cache["resident_bytes"].max()) if len(df_cache) > 0 else 0
        
        # Cache timeline summary
        timeline_df = df_cache[["layer_id", "expert_id", "action", "resident_bytes"]].copy()
        timeline_df["event_id"] = np.arange(len(timeline_df))
        timeline_df.to_csv(os.path.join(output_dir, "cache_timeline.csv"), index=False)
        
        print(f"  [CACHE] Read {len(df_cache)} cache events:")
        for act, cnt in action_counts.items():
            print(f"          - {act}: {cnt}")
        print(f"          - Final resident bytes: {cache_summary['final_resident_bytes']} B")
    else:
        print(f"  [CACHE] Warning: File not found at {cache_path}")

    # 4. Status File Integration
    status_summary = {}
    if os.path.exists(status_path):
        with open(status_path, "r") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    status_summary[k.strip()] = v.strip()

    metrics = {
        "status": status_summary,
        "routing": routing_summary,
        "load": load_summary,
        "cache": cache_summary
    }

    metrics_path = os.path.join(output_dir, "summary_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SUCCESS] Metrics summary written to {metrics_path}")
    return metrics

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "reproduction_A100/with_off_chip_dram/deepseekv3/results/ds5c10_stage1"
    analyze_experiment(target_dir)
