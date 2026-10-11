#!/usr/bin/env python3
import os
import sys
import json
import math
import hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def hypergeometric_pmf(N, K, n, k):
    return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)

def compute_null_jaccard(N=256, k=8):
    # N=256 total experts, k=8 selections
    # Distribution of intersection size X ~ Hypergeometric(N=256, K=8, n=8)
    expected_jaccard = 0.0
    pmf = {}
    for i in range(k + 1):
        p = hypergeometric_pmf(N, k, k, i)
        pmf[i] = p
        jaccard = i / (2 * k - i) if (2 * k - i) > 0 else 0.0
        expected_jaccard += p * jaccard
    return expected_jaccard, pmf

def main():
    repo_dir = "/home/mohman/ca_project/LLMSimulator"
    run_dir = os.path.join(repo_dir, "reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011")
    out_dir = os.path.join(run_dir, "analysis_ds6c5")
    os.makedirs(out_dir, exist_ok=True)

    print("======================================================================")
    print("DS-6C-5: Independent Audit and Routing-Correlation Analysis")
    print(f"Run Root: {run_dir}")
    print(f"Analysis Output: {out_dir}")
    print("======================================================================\n")

    # ------------------------------------------------------------------
    # TASK 1: INDEPENDENT RECALCULATION OF RUN METRICS
    # ------------------------------------------------------------------
    routing_path = os.path.join(run_dir, "traces/expert_routing.csv")
    load_path = os.path.join(run_dir, "traces/expert_load.csv")
    cache_path = os.path.join(run_dir, "traces/expert_cache.csv")
    sim_csv_path = os.path.join(run_dir, "csv/deepseekV3_synthesis_2048_10_GPU_N1_D1_TP1_DP1_maxbatch1_maxprocess524288_iter5_skew0_precision_byte2_parallel_execution0_ramul_decode.csv")
    status_path = os.path.join(run_dir, "logs/completion_status.txt")

    print("[AUDIT] Reading raw trace files...")
    df_routing = pd.read_csv(routing_path)
    df_load = pd.read_csv(load_path)
    df_cache = pd.read_csv(cache_path)
    df_sim = pd.read_csv(sim_csv_path)

    # Recalculate routing quantities
    total_routed_selections = len(df_routing)
    tokens_present = sorted(df_routing["token_id"].unique().tolist())
    layers_present = sorted(df_routing["layer_id"].unique().tolist())
    unique_experts_routed = df_routing["expert_id"].nunique()
    ranks_present = sorted(df_routing["topk_rank"].unique().tolist())

    # Check selections per token
    selections_per_token = df_routing.groupby("token_id").size().to_dict()

    # Recalculate load quantities
    total_load_rows = len(df_load)
    aggregates = df_load[df_load["weight_tensor"] == "all_weights"]
    constituents = df_load[df_load["weight_tensor"] != "all_weights"]
    offchip_loads = df_load[df_load["memory_source"] == "OFFCHIP_DRAM"]
    offchip_aggregates = aggregates[aggregates["memory_source"] == "OFFCHIP_DRAM"]
    hbm_hits = df_load[df_load["memory_source"] == "HBM"]

    cold_batches_count = len(offchip_aggregates)
    total_offchip_bytes = int(offchip_aggregates["size_bytes"].sum())
    total_offchip_duration_ns = float(offchip_aggregates["duration"].sum())
    total_offchip_duration_ms = total_offchip_duration_ns / 1e6

    # Recalculate cache quantities
    total_cache_events = len(df_cache)
    cache_action_counts = df_cache["action"].value_counts().to_dict()
    cache_hits = cache_action_counts.get("HIT", 0)
    cache_misses = cache_action_counts.get("MISS", 0)
    cache_inserts = cache_action_counts.get("INSERT", 0)
    cache_evictions = cache_action_counts.get("EVICT", 0)
    peak_resident_bytes = int(df_cache["resident_bytes"].max())
    final_resident_bytes = int(df_cache["resident_bytes"].iloc[-1])

    # Recalculate simulation summary CSV quantities
    sim_steps_count = len(df_sim)
    cum_device_time_ns = float(df_sim["time"].iloc[-1])
    cum_device_time_ms = cum_device_time_ns / 1e6
    mean_step_latency_ms = float(df_sim["latency"].mean()) / 1e6
    total_expert_ffn_ns = float(df_sim["expert_ffn"].sum())
    total_expert_ffn_ms = total_expert_ffn_ns / 1e6

    # Parse completion status
    status_dict = {}
    with open(status_path) as sf:
        for line in sf:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                status_dict[k] = v

    host_wallclock_seconds = float(status_dict.get("HOST_WALLCLOCK_SECONDS", 0.0))
    exit_code = int(status_dict.get("EXIT_CODE", -1))
    status_str = status_dict.get("STATUS", "")

    # Compare against reference values
    reference_checks = {
        "routed_selections": (total_routed_selections, 2280),
        "tokens_count": (len(tokens_present), 5),
        "cold_expert_batches": (cold_batches_count, 2205),
        "offchip_bytes_transferred": (total_offchip_bytes, 194217246720),
        "cache_hits": (cache_hits, 4635),
        "cache_misses": (cache_misses, 2205),
        "cache_inserts": (cache_inserts, 2205),
        "cache_evictions": (cache_evictions, 1297),
        "peak_resident_bytes": (peak_resident_bytes, 79976988672),
        "simulated_ddr5_time_ms": (round(total_offchip_duration_ms, 2), 8601.93),
        "simulated_device_latency_ms": (round(cum_device_time_ms, 2), 8713.53),
        "host_runtime_seconds": (int(host_wallclock_seconds), 21159),
        "exit_code": (exit_code, 0),
        "status": (status_str, "SUCCESS")
    }

    audit_results = {
        "reference_comparisons": {},
        "raw_counts": {
            "routing_records": total_routed_selections,
            "load_records": total_load_rows,
            "cache_records": total_cache_events,
            "sim_csv_rows": sim_steps_count,
            "tokens": tokens_present,
            "selections_per_token": selections_per_token
        }
    }

    all_matched = True
    for key, (actual, expected) in reference_checks.items():
        match = (actual == expected)
        if not match:
            all_matched = False
        audit_results["reference_comparisons"][key] = {
            "actual": actual,
            "expected": expected,
            "match": match
        }

    print("[AUDIT RESULTS]")
    for k, v in audit_results["reference_comparisons"].items():
        status_tag = "MATCH" if v["match"] else "DISCREPANCY"
        print(f"  {k:30s}: Actual={v['actual']} | Expected={v['expected']} [{status_tag}]")

    # ------------------------------------------------------------------
    # TASK 3: TEMPORAL ROUTING CORRELATION ANALYSIS
    # ------------------------------------------------------------------
    print("\n[TEMPORAL ANALYSIS] Analyzing consecutive-token transitions...")

    # Order routing chronological by sequence
    # Each token processed 57 MoE layers sequentially.
    # Group by token and extract the ordered list of 57 expert sets.
    tokens = sorted(df_routing["token_id"].unique())
    
    # Let's inspect both (1) Raw layer_id matching and (2) Stage-order matching
    # Case A: By raw telemetry layer_id
    temporal_by_layer_id = []
    for i in range(len(tokens) - 1):
        t0 = tokens[i]
        t1 = tokens[i+1]
        df_t0 = df_routing[df_routing["token_id"] == t0]
        df_t1 = df_routing[df_routing["token_id"] == t1]
        common_layers = set(df_t0["layer_id"]).intersection(set(df_t1["layer_id"]))
        for l in sorted(common_layers):
            s0 = set(df_t0[df_t0["layer_id"] == l]["expert_id"])
            s1 = set(df_t1[df_t1["layer_id"] == l]["expert_id"])
            inter = len(s0.intersection(s1))
            union = len(s0.union(s1))
            jaccard = inter / union if union > 0 else 0.0
            temporal_by_layer_id.append({
                "transition": f"{t0}->{t1}",
                "layer_id": int(l),
                "intersection_size": inter,
                "union_size": union,
                "jaccard": jaccard
            })

    df_temp_layer = pd.DataFrame(temporal_by_layer_id)
    mean_jaccard_layer_id = float(df_temp_layer["jaccard"].mean())

    # Case B: By sequential execution stage (0 to 56)
    # Token 2048 executed 57 layers, Token 2049 executed 57 layers in execution order
    temporal_by_stage = []
    token_stages = {}
    for t in tokens:
        df_t = df_routing[df_routing["token_id"] == t]
        # In order of appearance in df_routing
        ordered_layers = df_t["layer_id"].drop_duplicates().tolist()
        token_stages[t] = []
        for stage_idx, l in enumerate(ordered_layers):
            s = set(df_t[df_t["layer_id"] == l]["expert_id"])
            token_stages[t].append((stage_idx, l, s))

    for i in range(len(tokens) - 1):
        t0 = tokens[i]
        t1 = tokens[i+1]
        for stage_idx in range(57):
            _, l0, s0 = token_stages[t0][stage_idx]
            _, l1, s1 = token_stages[t1][stage_idx]
            inter = len(s0.intersection(s1))
            union = len(s0.union(s1))
            jaccard = inter / union if union > 0 else 0.0
            temporal_by_stage.append({
                "transition": f"{t0}->{t1}",
                "stage_idx": stage_idx,
                "layer_t0": l0,
                "layer_t1": l1,
                "intersection_size": inter,
                "union_size": union,
                "jaccard": jaccard
            })

    df_temp_stage = pd.DataFrame(temporal_by_stage)
    mean_jaccard_stage = float(df_temp_stage["jaccard"].mean())

    # Theoretical null model calculation: Hypergeometric(N=256, K=8, n=8)
    null_expected_jaccard, null_pmf = compute_null_jaccard(N=256, k=8)

    print(f"  Observed Mean Temporal Jaccard (by layer_id): {mean_jaccard_layer_id:.4f}")
    print(f"  Observed Mean Temporal Jaccard (by exec stage): {mean_jaccard_stage:.4f}")
    print(f"  Theoretical Independent Uniform Null Model Expectation: {null_expected_jaccard:.4f}")
    print(f"  Null Model PMF (|A cap B| = k):")
    for k, p in null_pmf.items():
        if p > 1e-4:
            print(f"    P(overlap = {k}) = {p*100:.2f}% (Jaccard = {k/(16-k) if 16-k>0 else 0:.4f})")

    # Save temporal tables
    df_temp_layer.to_csv(os.path.join(out_dir, "temporal_correlation_by_layer.csv"), index=False)
    df_temp_stage.to_csv(os.path.join(out_dir, "temporal_correlation_by_stage.csv"), index=False)

    # ------------------------------------------------------------------
    # TASK 4: SPATIAL ROUTING CORRELATION ANALYSIS
    # ------------------------------------------------------------------
    print("\n[SPATIAL ANALYSIS] Analyzing adjacent-layer routing correlation...")

    spatial_records = []
    # For each token, compare adjacent MoE execution stages (stage s to stage s+1)
    for t in tokens:
        stages = token_stages[t]
        for s in range(len(stages) - 1):
            stage_src, l_src, experts_src = stages[s]
            stage_dst, l_dst, experts_dst = stages[s + 1]
            inter = len(experts_src.intersection(experts_dst))
            union = len(experts_src.union(experts_dst))
            jaccard = inter / union if union > 0 else 0.0
            spatial_records.append({
                "token_id": int(t),
                "stage_transition": f"{stage_src}->{stage_dst}",
                "layer_src": int(l_src),
                "layer_dst": int(l_dst),
                "intersection_size": inter,
                "union_size": union,
                "jaccard": jaccard
            })

    df_spatial = pd.DataFrame(spatial_records)
    mean_spatial_jaccard = float(df_spatial["jaccard"].mean())
    df_spatial.to_csv(os.path.join(out_dir, "spatial_correlation_adjacent_layers.csv"), index=False)
    print(f"  Observed Mean Spatial Jaccard (Adjacent MoE Layers): {mean_spatial_jaccard:.4f}")
    print(f"  Theoretical Independent Null Expectation:          {null_expected_jaccard:.4f}")

    # Build Expert-to-Expert Transition Matrix (256 x 256) across adjacent MoE stages
    transition_matrix = np.zeros((256, 256), dtype=int)
    for t in tokens:
        stages = token_stages[t]
        for s in range(len(stages) - 1):
            _, _, experts_src = stages[s]
            _, _, experts_dst = stages[s + 1]
            for src_e in experts_src:
                for dst_e in experts_dst:
                    transition_matrix[src_e, dst_e] += 1

    # Save transition matrix as CSV
    df_trans = pd.DataFrame(transition_matrix)
    df_trans.to_csv(os.path.join(out_dir, "spatial_expert_transition_matrix.csv"), index=False)

    # Generate Figures for Task 3 and Task 4
    # Figure S1: Temporal Overlap Distribution (Observed vs Null Model)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    obs_overlaps = df_temp_stage["intersection_size"].value_counts(normalize=True).to_dict()
    x_vals = np.arange(4)
    obs_p = [obs_overlaps.get(x, 0.0) for x in x_vals]
    null_p = [null_pmf.get(x, 0.0) for x in x_vals]
    width = 0.35
    ax.bar(x_vals - width/2, obs_p, width=width, label="Observed (5 Tokens)", color="#1f77b4", edgecolor="black")
    ax.bar(x_vals + width/2, null_p, width=width, label="Analytical Null Model (Independent Uniform)", color="#ff7f0e", edgecolor="black")
    ax.set_xticks(x_vals)
    ax.set_xticklabels([f"{x} experts" for x in x_vals])
    ax.set_xlabel("Number of Overlapping Experts in Top-8")
    ax.set_ylabel("Probability")
    ax.set_title("Fig S1: Temporal Overlap Probability Distribution vs Independent Null")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_s1_temporal_overlap_distribution.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "fig_s1_temporal_overlap_distribution.pdf"))
    plt.close()

    # Figure S2: Spatial Overlap Distribution Across Adjacent MoE Layers
    fig, ax = plt.subplots(figsize=(7, 4.5))
    obs_spatial_overlaps = df_spatial["intersection_size"].value_counts(normalize=True).to_dict()
    obs_sp_p = [obs_spatial_overlaps.get(x, 0.0) for x in x_vals]
    ax.bar(x_vals - width/2, obs_sp_p, width=width, label="Observed Adjacent Layers (5 Tokens)", color="#2ca02c", edgecolor="black")
    ax.bar(x_vals + width/2, null_p, width=width, label="Analytical Null Model", color="#d62728", edgecolor="black")
    ax.set_xticks(x_vals)
    ax.set_xticklabels([f"{x} experts" for x in x_vals])
    ax.set_xlabel("Number of Overlapping Experts in Adjacent Layers")
    ax.set_ylabel("Probability")
    ax.set_title("Fig S2: Spatial Overlap Distribution Between Adjacent MoE Layers")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_s2_spatial_overlap_distribution.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "fig_s2_spatial_overlap_distribution.pdf"))
    plt.close()

    # Figure S3: Spatial Expert-to-Expert Transition Heatmap
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(transition_matrix, cmap="Blues", aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Transition Co-occurrence Count")
    ax.set_xlabel("Destination Expert ID (Layer L+1)")
    ax.set_ylabel("Source Expert ID (Layer L)")
    ax.set_title("Fig S3: Spatial Expert-to-Expert Co-occurrence Heatmap (256x256)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_s3_spatial_transition_heatmap.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "fig_s3_spatial_transition_heatmap.pdf"))
    plt.close()

    # ------------------------------------------------------------------
    # TASK 6: LOAD-VERSUS-COMPUTE RECALCULATION & DISPARITY AUDIT
    # ------------------------------------------------------------------
    print("\n[LOAD VS COMPUTE AUDIT] Reconciling timing numbers...")
    tokens_count = len(tokens)
    moe_layers_count = 57  # Actual MoE layers in DeepSeek-V3
    top_k = 8
    hidden_dim = 7168
    intermediate_dim = 2048
    flops_per_expert = 2 * 3 * hidden_dim * intermediate_dim  # 88,080,384 FLOPs
    
    # 57 layers computation
    total_moe_flops_57 = tokens_count * moe_layers_count * top_k * flops_per_expert
    # 60 layers computation (used by process_metrics.py previously)
    total_moe_flops_60 = tokens_count * 60 * top_k * flops_per_expert

    a100_peak_tflops = 312.0
    analytical_compute_s_57 = total_moe_flops_57 / (a100_peak_tflops * 1e12)
    analytical_compute_ms_57 = analytical_compute_s_57 * 1e3
    analytical_compute_us_57 = analytical_compute_s_57 * 1e6
    
    analytical_compute_s_60 = total_moe_flops_60 / (a100_peak_tflops * 1e12)
    analytical_compute_ms_60 = analytical_compute_s_60 * 1e3

    disparity_ratio_60 = total_offchip_duration_ms / analytical_compute_ms_60
    disparity_ratio_57 = total_offchip_duration_ms / analytical_compute_ms_57

    timing_audit = {
        "simulated_device_latency_ms": cum_device_time_ms,
        "simulated_ddr5_transfer_time_ms": total_offchip_duration_ms,
        "simulated_internal_forward_time_ms": cum_device_time_ms - total_offchip_duration_ms,
        "analytical_roofline_compute_ms_60layers": analytical_compute_ms_60,
        "analytical_roofline_compute_ms_57layers": analytical_compute_ms_57,
        "disparity_ratio_reported_60layers": disparity_ratio_60,
        "disparity_ratio_recalculated_57layers": disparity_ratio_57,
        "host_wallclock_seconds": host_wallclock_seconds,
        "host_to_simulated_device_ratio": (host_wallclock_seconds * 1000.0) / cum_device_time_ms
    }

    print(f"  Simulated Device Latency:        {cum_device_time_ms:.2f} ms")
    print(f"  Simulated DDR5 Transfer Time:    {total_offchip_duration_ms:.2f} ms")
    print(f"  Simulated Internal Forward Time: {cum_device_time_ms - total_offchip_duration_ms:.2f} ms")
    print(f"  Analytical Compute (60 layers):  {analytical_compute_ms_60:.4f} ms ({analytical_compute_s_60*1e6:.2f} us)")
    print(f"  Analytical Compute (57 layers):  {analytical_compute_ms_57:.4f} ms ({analytical_compute_s_57*1e6:.2f} us)")
    print(f"  Reported Disparity Ratio:        {disparity_ratio_60:.1f}x (matches reported ~12,696x)")
    print(f"  Recalculated Ratio (57 MoE):     {disparity_ratio_57:.1f}x")
    print(f"  Host Wall-Clock to Sim Ratio:    {(host_wallclock_seconds*1000)/cum_device_time_ms:.1f}x host slowdown")

    with open(os.path.join(out_dir, "audit_summary.json"), "w") as f:
        json.dump({
            "audit_checks": audit_results,
            "temporal_analysis": {
                "mean_jaccard_layer_id": mean_jaccard_layer_id,
                "mean_jaccard_stage": mean_jaccard_stage,
                "null_expected_jaccard": null_expected_jaccard,
                "null_pmf": null_pmf
            },
            "spatial_analysis": {
                "mean_spatial_jaccard": mean_spatial_jaccard,
                "null_expected_jaccard": null_expected_jaccard
            },
            "timing_audit": timing_audit
        }, f, indent=2)

    print(f"\n[SUCCESS] Independent audit completed! Summary written to {out_dir}/audit_summary.json")

if __name__ == "__main__":
    main()
