import csv
from collections import defaultdict
import statistics
import os

b1_load_csv = "reproduction_A100/with_off_chip_dram/results/b1_full/traces/expert_load.csv"
b1_compute_csv = "reproduction_A100/with_off_chip_dram/results/b1_full/traces/processor_trace.csv"

out_dir = "reproduction_A100/with_off_chip_dram/results/phase8_memory_analysis/analysis"

# Part D: Expert weight size
# Read expert_load.csv to find W1, W3, W2 sizes
tensor_sizes = {}
total_expert_size = 0
loads = []

with open(b1_load_csv, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        loads.append(row)
        size = int(row['size_bytes'])
        t = row['weight_tensor']
        if t not in tensor_sizes:
            tensor_sizes[t] = size

total_expert_size = sum(tensor_sizes.values())
bytes_val = total_expert_size
mb_val = bytes_val / 1e6
mib_val = bytes_val / (1024**2)

total_footprint_bytes = 256 * total_expert_size
total_footprint_gb = total_footprint_bytes / 1e9
total_footprint_gib = total_footprint_bytes / (1024**3)

print("=== PART D: EXPERT WEIGHT SIZE ===")
print(f"Tensor Sizes: {tensor_sizes}")
print(f"Total Expert Size: {bytes_val} bytes ({mb_val:.2f} MB, {mib_val:.2f} MiB)")
print(f"Total 256 Expert Footprint: {total_footprint_gb:.2f} GB ({total_footprint_gib:.2f} GiB)")

# Part E: Cold Load Analysis
total_tensor_loads = len(loads)
cold_miss_events = [l for l in loads if l['cache_hit_or_miss'] == 'miss']
hit_events = [l for l in loads if l['cache_hit_or_miss'] == 'hit']

unique_experts_loaded = set((l['layer_id'], l['expert_id']) for l in loads)

# Calculate total load time (sum of durations of all misses)
total_ddr5_load_time = sum(float(l['duration']) for l in cold_miss_events)

# Expert level load times (sum of tensor load times for the expert)
expert_miss_times = defaultdict(float)
expert_hit_times = defaultdict(float)

for l in loads:
    key = (l['layer_id'], l['expert_id'], l['request_id']) # group by request_id if they form one load, but actually request_id is per tensor.
    # We should group by the first occurrence for each expert.
    pass

# We can just sum up the 3 tensor durations per unique miss event.
# Actually, it's easier to just compute per-tensor and per-expert.
miss_durations_per_expert = defaultdict(float)
for l in cold_miss_events:
    miss_durations_per_expert[(l['layer_id'], l['expert_id'])] += float(l['duration'])

miss_times_list = list(miss_durations_per_expert.values())

mean_load = statistics.mean(miss_times_list)
median_load = statistics.median(miss_times_list)
min_load = min(miss_times_list)
max_load = max(miss_times_list)
std_load = statistics.stdev(miss_times_list) if len(miss_times_list) > 1 else 0

print("\n=== PART E: COLD-LOAD ANALYSIS ===")
print(f"Total Tensor Load Events: {total_tensor_loads}")
print(f"Number of Tensor Miss Events: {len(cold_miss_events)}")
print(f"Number of Tensor Hit Events: {len(hit_events)}")
print(f"Number of Expert Miss Events (unique experts): {len(miss_durations_per_expert)}")
print(f"Number of Unique Experts Loaded: {len(unique_experts_loaded)}")
print(f"Total DDR5 Load Time (ns): {total_ddr5_load_time}")
print(f"Mean Expert Load Time (ns): {mean_load}")
print(f"Median Expert Load Time (ns): {median_load}")
print(f"Min Expert Load Time (ns): {min_load}")
print(f"Max Expert Load Time (ns): {max_load}")
print(f"StdDev Expert Load Time (ns): {std_load}")

# Part F: T_load vs T_compute
# T_compute from processor_trace.csv
computes = []
with open(b1_compute_csv, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['operation'].startswith('expert_FFN_'):
            computes.append(float(row['duration']))

t_compute_mean = statistics.mean(computes)
t_compute_median = statistics.median(computes)
t_compute_min = min(computes)
t_compute_max = max(computes)

ratio_mean = mean_load / t_compute_mean
ratio_median = median_load / t_compute_median

print("\n=== PART F: T_LOAD VS T_COMPUTE ===")
print(f"Mean T_compute (ns): {t_compute_mean}")
print(f"Median T_compute (ns): {t_compute_median}")
print(f"Ratio (Mean T_load / Mean T_compute): {ratio_mean:.2f}x")
print(f"Ratio (Median T_load / Median T_compute): {ratio_median:.2f}x")

# Part G: Theoretical DDR5 Transfer Time
# Ramulator DDR5 config: 3200 MT/s, 2 channels, 16 pins per channel?
# Wait, DDR5_32Gb_x16 means x16 data width. DDR5 typically has two 32-bit subchannels per channel.
# If channel: 2, and x16, total width is 2 channels * 32 bits? No, 2 channels * 16 bits?
# Actually, a standard DDR5 DIMM is 64-bit data width (two 32-bit subchannels).
# With 3200 MT/s and 64-bit (8 bytes), bandwidth = 3200 * 10^6 * 8 = 25.6 GB/s per channel.
# For 2 channels, it is 51.2 GB/s.
bw_gbs = 51.2 # GB/s
# Let's calculate transfer time for 352MB
t_theoretical = (bytes_val / 1e9) / bw_gbs # seconds
t_theoretical_ns = t_theoretical * 1e9

print("\n=== PART G: THEORETICAL DDR5 TRANSFER TIME ===")
print(f"Assumed Theoretical BW (GB/s): {bw_gbs}")
print(f"Theoretical Transfer Time (ns): {t_theoretical_ns}")
print(f"Simulated / Theoretical Ratio: {mean_load / t_theoretical_ns:.2f}x")

# Save to CSV
with open(os.path.join(out_dir, "memory_summary.csv"), "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value", "Unit"])
    writer.writerow(["Total Expert Size", bytes_val, "bytes"])
    writer.writerow(["Total 256 Expert Footprint", total_footprint_gb, "GB"])
    writer.writerow(["Tensor Miss Events", len(cold_miss_events), "events"])
    writer.writerow(["Tensor Hit Events", len(hit_events), "events"])
    writer.writerow(["Expert Miss Events", len(miss_durations_per_expert), "events"])
    writer.writerow(["Mean Expert Load Time", mean_load, "ns"])
    writer.writerow(["Mean Expert Compute Time", t_compute_mean, "ns"])
    writer.writerow(["Ratio Load/Compute", ratio_mean, "x"])
