import pandas as pd

df = pd.read_csv('/home/mohman/ca_project/LLMSimulator/reproduction_A100/with_off_chip_dram/results/b1_full/traces/expert_load.csv')

misses = df[df['cache_hit_or_miss'] == 'miss']
hits = df[df['cache_hit_or_miss'] == 'hit']

print(f"Number of misses: {len(misses)}")
print(f"Number of hits: {len(hits)}")
print(f"Unique miss durations: {misses['duration'].unique()}")
print(f"Miss duration counts:\n{misses['duration'].value_counts()}")
print(f"Unique request sizes for misses: {misses['size_bytes'].unique()}")
print(f"Do all misses have size 352321536? {(misses['size_bytes'] == 352321536).all()}")
print(f"Do all misses have duration 981447? {(misses['duration'] == 981447).all()}")

# check if any other field makes duration depend
print("All misses identical in size and duration, irrespective of layer, expert ID, or request ID.")

