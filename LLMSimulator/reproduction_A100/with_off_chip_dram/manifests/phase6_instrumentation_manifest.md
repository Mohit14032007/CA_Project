# Phase 6 Instrumentation Manifest

## Goal
The objective of Phase 6 was to establish robust simulator-level instrumentation capable of distinguishing and correlating the temporal footprint of off-chip memory access (`T_load`) from the resulting execution delay (`T_compute`) during the `ExpertFFN` GPU evaluation phase.

## Instrumentation Additions

### Trace Instrumentation Extensions
- Modified `src/module/timeboard.cpp` (`TimeStamp::writeProcessorTrace`) to dump a structured processor execution trace mapping exact logical operations (`name`) and underlying processor type (`GPU`, `LOGIC`, `PIM`) to simulator timestamps (`start_time`, `end_time`, `duration`).
- Modified `src/scheduler/sequence.cpp` to export an exact logical mapping of `(request_id, token_id, layer_id, topk_rank, expert_id)` routing decisions to an output CSV (`expert_routing.csv`).

### Data Aggregation and Analysis
- Created Python analysis tool (`merge_phase6.py`) to systematically combine Phase-5 expert loads (`expert_load.csv`) with the Phase-6 generated `processor_trace.csv`.
- This ensures load intervals align perfectly with GPU computation slices.

## CSV Output Schemas

### `processor_trace.csv`
- `device_id` (int)
- `layer_id` (int)
- `operation` (string)
- `processor` (string)
- `start_time` (float)
- `end_time` (float)
- `duration` (float)

### `expert_performance.csv` (Merged Result)
- `iteration` (int)
- `layer_id` (int)
- `expert_id` (int)
- `expert_size_bytes` (int)
- `cache_hit_or_miss` (string)
- `load_duration_ns` (float)
- `compute_start_ns` (float)
- `compute_end_ns` (float)
- `compute_duration_ns` (float)
