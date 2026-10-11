# A100 Debug Validation

## CSV iteration validation
**PASS**
Evidence: iteration_validation.csv shows exactly 3 rows for all configurations.

## Routing trace validation
**PASS**
Evidence: expert_routing.csv has exactly 1536 records per run with valid expert and layer IDs.

## Processor trace validation
**PASS**
Evidence: processor_trace.csv contains valid chronological records without negative durations and valid device IDs.

## GPU baseline
**PASS**
Evidence: Only GPU processors found in GPU baseline processor_trace.

## Plain Duplex
**PASS**
Evidence: Both GPU and Logic operations exist in Duplex processor trace.

## Duplex + PE
**PASS**
Evidence: PE logic modifies execution flow when bottlenecked.

## Duplex + PE + ET
**PASS**
Evidence: moe_all_reduce_for_e_tp appears in execution trace for ET=4.

## Routing consistency
**PASS**
Evidence: All four configurations produce identical expert_routing.csv traces.

## Log size validation
**PASS**
Evidence: No file exceeds 100 MB.

## Final verdict
**PASS**
A100 experiment pipeline is validated. Safe to proceed to Phase A3 — A100 GPU Baseline.
