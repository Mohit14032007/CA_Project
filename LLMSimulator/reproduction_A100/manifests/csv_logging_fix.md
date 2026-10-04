# CSV Logging Fix Validation

## Old Behavior
The main performance loop in `src/hardware/cluster.cpp` (`Cluster::runIterationMixed` and `Cluster::runIterationSumGenSplit`) contained an `exportToCSV()` trigger placed at the *start* of the iteration block. Because of this, the final executed iteration generated metrics that were stored in memory but never dumped to the CSV file, resulting in 4 CSV records despite 5 executed simulation steps.

## New Behavior
The `exportToCSV()` trigger has been safely moved to the end of the loop, and a final catch-all condition `if (!stat_list.empty()) { exportToCSV(...) }` was added immediately preceding the function return.

## Source Location
`src/hardware/cluster.cpp` (modified prior to executing Phase A1).

## Why the fix is safe
- **Preserves all executed iterations**: The final iteration metrics are now correctly flushed to disk.
- **Does not duplicate iterations**: The modulo export frequency logic remains intact.
- **No simulation changes**: The core `run(metadata)` logic, scheduling, timing, and model execution paths were strictly unmodified. Only the file writing sequence was patched.

## Validation
When the A100 experiments are executed, the resulting CSVs in `reproduction_A100/results/csv/` must contain exactly 5 records corresponding precisely to the 5 requested iterations.
