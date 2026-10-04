# ET Device Mapping

Expert operations and ET all-reduces were found distributed across the following devices:
[np.int64(0), np.int64(1), np.int64(2), np.int64(3)]
This was established by querying the processor_trace.csv for `moe_all_reduce_for_e_tp` and related operations.
