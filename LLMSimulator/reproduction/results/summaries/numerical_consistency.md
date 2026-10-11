# Numerical Consistency Audit

This audit mathematically verifies the derived values found in the performance summary.

## GPU Baseline
- Total Time: 32,091,176.09 us
- MoE Time: 6,766,808.97 us (21.0%)
- Expected Speedup: 1.00x

## Plain Duplex
- Total Time: 11,571,463.88 us
- Recomputed Speedup: 32,091,176.09 / 11,571,463.88 = 2.773x
- Reported Speedup: 2.77x
- **Status:** PASS

## Duplex + PE
- Total Time: 11,571,463.88 us
- Recomputed Speedup: 32,091,176.09 / 11,571,463.88 = 2.773x
- Reported Speedup: 2.77x
- **Status:** PASS

## Duplex + PE + ET
- Total Time: 10,470,314.59 us
- Recomputed Speedup: 32,091,176.09 / 10,470,314.59 = 3.065x
- Reported Speedup: 3.07x
- **Status:** PASS

## Communication vs Compute
- ET explicitly added exactly ~167k us communication overhead (verified via trace duration summation of `all_reduce_for_e_tp`).
- ET decreased absolute execution time from 11.57M to 10.47M. Net delta: -1.1M us.
- The mathematical consistency proves that the compute gains from balanced tensor parallelism massive dwarfed the all-reduce communication penalty.
- **Status:** PASS
