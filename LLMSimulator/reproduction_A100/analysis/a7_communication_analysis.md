# Communication Analysis

A3 -> A4: Communication overhead increased due to Duplex offloading data movement requirements.
A4 -> A5: Communication duration is identical (9,121,656 ns).
A5 -> A6: Total communication decreased despite adding `moe_all_reduce_for_e_tp`. This indicates ET tensor distribution naturally reduced other baseline communication volumes per device, more than offsetting the 2.6M ns ET overhead.
