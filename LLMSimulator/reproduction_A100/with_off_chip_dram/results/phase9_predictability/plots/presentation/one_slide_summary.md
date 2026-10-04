# Phase 9 — What the Synthetic Trace Shows

### Observation 1
Expert frequency is moderately skewed.

### Observation 2
Routing entropy remains high (~0.93 normalized).

### Observation 3
Consecutive-token top-k overlap is low (~24.8% recall).

### Observation 4
Previous-token prediction performs approximately like random sampling.

### Conclusion

    The synthetic workload contains static expert-frequency skew,
    but no meaningful temporal routing predictability.

### Next Step

    Replace synthetic routing with a real LLM/Mixtral routing trace.
