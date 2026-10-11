# A4 Run Manifest

- **Git Commit**: 419252761fbdb95b789778a02256d458a5537ec7
- **Configuration Path**: reproduction_A100/configs/a100_duplex.yaml
- **Effective Configuration Diff**: processor_type = GPU+LOGIC
- **Model**: Mixtral 47B
- **Hardware**: A100 (80GB HBM, 312 TFLOPS, NVLink Gen 3)
- **Workload**: Decode-only (input=2048, batch=32)
- **Warmup**: 10000 iterations
- **Measured Iterations**: 5
- **Output Location**: reproduction_A100/results/a100_duplex/
- **Validation Status**: Completed
