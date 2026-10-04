# A100 Hardware Manifest

The simulator uses the explicit `A100` struct defined in `src/hardware/hardware_config.h`.

- **GPU**: A100-class
- **Node**: 1
- **Devices**: 4
- **Memory**: 80.0 GB HBM/device
- **Compute Peak (FP16)**: 312.0 TFLOPS
- **HBM Bandwidth**: 2.039 TB/s
- **Inter-device Bandwidth**: 150.0 GB/s (NVLink 3)
- **Inter-device Latency**: 3.0 us

## H100 Reference Comparison
*(Note: These values are strictly for historical contrast and are NOT used in the A100 simulations)*
- H100 Compute Peak: 989.4 TFLOPS
- H100 HBM Bandwidth: 3.352 TB/s
- H100 Inter-device Bandwidth: 450.0 GB/s (NVLink 4)
- H100 Inter-device Latency: 0.8 us
