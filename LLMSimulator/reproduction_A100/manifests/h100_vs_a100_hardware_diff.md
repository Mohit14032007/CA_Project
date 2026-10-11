# Hardware Configuration Differences: H100 vs A100

| Parameter | Old H100 Value | New A100 Value | Source File | Reason / Description |
|-----------|----------------|----------------|-------------|----------------------|
| **Device Interconnect Latency** | `0.8 * 1000` ns | `3.0 * 1000` ns | `src/hardware/hardware_config.h` | Represents NVLink Gen 4 (H100) vs NVLink Gen 3 (A100). |
| **Device Interconnect Bandwidth** | `450.0` GB/s | `150.0` GB/s | `src/hardware/hardware_config.h` | NVLink theoretical bandwidth changes dramatically between generations. |
| **Compute Peak FLOPS (FP16)** | `989.4` TFLOPS | `312.0` TFLOPS | `src/hardware/hardware_config.h` | Standard generation-over-generation compute capability difference. |
| **Memory Bandwidth** | `3.352` TB/s | `2.039` TB/s | `src/hardware/hardware_config.h` | Reflects HBM3 (H100) vs HBM2e (A100). |
| **Memory Capacity** | `80.0` GB | `80.0` GB | `src/hardware/hardware_config.h` | Both target the 80 GB variants. |
| **Node Interconnect Latency** | `130.0` ns | `130.0` ns | `src/hardware/hardware_config.h` | Unchanged (both assume ConnectX-7). |
| **Node Interconnect Bandwidth** | `50.0` GB/s | `50.0` GB/s | `src/hardware/hardware_config.h` | Unchanged. |
| **Logic PIM Mutliplier (`logic_x`)** | `4` | `4` | `src/hardware/hardware_config.h` | Same architectural multiplier assumed. |
| **Logic Ops/Byte (`logic_op_b`)** | `8` | `8` | `src/hardware/hardware_config.h` | Same assumption. |

This comparison confirms that changing the parameter to `A100` correctly acts as an isolated hardware swap, modifying critical compute, memory, and interconnect metrics without corrupting the model/workload state.
