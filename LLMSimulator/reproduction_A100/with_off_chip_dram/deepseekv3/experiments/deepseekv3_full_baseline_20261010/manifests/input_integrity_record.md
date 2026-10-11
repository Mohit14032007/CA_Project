# Input Integrity Record: DeepSeek-V3 Full Baseline Artifacts

**Experiment Directory:** `reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010/`  
**Verification Date:** October 10, 2026  
**Auditor:** DS-5C-12 Automated Telemetry & Results Audit  
**Status:** ALL ARTIFACTS VERIFIED & UNMODIFIED  

---

## 1. Raw Telemetry Artifacts

| Artifact Name | Relative Path | Size (Bytes) | Line Count (incl. Header) | SHA-256 Checksum | Schema / Content Description |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Routing Decisions** | `raw/expert_routing.csv` | 11,355 | 913 | `229fc7522e212f356d2846d79a72b6172b395f3ca9a8f194de8187d0a696e259` | 912 total routing decisions (456 Pass 1 + 456 Pass 2 replay) + 1 header |
| **Load Telemetry** | `raw/expert_load.csv` | 464,179 | 6,385 | `d984dd17a3c412bcbf97c7a0c44bab185bb7f70b4ba06933bd259fa178d28644` | 6,384 memory transfer and aggregate rows + 1 header |
| **Cache Transitions** | `raw/expert_cache.csv` | 72,902 | 3,193 | `3302371500fe0e5cb79c23d4b9e86cce1b107b25482ed042e6b4bdb73d02d181` | 3,192 cache events (MISS, INSERT, HIT) + 1 header |
| **Simulator Status** | `raw/ds5c10_completion_status.txt` | 555 | 22 | `f6e2ff8b133858b9ecaf50c89d33c689429f2a3539146f6b647ffb275b65e859` | Cycle-accurate simulated DDR5 hardware timers & line counts |
| **Runner Status** | `raw/ds5c11_completion_status.txt` | 1,708 | 21 | `6de4d7b226d2cade755555e27b452fbbcbeb7aa9f8ba52d6da7971f1d45f5ae0` | Host wall-clock timers, resource peaks, and environment status |

---

## 2. Execution Logs & Snapshots

| Artifact Name | Relative Path | Size (Bytes) | SHA-256 Checksum | Content Description |
| :--- | :--- | :---: | :--- | :--- |
| **Build Log** | `logs/build.log` | 563 | `ad2a90a5e1283e588431791da9a6a1a23c97bd3f77ebed6da2eca3d1811a3b44` | Full compilation and linking output of `test_deepseekv3_full_baseline` |
| **Run Console Log** | `logs/full_baseline_run.log` | 10,407 | `9641d8519d443a117399d1e1908227ef3eaf530ee8bf3bba982148dc29f3c44a` | Complete stdout/stderr and 60-second progress heartbeats |
| **DDR5 Config** | `config_snapshot/dram_config_DDR5_retry.yaml` | 445 | `7ef030f82d7f654339aa0ab1ff31abce311322c59aee693eb57774e071db9d97` | Frozen DDR5 Ramulator 2 configuration (TraceRecorder disabled) |
| **HBM3 Config** | `config_snapshot/dram_config_HBM3_80GB.yaml` | 519 | `e7ae1fb6802016fba28dd8d2cac439a15c01d2615c2063973eacb628903e79c6` | Frozen HBM3 Ramulator 2 configuration |
| **Checksum Manifest** | `config_snapshot/checksums.sha256` | 407 | `26ff641cb027b5f922736465dd8a9644f128c94627d2c0b056151a6297316315` | Pre-execution hash manifest of active configurations |

---

## 3. Verification Conclusion

All raw files match their execution hashes exactly. No files were truncated, appended, or modified post-simulation. Integrity verification passes unconditionally.
