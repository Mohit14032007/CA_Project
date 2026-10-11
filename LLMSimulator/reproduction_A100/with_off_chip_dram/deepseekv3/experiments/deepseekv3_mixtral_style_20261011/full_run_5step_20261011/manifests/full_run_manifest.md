# DS-6C-4: Controlled Full-Depth DeepSeek-V3 5-Step Run Manifest

**Run Identifier:** `full_run_5step_20261011`  
**Execution Date:** 2026-10-11  
**Completion Status:** `SUCCESS` (Exit Code `0`, Termination Reason: `NORMAL`)  
**Repository Path:** `/home/mohman/ca_project/LLMSimulator`  
**Source Commit (HEAD):** `f75d362ae157f81ef2d3bd4055c9b22e34882a9f`  
**Dedicated Run Root:** `reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/`  

---

## 1. Environment & Hardware Context

- **Host Operating System:** Linux (Kernel 6.6.137+ Sun Oct 11 2026)
- **Host Memory:** 7,794 MB Total RAM (Available at start: 6,401 MB; Minimum observed during run: 4,741 MB)
- **Host Storage:** 1,007 GB ext4 filesystem (Available at start: 929,789 MB; Available at end: 929,787 MB)
- **Peak Process RSS:** 1,084,719,104 bytes (~1,059.2 MB / 1.08 GB)
- **Simulated Target Architecture:** Single-Device NVIDIA A100-SXM4-80GB Reference Accelerator
  - Compute Peak: 312.0 TFLOP/s FP16 Dense Tensor Core
  - Local Memory (HBM3): 80,000,000,000 bytes (~74.51 GiB / 80 GB) LRU Expert Cache Tier
  - Interconnect: Single Device (`num_node: 1`, `num_device: 1`), Logic-PIM and TP disabled
  - Backing Memory: Off-Chip DDR5 DRAM with expanded 10,485,760 rows (2.75 TB / 2.5 TiB addressable)

---

## 2. Source & Executable Provenance

| Component | Path | SHA-256 Checksum |
|:---|:---|:---|
| **Simulator Binary** | `build/run_deepseekv3_mixtral_style` | `819a752527b536224b1467a297d72158281da5c7d5e54e7f792a0d521c4e1093` |
| **Run Configuration** | `configs/deepseekv3_b1_full_run_5step.yaml` | `1bdcaee0f058c448a69f94e256998dc44da6acca5fdbd01b74db41b3a3954f6d` |
| **DDR5 Memory Config** | `config_snapshot/dram_config_DDR5_retry.yaml` | `7ef030f82d7f654339aa0ab1ff31abce311322c59aee693eb57774e071db9d97` |
| **HBM3 Memory Config** | `config_snapshot/dram_config_HBM3_80GB.yaml` | `e7ae1fb6802016fba28dd8d2cac439a15c01d2615c2063973eacb628903e79c6` |
| **Runner Script** | `scripts/run_mixtral_style_experiment.sh` | (Parameterized with `WATCHDOG_TIMEOUT=25200`) |

---

## 3. Invocation & Execution Timestamps

- **Invocation Command:**
  ```bash
  WATCHDOG_TIMEOUT=25200 ./scripts/run_mixtral_style_experiment.sh configs/deepseekv3_b1_full_run_5step.yaml
  ```
- **Process ID (PID):** 5216
- **Start Timestamp (UTC):** `2026-10-11T09:35:22Z`
- **End Timestamp (UTC):** `2026-10-11T15:28:01Z`
- **Host Wall-Clock Elapsed Time:** **21,159.0 seconds** (352.65 minutes / **5.877 hours**)
- **Dedicated Watchdog Timeout:** 25,200 seconds (7.0 hours)
- **Watchdog Headroom Margin Remaining:** **4,041.0 seconds** (67.35 minutes)
- **Process Exit Code:** `0`

---

## 4. Raw Output Trace Inventory

| Trace File | Location | Line Count | File Size | SHA-256 Checksum |
|:---|:---|:---:|:---:|:---|
| **Routing Trace** | `traces/expert_routing.csv` | 2,281 | 35,030 B | `04053306f40685f13ddf065c7b230501a63743942affae21415a1ecf53d95f6f` |
| **Load Telemetry** | `traces/expert_load.csv` | 18,091 | 1,227,157 B | `d47a7d540d5b332178d338086df2ecb6f3ef1d44091591bad68325f82be8977a` |
| **Cache Events** | `traces/expert_cache.csv` | 10,343 | 290,443 B | `4e4f0871fb1855eb70fa1f09190a7ccf4fe9122adf29ce200984fa73b20f7952` |
| **Canonical CSV** | `csv/deepseekV3_synthesis_..._ramul_decode.csv` | 6 | 2,156 B | `95f2f28b29c219503e8658d75543b81c912e075da5bdd26fb157313c424d2ac0` |
| **Status File** | `logs/completion_status.txt` | 19 | 1,369 B | `d175b94f6f4c7d6c5471d2b7d91ec0dc6308cfdc5928d3ba134604e334df37f3` |

---

## 5. Derived Analysis & Metric Artifacts

| Artifact | Path | Size | SHA-256 Checksum |
|:---|:---|:---:|:---|
| **Summary Metrics JSON** | `processed/summary_metrics.json` | 3,140 B | `a552679657605c3911215166d898c257cb9e7c98e41267dbaaead8dd4d84da6b` |
| **Summary Metrics CSV** | `processed/summary_metrics.csv` | 2,555 B | `3b75016cd92f28ba96c61ff50a068774008afafeee40913ed606de1526dc2d30` |
| **Advanced Analysis JSON**| `processed/advanced_analysis.json` | 362 B | `1d77d3e08f1f693e6e593727059b2b204057f2da8dc8992b1e02195a53c17126` |
| **Integrity Checks JSON** | `processed/integrity_checks.json` | 124 B | `0752860a0f73c94240694c6ecbfb19be71b34e8a574589f0b8312c72da47ca63` |
| **Per-Token Metrics** | `processed/per_token_metrics.csv` | 106 B | `ce2eab950bd48a2fb673c5811c1b7c2b73d3c913b5c4feacbc835939fefcc659` |
| **Per-Layer Load Table** | `processed/per_layer_loads.csv` | 2,857 B | `247015de3741f87c7ef3ae8ed9de275ac38fe529110425fbd5e92374c8a7d472` |
| **Per-Layer Routing Table**| `processed/per_layer_routing.csv` | 571 B | `a95bc1aa5c42448d4d117f3203b8639d27a49861ac6681aa6d160656b51d7ab4` |
| **Per-Expert Metrics** | `processed/per_expert_metrics.csv` | 7,297 B | `f886e87b2f50452cbfcfebf9a7e85c92129bc241b586dad84dae3251010419c4` |
| **Expert Frequency** | `processed/expert_frequency.csv` | 1,559 B | `012faf8af37c52e1507a3574eacbfde075b5aba1ba5e64a689ab9ceff14fa35e` |
| **Cache Event Counts** | `processed/cache_event_counts.csv` | 55 B | `0a8130400e0db47fe78a562f1b9b17bc5b02ff26bbe1d4b1e9ab5ac7704f6b00` |
| **Temporal Jaccard Table**| `processed/temporal_routing_jaccard.csv` | 5,885 B | `9036d1c30ba30aa613cc975f101afab1a48093d938ea0edb208ce19b9c844fe5` |

---

## 6. Generated Publication Figures

| Figure | Formats | Resolution | SHA-256 Checksums |
|:---|:---:|:---:|:---|
| **Fig 1: Temporal Routing Correlation** | PNG, PDF | 300 DPI / Vector | PNG: `e265a67dd4d4fbb6ba69d43cacfaae09b5ea277f1506d4877ff79ee2569ea1b7`<br>PDF: `8b6237f23ac5624d7e84926e316b7aae24e720a5461a18b50fa0a0b066c70282` |
| **Fig 2: Cache Occupancy Evolution** | PNG, PDF | 300 DPI / Vector | PNG: `03edc0dcb436ee10c950bf9bbef24cac6bdcd57440a72917c17c8ae6e9663c90`<br>PDF: `5dc0fddac0ef4954325a6e12150746e640b71e05d6126fe2c862be02ed13472c` |
| **Fig 3: Cache Event Composition** | PNG, PDF | 300 DPI / Vector | PNG: `2e4348485b83e758e0293842a293747b82b65d7dc07d302f84dfd6295cfd6ecb`<br>PDF: `dd8cd6f30ce4dddacbd49a93a09fc2cf4544151e180ea225c4d6c990c3683e04` |
| **Fig 4: Per-Layer DDR5 Latency** | PNG, PDF | 300 DPI / Vector | PNG: `856a0aa0c6a2e037a90469b7582b5db6471f071f56af33e59bc09985844c6bc3`<br>PDF: `ac980852235c56bd3681dc47468e045962d1600125a0cb1962bca4135c10f951` |
| **Fig 5: Reconciled Timing Disparity** | PNG, PDF | 300 DPI / Vector | PNG: `e713b410ebd015ff360cddf8703c703b3e92d435bb3a5fe3b4bc0d32840a331e`<br>PDF: `4000e48d5157de7f469170819d495602540a11e359d1df419b2b082adc2adb72` |

---

## 7. Reproduction Instructions

To reproduce the analysis, tables, and figures from the raw simulation outputs:
```bash
cd /home/mohman/ca_project/LLMSimulator

# 1. Process telemetry CSVs into structured tabular summaries
python3 reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/scripts/process_metrics.py \
    --trace-dir reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/traces \
    --csv-dir reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/csv \
    --output-dir reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/processed

# 2. Run multi-token sequential routing transition analysis
python3 reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/scripts/analyze_results.py \
    --traces-dir reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/traces \
    --processed-dir reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/processed

# 3. Regenerate publication figures
python3 reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/scripts/generate_figures.py \
    --processed-dir reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/processed \
    --traces-dir reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/traces \
    --output-dir reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011/full_run_5step_20261011/figures
```
