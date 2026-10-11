# Reproducibility Commands

The following commands are strictly structured to execute the A100 benchmarks while controlling debug output to avoid log bloat.

## 1. Build Command
```bash
make clean
make -j $(nproc)
```

## 2. Debug Command (Test Run)
```bash
# Validating binary integrity on a reduced config
./sim reproduction_A100/configs/a100_gpu_baseline.yaml
```

## 3. GPU Baseline Command
```bash
./sim reproduction_A100/configs/a100_gpu_baseline.yaml > reproduction_A100/logs/a100_gpu_baseline_concise_run.log 2>&1
```

## 4. Duplex Command
```bash
./sim reproduction_A100/configs/a100_duplex.yaml > reproduction_A100/logs/a100_duplex_concise_run.log 2>&1
```

## 5. PE Command
```bash
./sim reproduction_A100/configs/a100_duplex_pe.yaml > reproduction_A100/logs/a100_duplex_pe_concise_run.log 2>&1
```

## 6. ET Command
```bash
./sim reproduction_A100/configs/a100_duplex_pe_et.yaml > reproduction_A100/logs/a100_duplex_pe_et_concise_run.log 2>&1
```

## 7. Analysis & Plotting Commands
```bash
# Executed sequentially after completion
python3 scratch/analyze_A100_routing.py
python3 scratch/analyze_A100_performance.py
python3 scratch/generate_A100_plots.py
```
*(Analysis scripts will be developed strictly referencing the frozen metrics)*
