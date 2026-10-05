#!/bin/bash
# run_reproduction.sh
# Execute this script from the root of LLMSimulator to reproduce all results.

set -e

echo "Building Simulator..."
make clean
make

echo "Running Phase B: GPU Baseline..."
TRACE_DIR=reproduction/results/traces/gpu_baseline ./build/run reproduction/configs/gpu_baseline.yaml > reproduction/results/traces/gpu_baseline/gpu_baseline_run.log 2>&1

echo "Running Phase C: Plain Duplex..."
TRACE_DIR=reproduction/results/traces/duplex ./build/run reproduction/configs/duplex.yaml > reproduction/results/traces/duplex/duplex_run.log 2>&1

echo "Running Phase E: Duplex + PE..."
TRACE_DIR=reproduction/results/traces/duplex_pe ./build/run reproduction/configs/duplex_pe.yaml > reproduction/results/traces/duplex_pe/duplex_pe_run.log 2>&1

echo "Running Phase F: Duplex + PE + ET..."
TRACE_DIR=reproduction/results/traces/duplex_pe_et ./build/run reproduction/configs/duplex_pe_et.yaml > reproduction/results/traces/duplex_pe_et/duplex_pe_et_run.log 2>&1

echo "Simulations complete. Results are available in reproduction/results/csv/ and reproduction/results/traces/"
echo "Note: The python analysis scripts are external to the C++ simulator and were executed separately in the host environment."
