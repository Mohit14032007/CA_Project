#!/usr/bin/env bash
set -uo pipefail

REPO_DIR="/home/mohman/ca_project/LLMSimulator"
cd "$REPO_DIR"

EXP_DIR="${REPO_DIR}/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_full_baseline_20261010"
LOCK_FILE="${EXP_DIR}/deepseekv3_full.lock"
LOG_FILE="${EXP_DIR}/logs/full_baseline_run.log"
RAW_DIR="${EXP_DIR}/raw"
CONFIG_DDR5="${EXP_DIR}/config_snapshot/dram_config_DDR5_retry.yaml"
EXECUTABLE="./build/test_deepseekv3_full_baseline"
RUN_ID="deepseekv3_full_baseline_20261010"
STATUS_FILE="${EXP_DIR}/logs/ds5c11_completion_status.txt"

# Trap cleanup to remove lock file on exit or signals
cleanup() {
    rm -f "$LOCK_FILE"
}
trap cleanup EXIT INT TERM

# 1. Single-run lock guard
if [ -e "$LOCK_FILE" ]; then
    LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "unknown")
    if [ -n "$LOCK_PID" ] && kill -0 "$LOCK_PID" 2>/dev/null; then
        echo "[ERROR] A full baseline run is already active under PID $LOCK_PID. Aborting."
        exit 1
    else
        echo "[WARN] Stale lock file found. Removing."
        rm -f "$LOCK_FILE"
    fi
fi
echo "$$" > "$LOCK_FILE"

# 2. Check available system RAM (minimum 2.5 GB / 2560 MB)
AVAIL_RAM_START_MB=$(free -m | awk '/^Mem:/{print $7}')
echo "[CHECK] Current Available RAM: ${AVAIL_RAM_START_MB} MB"
if [ "$AVAIL_RAM_START_MB" -lt 2560 ]; then
    echo "[ABORT] Available RAM (${AVAIL_RAM_START_MB} MB) is below 2.5 GB safety threshold. Refusing to start."
    exit 2
fi

# 3. Check available disk space (minimum 5 GB)
AVAIL_DISK_START_KB=$(df -k "$EXP_DIR" | awk 'NR==2{print $4}')
AVAIL_DISK_START_MB=$((AVAIL_DISK_START_KB / 1024))
echo "[CHECK] Current Available Disk: ${AVAIL_DISK_START_MB} MB"
if [ "$AVAIL_DISK_START_MB" -lt 5120 ]; then
    echo "[ABORT] Available Disk (${AVAIL_DISK_START_MB} MB) is below 5 GB threshold. Refusing to start."
    exit 3
fi

# 4. Ensure directories exist
mkdir -p "$RAW_DIR"
mkdir -p "${EXP_DIR}/logs"
mkdir -p "${EXP_DIR}/processed"
mkdir -p "${EXP_DIR}/figures"
mkdir -p "${EXP_DIR}/report"

# 5. Check executable
if [ ! -x "$EXECUTABLE" ]; then
    echo "[ERROR] Executable $EXECUTABLE not found or not executable. Compile first."
    exit 4
fi

export TRACE_DIR="$RAW_DIR"
export OFFCHIP_DRAM_CFG="$CONFIG_DDR5"

START_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
START_EPOCH=$(date +%s)

echo "======================================================================" | tee -a "$LOG_FILE"
echo "DS-5C-11: Launching DeepSeek-V3 Full Baseline" | tee -a "$LOG_FILE"
echo "Executable: $EXECUTABLE" | tee -a "$LOG_FILE"
echo "Output Raw Dir: $RAW_DIR" | tee -a "$LOG_FILE"
echo "Start Time (UTC): $START_UTC" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

"$EXECUTABLE" >> "$LOG_FILE" 2>&1 &
SIM_PID=$!
echo "[INFO] Simulator running under PID: $SIM_PID" | tee -a "$LOG_FILE"

TERMINATION_REASON="NORMAL"
TIMEOUT_LIMIT=7200 # 2 hours hard watchdog
LAST_LOG_TIME=$START_EPOCH
MIN_AVAIL_RAM_OBSERVED=$AVAIL_RAM_START_MB
PEAK_RSS_KB=0

# 6. Watchdog and health monitoring loop
while kill -0 "$SIM_PID" 2>/dev/null; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_EPOCH))
    
    # Check timeout
    if [ "$ELAPSED" -ge "$TIMEOUT_LIMIT" ]; then
        echo "[WATCHDOG] Hard timeout limit of ${TIMEOUT_LIMIT}s (120m) reached! Terminating PID $SIM_PID." | tee -a "$LOG_FILE"
        TERMINATION_REASON="TIMEOUT"
        kill -TERM "$SIM_PID" 2>/dev/null || true
        sleep 5
        kill -KILL "$SIM_PID" 2>/dev/null || true
        break
    fi
    
    # Check RAM safety (strictly enforced 2.5 GB / 2560 MB threshold)
    CURR_RAM_MB=$(free -m | awk '/^Mem:/{print $7}')
    if [ "$CURR_RAM_MB" -lt "$MIN_AVAIL_RAM_OBSERVED" ]; then
        MIN_AVAIL_RAM_OBSERVED=$CURR_RAM_MB
    fi
    if [ "$CURR_RAM_MB" -lt 2560 ]; then
        echo "[WATCHDOG] Available RAM dropped to ${CURR_RAM_MB} MB (< 2.5 GB). Stopping simulator safely." | tee -a "$LOG_FILE"
        TERMINATION_REASON="OOM_PROTECTION"
        kill -TERM "$SIM_PID" 2>/dev/null || true
        sleep 5
        kill -KILL "$SIM_PID" 2>/dev/null || true
        break
    fi

    # Measure current RSS
    CURR_RSS=$(ps -p "$SIM_PID" -o rss= 2>/dev/null | tr -d ' ' || echo 0)
    if [ -n "$CURR_RSS" ] && [ "$CURR_RSS" -gt "$PEAK_RSS_KB" ] 2>/dev/null; then
        PEAK_RSS_KB=$CURR_RSS
    fi

    # Progress heartbeat every 60 seconds
    if [ $((CURRENT_TIME - LAST_LOG_TIME)) -ge 60 ]; then
        LAST_LOG_TIME=$CURRENT_TIME
        ROUTING_LINES=$(wc -l < "${RAW_DIR}/expert_routing.csv" 2>/dev/null || echo 0)
        LOAD_LINES=$(wc -l < "${RAW_DIR}/expert_load.csv" 2>/dev/null || echo 0)
        CACHE_LINES=$(wc -l < "${RAW_DIR}/expert_cache.csv" 2>/dev/null || echo 0)
        echo "[PROGRESS] Elapsed: ${ELAPSED}s / ${TIMEOUT_LIMIT}s | RAM: ${CURR_RAM_MB}MB | RSS: ${CURR_RSS}KB | Routing: ${ROUTING_LINES} | Load: ${LOAD_LINES} | Cache: ${CACHE_LINES}" | tee -a "$LOG_FILE"
    fi
    
    sleep 5
done

# Wait for process exit and capture exact exit code
wait "$SIM_PID" 2>/dev/null
EXE_EXIT_CODE=$?
END_EPOCH=$(date +%s)
END_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
TOTAL_WALLCLOCK=$((END_EPOCH - START_EPOCH))

AVAIL_DISK_END_KB=$(df -k "$EXP_DIR" | awk 'NR==2{print $4}')
AVAIL_DISK_END_MB=$((AVAIL_DISK_END_KB / 1024))
PEAK_RSS_BYTES=$((PEAK_RSS_KB * 1024))

FINAL_STATUS="FAILED"
if [ "$EXE_EXIT_CODE" -eq 0 ] && [ "$TERMINATION_REASON" = "NORMAL" ]; then
    FINAL_STATUS="SUCCESS"
elif [ "$TERMINATION_REASON" = "TIMEOUT" ]; then
    FINAL_STATUS="INCOMPLETE"
fi

echo "======================================================================" | tee -a "$LOG_FILE"
echo "Process ended after ${TOTAL_WALLCLOCK}s with code ${EXE_EXIT_CODE} at $END_UTC" | tee -a "$LOG_FILE"
echo "DS-5C-11 FULL BASELINE STATUS: $FINAL_STATUS (${TERMINATION_REASON})" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

# Raw row counts and file hashes
ROUTING_COUNT=$(wc -l < "${RAW_DIR}/expert_routing.csv" 2>/dev/null || echo 0)
LOAD_COUNT=$(wc -l < "${RAW_DIR}/expert_load.csv" 2>/dev/null || echo 0)
CACHE_COUNT=$(wc -l < "${RAW_DIR}/expert_cache.csv" 2>/dev/null || echo 0)

HASH_ROUTING=$(sha256sum "${RAW_DIR}/expert_routing.csv" 2>/dev/null | awk '{print $1}' || echo "N/A")
HASH_LOAD=$(sha256sum "${RAW_DIR}/expert_load.csv" 2>/dev/null | awk '{print $1}' || echo "N/A")
HASH_CACHE=$(sha256sum "${RAW_DIR}/expert_cache.csv" 2>/dev/null | awk '{print $1}' || echo "N/A")

COMMIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
CONFIG_CHECKSUMS=$(cat "${EXP_DIR}/config_snapshot/checksums.sha256" 2>/dev/null | tr '\n' '; ' || echo "N/A")

cat <<EOF > "$STATUS_FILE"
RUN_ID=${RUN_ID}
STATUS=${FINAL_STATUS}
START_TIMESTAMP=${START_UTC}
END_TIMESTAMP=${END_UTC}
HOST_WALLCLOCK_SECONDS=${TOTAL_WALLCLOCK}
EXIT_CODE=${EXE_EXIT_CODE}
WATCHDOG_SECONDS=${TIMEOUT_LIMIT}
MODEL_VARIANT=deepseekV3_60layer_FP16_synthetic
ROUTING_TYPE=synthetic_prng_with_cached_replay
CONFIG_CHECKSUMS=${CONFIG_CHECKSUMS}
SOURCE_COMMIT=${COMMIT_HASH}
RAW_ROUTING_PATH=${RAW_DIR}/expert_routing.csv
RAW_LOAD_PATH=${RAW_DIR}/expert_load.csv
RAW_CACHE_PATH=${RAW_DIR}/expert_cache.csv
RAW_FILE_HASHES=routing:${HASH_ROUTING};load:${HASH_LOAD};cache:${HASH_CACHE}
RAW_ROW_COUNTS=routing:${ROUTING_COUNT};load:${LOAD_COUNT};cache:${CACHE_COUNT}
AVAILABLE_DISK_AT_START_MB=${AVAIL_DISK_START_MB}
AVAILABLE_DISK_AT_END_MB=${AVAIL_DISK_END_MB}
MIN_AVAILABLE_RAM_OBSERVED_MB=${MIN_AVAIL_RAM_OBSERVED}
PEAK_RSS_BYTES=${PEAK_RSS_BYTES}
ERROR_OR_TERMINATION_REASON=${TERMINATION_REASON}
EOF

# Copy to raw directory as well
cp -f "$STATUS_FILE" "${RAW_DIR}/ds5c11_completion_status.txt" 2>/dev/null || true

exit "$EXE_EXIT_CODE"
