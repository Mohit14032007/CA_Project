#!/usr/bin/env bash
set -uo pipefail

REPO_DIR="/home/mohman/ca_project/LLMSimulator"
cd "$REPO_DIR"

EXP_DIR="${REPO_DIR}/reproduction_A100/with_off_chip_dram/deepseekv3/experiments/deepseekv3_mixtral_style_20261011"
CONFIG_FILE="${1:-${EXP_DIR}/configs/deepseekv3_b1_pilot.yaml}"
if [ ! -f "$CONFIG_FILE" ] && [ -f "${EXP_DIR}/${CONFIG_FILE}" ]; then
    CONFIG_FILE="${EXP_DIR}/${CONFIG_FILE}"
fi
EXECUTABLE="${REPO_DIR}/build/run_deepseekv3_mixtral_style"

# Read configured directories from YAML if available
CFG_TRACES=$(python3 -c "import yaml; cfg=yaml.safe_load(open('$CONFIG_FILE')); print(cfg.get('log', {}).get('traces_directory', ''))" 2>/dev/null || true)
CFG_CSV=$(python3 -c "import yaml; cfg=yaml.safe_load(open('$CONFIG_FILE')); print(cfg.get('log', {}).get('output_directory', ''))" 2>/dev/null || true)
CFG_RAW=$(python3 -c "import yaml; cfg=yaml.safe_load(open('$CONFIG_FILE')); print(cfg.get('log', {}).get('gantt_directory', ''))" 2>/dev/null || true)

if [ -n "$CFG_TRACES" ]; then
    TRACES_DIR="${REPO_DIR}/${CFG_TRACES%/}"
else
    TRACES_DIR="${EXP_DIR}/traces"
fi

if [ -n "$CFG_CSV" ]; then
    CSV_DIR="${REPO_DIR}/${CFG_CSV%/}"
else
    CSV_DIR="${EXP_DIR}/csv"
fi

if [ -n "$CFG_RAW" ]; then
    RAW_DIR="${REPO_DIR}/${CFG_RAW%/}"
else
    RAW_DIR="${EXP_DIR}/raw"
fi

RUN_ROOT=$(dirname "$TRACES_DIR")
if [ "$RUN_ROOT" = "$EXP_DIR" ]; then
    LOG_FILE="${EXP_DIR}/logs/run.log"
    STATUS_FILE="${EXP_DIR}/logs/completion_status.txt"
    LOCK_FILE="${EXP_DIR}/deepseekv3_mixtral.lock"
    RUN_ID="deepseekv3_mixtral_style_20261011"
    PROCESSED_DIR="${EXP_DIR}/processed"
    FIGURES_DIR="${EXP_DIR}/figures"
else
    LOG_FILE="${RUN_ROOT}/logs/run.log"
    STATUS_FILE="${RUN_ROOT}/logs/completion_status.txt"
    LOCK_FILE="${RUN_ROOT}/run.lock"
    RUN_ID=$(basename "$RUN_ROOT")
    PROCESSED_DIR="${RUN_ROOT}/processed"
    FIGURES_DIR="${RUN_ROOT}/figures"
fi

# Trap cleanup to remove lock file on exit or signals
cleanup() {
    rm -f "$LOCK_FILE"
}
trap cleanup EXIT INT TERM

# 1. Single-run lock guard
if [ -e "$LOCK_FILE" ]; then
    LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "unknown")
    if [ -n "$LOCK_PID" ] && kill -0 "$LOCK_PID" 2>/dev/null; then
        echo "[ERROR] An experiment run is already active under PID $LOCK_PID. Aborting."
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
mkdir -p "$TRACES_DIR"
mkdir -p "$CSV_DIR"
mkdir -p "$RAW_DIR"
mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$PROCESSED_DIR"
mkdir -p "$FIGURES_DIR"

# 5. Check executable
if [ ! -x "$EXECUTABLE" ]; then
    echo "[ERROR] Executable $EXECUTABLE not found or not executable. Compile first."
    exit 4
fi

# 6. Check config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Configuration file $CONFIG_FILE not found."
    exit 5
fi

export TRACE_DIR="$TRACES_DIR"
export OFFCHIP_DRAM_CFG="${EXP_DIR}/config_snapshot/dram_config_DDR5_retry.yaml"

START_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
START_EPOCH=$(date +%s)

echo "======================================================================" | tee -a "$LOG_FILE"
echo "DS-6C: Launching DeepSeek-V3 Mixtral-Style Experiment" | tee -a "$LOG_FILE"
echo "Executable: $EXECUTABLE" | tee -a "$LOG_FILE"
echo "Config: $CONFIG_FILE" | tee -a "$LOG_FILE"
echo "Target Trace Dir: $TRACES_DIR" | tee -a "$LOG_FILE"
echo "Target CSV Dir: $CSV_DIR" | tee -a "$LOG_FILE"
echo "Log File: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Start Time (UTC): $START_UTC" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

"$EXECUTABLE" "$CONFIG_FILE" >> "$LOG_FILE" 2>&1 &
SIM_PID=$!
echo "[INFO] Simulator running under PID: $SIM_PID" | tee -a "$LOG_FILE"

TERMINATION_REASON="NORMAL"
TIMEOUT_LIMIT="${WATCHDOG_TIMEOUT:-14400}" # Default 4 hours watchdog limit; overridable via WATCHDOG_TIMEOUT
LAST_LOG_TIME=$START_EPOCH
MIN_AVAIL_RAM_OBSERVED=$AVAIL_RAM_START_MB
PEAK_RSS_KB=0

# 7. Watchdog and health monitoring loop
while kill -0 "$SIM_PID" 2>/dev/null; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_EPOCH))
    
    # Check timeout
    if [ "$ELAPSED" -ge "$TIMEOUT_LIMIT" ]; then
        echo "[WATCHDOG] Hard timeout limit of ${TIMEOUT_LIMIT}s reached! Terminating PID $SIM_PID." | tee -a "$LOG_FILE"
        TERMINATION_REASON="TIMEOUT"
        kill -TERM "$SIM_PID" 2>/dev/null || true
        sleep 5
        kill -KILL "$SIM_PID" 2>/dev/null || true
        break
    fi
    
    # Check RAM safety
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

    # Progress heartbeat every 30 seconds
    if [ $((CURRENT_TIME - LAST_LOG_TIME)) -ge 30 ]; then
        LAST_LOG_TIME=$CURRENT_TIME
        ROUTING_LINES=$(wc -l < "${TRACES_DIR}/expert_routing.csv" 2>/dev/null || echo 0)
        LOAD_LINES=$(wc -l < "${TRACES_DIR}/expert_load.csv" 2>/dev/null || echo 0)
        CACHE_LINES=$(wc -l < "${TRACES_DIR}/expert_cache.csv" 2>/dev/null || echo 0)
        echo "[PROGRESS] Elapsed: ${ELAPSED}s | RAM: ${CURR_RAM_MB}MB | RSS: ${CURR_RSS}KB | Routing: ${ROUTING_LINES} | Load: ${LOAD_LINES} | Cache: ${CACHE_LINES}" | tee -a "$LOG_FILE"
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
    FINAL_STATUS="TIMEOUT"
fi

echo "======================================================================" | tee -a "$LOG_FILE"
echo "Process ended after ${TOTAL_WALLCLOCK}s with code ${EXE_EXIT_CODE} at $END_UTC" | tee -a "$LOG_FILE"
echo "DS-6C STATUS: $FINAL_STATUS (${TERMINATION_REASON})" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"

ROUTING_COUNT=$(wc -l < "${TRACES_DIR}/expert_routing.csv" 2>/dev/null || echo 0)
LOAD_COUNT=$(wc -l < "${TRACES_DIR}/expert_load.csv" 2>/dev/null || echo 0)
CACHE_COUNT=$(wc -l < "${TRACES_DIR}/expert_cache.csv" 2>/dev/null || echo 0)

HASH_ROUTING=$(sha256sum "${TRACES_DIR}/expert_routing.csv" 2>/dev/null | awk '{print $1}' || echo "N/A")
HASH_LOAD=$(sha256sum "${TRACES_DIR}/expert_load.csv" 2>/dev/null | awk '{print $1}' || echo "N/A")
HASH_CACHE=$(sha256sum "${TRACES_DIR}/expert_cache.csv" 2>/dev/null | awk '{print $1}' || echo "N/A")
COMMIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

cat <<EOF > "$STATUS_FILE"
RUN_ID=${RUN_ID}
STATUS=${FINAL_STATUS}
START_TIMESTAMP=${START_UTC}
END_TIMESTAMP=${END_UTC}
HOST_WALLCLOCK_SECONDS=${TOTAL_WALLCLOCK}
EXIT_CODE=${EXE_EXIT_CODE}
CONFIG_FILE=${CONFIG_FILE}
SOURCE_COMMIT=${COMMIT_HASH}
ROUTING_PATH=${TRACES_DIR}/expert_routing.csv
LOAD_PATH=${TRACES_DIR}/expert_load.csv
CACHE_PATH=${TRACES_DIR}/expert_cache.csv
FILE_HASHES=routing:${HASH_ROUTING};load:${HASH_LOAD};cache:${HASH_CACHE}
ROW_COUNTS=routing:${ROUTING_COUNT};load:${LOAD_COUNT};cache:${CACHE_COUNT}
AVAILABLE_DISK_AT_START_MB=${AVAIL_DISK_START_MB}
AVAILABLE_DISK_AT_END_MB=${AVAIL_DISK_END_MB}
MIN_AVAILABLE_RAM_OBSERVED_MB=${MIN_AVAIL_RAM_OBSERVED}
PEAK_RSS_BYTES=${PEAK_RSS_BYTES}
ERROR_OR_TERMINATION_REASON=${TERMINATION_REASON}
EOF

cp -f "$STATUS_FILE" "${RAW_DIR}/completion_status.txt" 2>/dev/null || true

if [ "$FINAL_STATUS" = "SUCCESS" ]; then
    echo "[INFO] Running telemetry post-processing..." | tee -a "$LOG_FILE"
    python3 "${EXP_DIR}/scripts/process_metrics.py" \
        --trace-dir "$TRACES_DIR" \
        --csv-dir "$CSV_DIR" \
        --output-dir "$PROCESSED_DIR" | tee -a "$LOG_FILE"
    
    echo "[INFO] Running sequential routing analysis..." | tee -a "$LOG_FILE"
    python3 "${EXP_DIR}/scripts/analyze_results.py" \
        --traces-dir "$TRACES_DIR" \
        --processed-dir "$PROCESSED_DIR" | tee -a "$LOG_FILE"
        
    echo "[INFO] Generating publication figures..." | tee -a "$LOG_FILE"
    python3 "${EXP_DIR}/scripts/generate_figures.py" \
        --processed-dir "$PROCESSED_DIR" \
        --traces-dir "$TRACES_DIR" \
        --output-dir "$FIGURES_DIR" | tee -a "$LOG_FILE"
fi

exit "$EXE_EXIT_CODE"
