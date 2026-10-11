#!/usr/bin/env bash
set -uo pipefail

REPO_DIR="/home/mohman/ca_project/LLMSimulator"
cd "$REPO_DIR"

LOCK_FILE="${REPO_DIR}/eval/ds5c8_retry.lock"

# Trap cleanup to remove lock file on exit or signals
cleanup() {
    rm -f "$LOCK_FILE"
}
trap cleanup EXIT INT TERM

# 1. Single-run lock
if [ -e "$LOCK_FILE" ]; then
    LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "unknown")
    if [ -n "$LOCK_PID" ] && kill -0 "$LOCK_PID" 2>/dev/null; then
        echo "[ERROR] A retry is already running under PID $LOCK_PID. Aborting."
        exit 1
    else
        echo "[WARN] Stale lock file found. Removing."
        rm -f "$LOCK_FILE"
    fi
fi
echo "$$" > "$LOCK_FILE"

# 2. Check available system RAM (minimum 2.5 GB / 2560 MB)
AVAIL_RAM_MB=$(free -m | awk '/^Mem:/{print $7}')
echo "[CHECK] Current Available RAM: ${AVAIL_RAM_MB} MB"
if [ "$AVAIL_RAM_MB" -lt 2560 ]; then
    echo "[ABORT] Available RAM (${AVAIL_RAM_MB} MB) is below 2.5 GB safety threshold. Refusing to start."
    exit 2
fi

# 3. Determine unique retry log file
LOG_FILE="eval/staged_retry2_output.txt"
echo "[LOG] Writing output to: $LOG_FILE"

# Ensure output directory exists
RETRY_DIR="reproduction_A100/with_off_chip_dram/deepseekv3/results/ds5c8_retry2"
mkdir -p "$RETRY_DIR"

# 4. Launch executable
EXECUTABLE="./build/test_staged_module_graph"
if [ ! -x "$EXECUTABLE" ]; then
    echo "[ERROR] Executable $EXECUTABLE not found or not executable. Please compile first."
    exit 3
fi

export TRACE_DIR="$RETRY_DIR"
export OFFCHIP_DRAM_CFG="reproduction_A100/with_off_chip_dram/configs/dram_config_DDR5_retry.yaml"

echo "[START] Launching $EXECUTABLE at $(date -u)" | tee -a "$LOG_FILE"
START_TIME=$(date +%s)

"$EXECUTABLE" > "$LOG_FILE" 2>&1 &
SIM_PID=$!
echo "[INFO] Simulator running under PID: $SIM_PID" | tee -a "$LOG_FILE"

TERMINATION_REASON="NORMAL"
TIMEOUT_LIMIT=300

# 5. Watchdog loop: 300s timeout & memory guard
while kill -0 "$SIM_PID" 2>/dev/null; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    
    # Check timeout
    if [ "$ELAPSED" -ge "$TIMEOUT_LIMIT" ]; then
        echo "[WATCHDOG] Wall-clock timeout of ${TIMEOUT_LIMIT}s reached! Terminating PID $SIM_PID." | tee -a "$LOG_FILE"
        TERMINATION_REASON="TIMEOUT"
        kill -TERM "$SIM_PID" 2>/dev/null || true
        sleep 3
        kill -KILL "$SIM_PID" 2>/dev/null || true
        break
    fi
    
    # Check RAM
    CURR_RAM_MB=$(free -m | awk '/^Mem:/{print $7}')
    if [ "$CURR_RAM_MB" -lt 2560 ]; then
        echo "[WATCHDOG] Available RAM dropped to ${CURR_RAM_MB} MB (< 2.5 GB). Stopping simulator safely." | tee -a "$LOG_FILE"
        TERMINATION_REASON="OOM_PROTECTION"
        kill -TERM "$SIM_PID" 2>/dev/null || true
        sleep 3
        kill -KILL "$SIM_PID" 2>/dev/null || true
        break
    fi
    
    sleep 2
done

# Wait for process exit and capture exact exit code
wait "$SIM_PID" 2>/dev/null
EXE_EXIT_CODE=$?
END_TIME=$(date +%s)
TOTAL_WALLCLOCK=$((END_TIME - START_TIME))

echo "[END] Process ended after ${TOTAL_WALLCLOCK}s with code ${EXE_EXIT_CODE} at $(date -u)" | tee -a "$LOG_FILE"

# 6. Determine final status
if [ "$TERMINATION_REASON" = "TIMEOUT" ]; then
    FINAL_STATUS="TIMEOUT"
elif [ "$TERMINATION_REASON" = "OOM_PROTECTION" ]; then
    FINAL_STATUS="INTERRUPTED"
elif [ "$EXE_EXIT_CODE" -eq 0 ]; then
    FINAL_STATUS="SUCCESS"
else
    FINAL_STATUS="FAILURE"
fi

echo "==================================================" | tee -a "$LOG_FILE"
echo "DS-5C-8R RETRY FINAL STATUS: ${FINAL_STATUS}" | tee -a "$LOG_FILE"
echo "EXIT CODE: ${EXE_EXIT_CODE}" | tee -a "$LOG_FILE"
echo "WALLCLOCK DURATION: ${TOTAL_WALLCLOCK}s" | tee -a "$LOG_FILE"
echo "LOG FILE: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

exit "$EXE_EXIT_CODE"
