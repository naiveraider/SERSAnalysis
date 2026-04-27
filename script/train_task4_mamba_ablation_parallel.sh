#!/bin/bash
# Run the original Mamba model plus Mamba pooling ablations in parallel for Task 4.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "Usage: bash script/train_task4_mamba_ablation_parallel.sh"
  echo ""
  echo "Runs 4 jobs in parallel for Task 4:"
  echo "  1. Original mamba model"
  echo "  2. mamba_mean"
  echo "  3. mamba_max"
  echo "  4. mamba_meanmax"
  echo ""
  echo "Environment variables:"
  echo "  N_RUNS          Number of runs per model (default: 10)"
  echo "  EPOCHS          Training epochs (default: 200)"
  echo "  BATCH_SIZE      Batch size (default: 32)"
  echo "  LEARNING_RATE   Learning rate (default: 0.0005)"
  echo "  VALIDATION_SPLIT Validation split (default: 0.25)"
  echo "  DEVICE          Fallback device for all jobs when DEVICE_LIST is unset"
  echo "  DEVICE_LIST     Comma or space separated device list for the 4 jobs"
  echo "                  Order: mamba,mamba_mean,mamba_max,mamba_meanmax"
  echo "  PYTHON_BIN      Python executable to use (default: .venv/bin/python)"
  echo "  DRY_RUN=1       Only print commands; do not launch any jobs"
  exit 0
fi

TASK_ID=4
EPOCHS=${EPOCHS:-200}
LEARNING_RATE=${LEARNING_RATE:-0.0005}
VALIDATION_SPLIT=${VALIDATION_SPLIT:-0.25}
BATCH_SIZE=${BATCH_SIZE:-32}
N_RUNS=${N_RUNS:-10}
DEVICE=${DEVICE:-}
DRY_RUN=${DRY_RUN:-0}
TIMESTAMP=${TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}

if [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON_BIN="$PYTHON_BIN"
elif [ -x "$PWD/.venv/bin/python" ]; then
  PYTHON_BIN="$PWD/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=$(command -v python3)
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=$(command -v python)
else
  echo "Could not find a Python interpreter." >&2
  exit 1
fi

export PATH="$(dirname "$PYTHON_BIN"):$PATH"

mkdir -p results/log
WRAPPER_LOG="results/log/$(basename "$0" .sh)_${TIMESTAMP}.log"
exec > >(tee -a "$WRAPPER_LOG") 2>&1

echo "[Wrapper log] $WRAPPER_LOG"
echo "[Python] $PYTHON_BIN"

SELECTED_SCRIPT="script/train_task4_selected_models.sh"
SUMMARY_FILE="results/task4_mamba_ablation_parallel_${TIMESTAMP}.txt"

DEVICE_LIST_NORMALIZED="${DEVICE_LIST:-}"
DEVICE_LIST_NORMALIZED="${DEVICE_LIST_NORMALIZED//,/ }"
read -r -a JOB_DEVICES <<< "$DEVICE_LIST_NORMALIZED"
while [ "${#JOB_DEVICES[@]}" -lt 4 ]; do
  JOB_DEVICES+=("$DEVICE")
done

declare -a JOB_LABELS=()
declare -a JOB_PIDS=()
declare -a JOB_LOGS=()

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

start_job() {
  local label="$1"
  local job_log="$2"
  shift 2

  JOB_LABELS+=("$label")
  JOB_LOGS+=("$job_log")

  if [ "$DRY_RUN" = "1" ]; then
    echo "[Start] $label"
    echo "[Job log] $job_log"
    print_command "$@"
    echo "[Dry run] $label was not started."
    JOB_PIDS+=("dry-run")
    return
  fi

  (
    exec "$@" >> "$job_log" 2>&1
  ) >/dev/null 2>&1 &
  JOB_PIDS+=("$!")
}

start_ablation_job() {
  local label="$1"
  local model_name="$2"
  local device_value="$3"
  local job_log="$4"
  local -a command=(
    "$PYTHON_BIN" script/ablation_test.py
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --learning_rate "$LEARNING_RATE"
    --validation_split "$VALIDATION_SPLIT"
    --n_runs "$N_RUNS"
    --task_id "$TASK_ID"
  )

  if [ -n "$device_value" ]; then
    command+=(--cuda "$device_value")
  fi

  command+=("$model_name")
  start_job "$label" "$job_log" "${command[@]}"
}

start_job \
  "mamba" \
  "results/log/mamba_task4_${TIMESTAMP}.log" \
  env \
  TIMESTAMP="${TIMESTAMP}_mamba" \
  EPOCHS="$EPOCHS" \
  BATCH_SIZE="$BATCH_SIZE" \
  LEARNING_RATE="$LEARNING_RATE" \
  VALIDATION_SPLIT="$VALIDATION_SPLIT" \
  N_RUNS="$N_RUNS" \
  DEVICE="${JOB_DEVICES[0]}" \
  bash "$SELECTED_SCRIPT" mamba

start_ablation_job \
  "mamba_mean" \
  "mamba_mean" \
  "${JOB_DEVICES[1]}" \
  "results/log/mamba_mean_task4_${TIMESTAMP}.log"

start_ablation_job \
  "mamba_max" \
  "mamba_max" \
  "${JOB_DEVICES[2]}" \
  "results/log/mamba_max_task4_${TIMESTAMP}.log"

start_ablation_job \
  "mamba_meanmax" \
  "mamba_meanmax" \
  "${JOB_DEVICES[3]}" \
  "results/log/mamba_meanmax_task4_${TIMESTAMP}.log"

if [ "$DRY_RUN" = "1" ]; then
  echo "[Dry run] Commands printed only. Remove DRY_RUN=1 to launch all 4 jobs in parallel."
  exit 0
fi

{
  echo "Task 4 - Parallel Mamba Ablation Launcher"
  echo "Generated: $(date)"
  echo "Wrapper log: $WRAPPER_LOG"
  echo "Python: $PYTHON_BIN"
  echo "N_RUNS=$N_RUNS"
  echo "EPOCHS=$EPOCHS"
  echo "BATCH_SIZE=$BATCH_SIZE"
  echo "LEARNING_RATE=$LEARNING_RATE"
  echo "VALIDATION_SPLIT=$VALIDATION_SPLIT"
  echo "Mode=background-dispatch"
  echo ""
  for i in "${!JOB_LABELS[@]}"; do
    echo "Job: ${JOB_LABELS[$i]}"
    echo "  PID: ${JOB_PIDS[$i]}"
    echo "  Job log: ${JOB_LOGS[$i]}"
    echo ""
  done
} > "$SUMMARY_FILE"

echo "[Summary] $SUMMARY_FILE"
echo "[Status] 4 jobs dispatched in parallel in the background."
exit 0