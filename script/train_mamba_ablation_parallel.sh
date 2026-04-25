#!/bin/bash
# Run the original Mamba model plus Mamba pooling ablations in parallel.
#
# Usage:
#   bash script/train_mamba_ablation_parallel.sh 1
#   TASK_ID=4 N_RUNS=5 DEVICE_LIST="cuda:0,cuda:1,cuda:2,cuda:3" bash script/train_mamba_ablation_parallel.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "Usage: bash script/train_mamba_ablation_parallel.sh [task_id]"
  echo ""
  echo "Runs 4 jobs in parallel for the selected task:"
  echo "  1. Original mamba model"
  echo "  2. mamba_mean"
  echo "  3. mamba_max"
  echo "  4. mamba_meanmax"
  echo ""
  echo "Environment variables:"
  echo "  TASK_ID         Task id when not passed as positional arg (default: 1)"
  echo "  N_RUNS          Number of runs per model (default: 10)"
  echo "  EPOCHS          Training epochs (task defaults match existing scripts)"
  echo "  BATCH_SIZE      Batch size (default: 32)"
  echo "  LEARNING_RATE   Learning rate (task defaults match existing scripts)"
  echo "  VALIDATION_SPLIT Validation split (task defaults match existing scripts)"
  echo "  DEVICE          Fallback device for all jobs when DEVICE_LIST is unset"
  echo "  DEVICE_LIST     Comma or space separated device list for the 4 jobs"
  echo "                  Order: mamba,mamba_mean,mamba_max,mamba_meanmax"
  echo "  PYTHON_BIN      Python executable to use (default: .venv/bin/python)"
  echo "  DRY_RUN=1       Print commands without launching jobs"
  exit 0
fi

TASK_ID="${1:-${TASK_ID:-1}}"
case "$TASK_ID" in
  1|2|3|4)
    ;;
  *)
    echo "Invalid TASK_ID: $TASK_ID. Expected one of 1, 2, 3, 4." >&2
    exit 1
    ;;
esac

case "$TASK_ID" in
  4)
    EPOCHS=${EPOCHS:-200}
    LEARNING_RATE=${LEARNING_RATE:-0.0005}
    VALIDATION_SPLIT=${VALIDATION_SPLIT:-0.25}
    ;;
  *)
    EPOCHS=${EPOCHS:-100}
    LEARNING_RATE=${LEARNING_RATE:-0.001}
    VALIDATION_SPLIT=${VALIDATION_SPLIT:-0.2}
    ;;
esac

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
WRAPPER_LOG="results/log/$(basename "$0" .sh)_task${TASK_ID}_${TIMESTAMP}.log"
exec > >(tee -a "$WRAPPER_LOG") 2>&1

echo "[Wrapper log] $WRAPPER_LOG"
echo "[Python] $PYTHON_BIN"

TRAIN_SCRIPT="script/train_task${TASK_ID}.py"
SELECTED_SCRIPT="script/train_task${TASK_ID}_selected_models.sh"
SUMMARY_FILE="results/mamba_ablation_parallel_task${TASK_ID}_${TIMESTAMP}.txt"

MAMBA_D_MODEL=${MAMBA_D_MODEL:-256}
MAMBA_N_LAYERS=${MAMBA_N_LAYERS:-4}
MAMBA_D_STATE=${MAMBA_D_STATE:-64}
MAMBA_DROPOUT=${MAMBA_DROPOUT:-0.1}

DEVICE_LIST_NORMALIZED="${DEVICE_LIST:-}"
DEVICE_LIST_NORMALIZED="${DEVICE_LIST_NORMALIZED//,/ }"
read -r -a JOB_DEVICES <<< "$DEVICE_LIST_NORMALIZED"
while [ "${#JOB_DEVICES[@]}" -lt 4 ]; do
  JOB_DEVICES+=("$DEVICE")
done

declare -a JOB_LABELS=()
declare -a JOB_PIDS=()
declare -a JOB_LOGS=()
declare -a JOB_RESULTS=()
declare -a JOB_EXIT_CODES=()

build_device_args() {
  local device_value="$1"
  DEVICE_ARGS=()
  if [ -n "$device_value" ]; then
    DEVICE_ARGS=(--device "$device_value")
  fi
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

start_job() {
  local label="$1"
  local job_log="$2"
  shift 2

  echo "[Start] $label"
  echo "[Job log] $job_log"
  print_command "$@"

  JOB_LABELS+=("$label")
  JOB_LOGS+=("$job_log")

  if [ "$DRY_RUN" = "1" ]; then
    JOB_PIDS+=("dry-run")
    return
  fi

  (
    "$@" 2>&1 | tee -a "$job_log"
  ) &
  JOB_PIDS+=("$!")
}

build_device_args "${JOB_DEVICES[0]}"
start_job \
  "mamba" \
  "results/log/mamba_task${TASK_ID}_${TIMESTAMP}.log" \
  env \
  TIMESTAMP="${TIMESTAMP}_mamba" \
  EPOCHS="$EPOCHS" \
  BATCH_SIZE="$BATCH_SIZE" \
  LEARNING_RATE="$LEARNING_RATE" \
  VALIDATION_SPLIT="$VALIDATION_SPLIT" \
  N_RUNS="$N_RUNS" \
  DEVICE="${JOB_DEVICES[0]}" \
  bash "$SELECTED_SCRIPT" mamba

build_device_args "${JOB_DEVICES[1]}"
start_job \
  "mamba_mean" \
  "results/log/mamba_mean_task${TASK_ID}_${TIMESTAMP}.log" \
  "$PYTHON_BIN" script/ablation_test.py \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --learning_rate "$LEARNING_RATE" \
  --validation_split "$VALIDATION_SPLIT" \
  --n_runs "$N_RUNS" \
  --task_id "$TASK_ID" \
  "${DEVICE_ARGS[@]}" \
  mamba_mean

build_device_args "${JOB_DEVICES[2]}"
start_job \
  "mamba_max" \
  "results/log/mamba_max_task${TASK_ID}_${TIMESTAMP}.log" \
  "$PYTHON_BIN" script/ablation_test.py \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --learning_rate "$LEARNING_RATE" \
  --validation_split "$VALIDATION_SPLIT" \
  --n_runs "$N_RUNS" \
  --task_id "$TASK_ID" \
  "${DEVICE_ARGS[@]}" \
  mamba_max

build_device_args "${JOB_DEVICES[3]}"
start_job \
  "mamba_meanmax" \
  "results/log/mamba_meanmax_task${TASK_ID}_${TIMESTAMP}.log" \
  "$PYTHON_BIN" script/ablation_test.py \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --learning_rate "$LEARNING_RATE" \
  --validation_split "$VALIDATION_SPLIT" \
  --n_runs "$N_RUNS" \
  --task_id "$TASK_ID" \
  "${DEVICE_ARGS[@]}" \
  mamba_meanmax

if [ "$DRY_RUN" = "1" ]; then
  echo "[Dry run] Commands printed only."
  exit 0
fi

overall_status=0
for i in "${!JOB_PIDS[@]}"; do
  pid="${JOB_PIDS[$i]}"
  label="${JOB_LABELS[$i]}"

  if wait "$pid"; then
    JOB_EXIT_CODES+=("0")
    echo "[Done] $label"
  else
    JOB_EXIT_CODES+=("$?")
    echo "[Failed] $label" >&2
    overall_status=1
  fi

  result_file=$(python - <<'PY' "${JOB_LOGS[$i]}"
import sys

result_path = ""
with open(sys.argv[1], "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if "Results written to " in line:
            result_path = line.strip().split("Results written to ", 1)[1]
print(result_path)
PY
)
  JOB_RESULTS+=("$result_file")
done

{
  echo "Task ${TASK_ID} - Parallel Mamba Ablation Launcher"
  echo "Generated: $(date)"
  echo "Wrapper log: $WRAPPER_LOG"
  echo "Python: $PYTHON_BIN"
  echo "N_RUNS=$N_RUNS"
  echo "EPOCHS=$EPOCHS"
  echo "BATCH_SIZE=$BATCH_SIZE"
  echo "LEARNING_RATE=$LEARNING_RATE"
  echo "VALIDATION_SPLIT=$VALIDATION_SPLIT"
  echo ""
  for i in "${!JOB_LABELS[@]}"; do
    echo "Job: ${JOB_LABELS[$i]}"
    echo "  Exit code: ${JOB_EXIT_CODES[$i]}"
    echo "  Job log: ${JOB_LOGS[$i]}"
    if [ -n "${JOB_RESULTS[$i]}" ]; then
      echo "  Results file: ${JOB_RESULTS[$i]}"
    fi
    echo ""
  done
} > "$SUMMARY_FILE"

echo "[Summary] $SUMMARY_FILE"
exit "$overall_status"