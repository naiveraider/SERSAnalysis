#!/bin/bash
# Train Task 1 with LSTM+Attention multiple times and average the final test accuracy.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-0.001}
VALIDATION_SPLIT=${VALIDATION_SPLIT:-0.2}
LSTM_HIDDEN_SIZE=${LSTM_HIDDEN_SIZE:-128}
LSTM_NUM_LAYERS=${LSTM_NUM_LAYERS:-2}
LSTM_DROPOUT=${LSTM_DROPOUT:-0.2}
N_RUNS=${N_RUNS:-10}

mkdir -p results
RESULTS_FILE="results/task1_lstm_attention_summary.txt"
{
  echo "Task 1 - LSTM+Attention Test Accuracy Summary"
  echo "Generated: $(date)"
  echo "N_RUNS=$N_RUNS"
  echo ""
} > "$RESULTS_FILE"

run_n_times_and_average() {
  local results_file="$1"
  local label="$2"
  shift 2
  local n=$N_RUNS
  local sum=0
  local count=0
  local accs=()
  local i

  set +e
  for i in $(seq 1 $n); do
    echo "---------- Run $i/$n ----------"
    local log_file
    log_file=$(mktemp)
    if "$@" 2>&1 | tee "$log_file"; then
      local acc
      acc=$(python - <<'PY' "$log_file"
import re
import sys

path = sys.argv[1]
value = ""
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        match = re.search(r"FINAL_TEST_ACC=([0-9.]+)", line)
        if match:
            value = match.group(1)
print(value)
PY
)
      if [ -n "$acc" ]; then
        sum=$(python - <<'PY' "$sum" "$acc"
import sys
print(float(sys.argv[1]) + float(sys.argv[2]))
PY
)
        count=$((count + 1))
        accs+=("$acc")
      fi
    fi
    rm -f "$log_file"
  done
  set -e

  if [ "$count" -gt 0 ]; then
    local avg
    avg=$(python - <<'PY' "$sum" "$count"
import sys
print(f"{float(sys.argv[1]) / int(sys.argv[2]):.2f}")
PY
)
    echo ""
    echo "=========================================="
    echo "$label - Average Test Accuracy ($count/$n runs): ${avg}%"
    echo "=========================================="
    echo "Model: $label" >> "$results_file"
    for i in "${!accs[@]}"; do
      echo "  Run $((i+1)): ${accs[i]}%" >> "$results_file"
    done
    echo "  Average: ${avg}%" >> "$results_file"
    echo "" >> "$results_file"
  else
    echo "WARNING: No successful runs for $label" >&2
  fi
}

echo "=========================================="
echo "Training LSTM+Attention for Task 1 ($N_RUNS runs, result averaged)"
echo "=========================================="
run_n_times_and_average "$RESULTS_FILE" "LSTM+Attention" python script/train_task1.py \
  --model lstm_attention \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --learning_rate "$LEARNING_RATE" \
  --validation_split "$VALIDATION_SPLIT" \
  --lstm_hidden_size "$LSTM_HIDDEN_SIZE" \
  --lstm_num_layers "$LSTM_NUM_LAYERS" \
  --lstm_dropout "$LSTM_DROPOUT"

echo ""
echo "Results written to $RESULTS_FILE"
