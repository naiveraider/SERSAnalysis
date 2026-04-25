#!/bin/bash
# Run specified Task 1 models N times each and average final classification metrics.
#
# Usage:
#   bash script/train_task1_selected_models.sh lstm gru bigru_meanmax
#   N_RUNS=10 EPOCHS=100 bash script/train_task1_selected_models.sh bilstm layernorm_bigru

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

mkdir -p results/log
TIMESTAMP=${TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}
LOG_FILE="results/log/$(basename "$0" .sh)_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[Log will be written to $LOG_FILE]"

if [ "$#" -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  echo "Usage: bash script/train_task1_selected_models.sh <model1> [model2 ...]"
  echo ""
  echo "Example:"
  echo "  bash script/train_task1_selected_models.sh lstm gru bigru_meanmax"
  echo "  N_RUNS=10 EPOCHS=100 bash script/train_task1_selected_models.sh bilstm layernorm_bigru"
  exit 0
fi

EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-0.001}
VALIDATION_SPLIT=${VALIDATION_SPLIT:-0.2}
N_RUNS=${N_RUNS:-10}
DEVICE=${DEVICE:-}

COMMON_ARGS=(
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --learning_rate "$LEARNING_RATE"
  --validation_split "$VALIDATION_SPLIT"
)

if [ -n "$DEVICE" ]; then
  COMMON_ARGS+=(--device "$DEVICE")
fi

LSTM_HIDDEN_SIZE=${LSTM_HIDDEN_SIZE:-128}
LSTM_NUM_LAYERS=${LSTM_NUM_LAYERS:-2}
LSTM_DROPOUT=${LSTM_DROPOUT:-0.2}

mkdir -p results
RESULTS_FILE="results/task1_selected_models_${TIMESTAMP}.txt"
{
  echo "Task 1 - Selected Models Test Metrics Summary"
  echo "Generated: $(date)"
  echo "N_RUNS=$N_RUNS"
  echo "Models: $*"
  echo ""
} > "$RESULTS_FILE"

compute_average() {
  local precision="$1"
  shift
  python - <<'PY' "$precision" "$@"
import math
import sys

precision = int(sys.argv[1])
values = []
for raw in sys.argv[2:]:
    try:
        value = float(raw)
    except ValueError:
        continue
    if math.isnan(value):
        continue
    values.append(value)

if not values:
    print("nan")
else:
    print(f"{sum(values) / len(values):.{precision}f}")
PY
}

run_n_times_and_average() {
  local results_file="$1"
  local label="$2"
  shift 2
  local n=$N_RUNS
  local count=0
  local accs=()
  local f1s=()
  local jaccards=()
  local auprcs=()
  local aurocs=()
  local i

  set +e
  for i in $(seq 1 $n); do
    echo "---------- $label Run $i/$n ----------"
    local log_file
    log_file=$(mktemp)
    if "$@" 2>&1 | tee "$log_file"; then
      local metrics
      metrics=$(python - <<'PY' "$log_file"
import sys

keys = [
    "FINAL_TEST_ACC",
    "FINAL_TEST_F1",
    "FINAL_TEST_JACCARD",
    "FINAL_TEST_AUPRC",
    "FINAL_TEST_AUROC",
]
values = {key: "" for key in keys}

with open(sys.argv[1], "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        for key in keys:
            prefix = f"{key}="
            if line.startswith(prefix):
                values[key] = line[len(prefix):]

print("\t".join(values[key] for key in keys))
PY
)
      local acc f1 jaccard auprc auroc
      IFS=$'\t' read -r acc f1 jaccard auprc auroc <<< "$metrics"
      if [ -n "$acc" ] && [ -n "$f1" ] && [ -n "$jaccard" ] && [ -n "$auprc" ] && [ -n "$auroc" ]; then
        count=$((count + 1))
        accs+=("$acc")
        f1s+=("$f1")
        jaccards+=("$jaccard")
        auprcs+=("$auprc")
        aurocs+=("$auroc")
        echo "Extracted metrics: accuracy_score=${acc}% f1_score=${f1} jaccard_score=${jaccard} average_precision_score=${auprc} roc_auc_score=${auroc}"
      fi
    fi
    rm -f "$log_file"
  done
  set -e

  if [ "$count" -gt 0 ]; then
    local avg_acc avg_f1 avg_jaccard avg_auprc avg_auroc
    avg_acc=$(compute_average 2 "${accs[@]}")
    avg_f1=$(compute_average 4 "${f1s[@]}")
    avg_jaccard=$(compute_average 4 "${jaccards[@]}")
    avg_auprc=$(compute_average 4 "${auprcs[@]}")
    avg_auroc=$(compute_average 4 "${aurocs[@]}")
    echo ""
    echo "=========================================="
    echo "$label - Average Test Metrics ($count/$n runs):"
    echo "  accuracy_score: ${avg_acc}%"
    echo "  f1_score: ${avg_f1}"
    echo "  jaccard_score: ${avg_jaccard}"
    echo "  average_precision_score: ${avg_auprc}"
    echo "  roc_auc_score: ${avg_auroc}"
    echo "=========================================="
    echo "Model: $label" >> "$results_file"
    for i in "${!accs[@]}"; do
      echo "  Run $((i + 1)): accuracy_score=${accs[i]}%, f1_score=${f1s[i]}, jaccard_score=${jaccards[i]}, average_precision_score=${auprcs[i]}, roc_auc_score=${aurocs[i]}" >> "$results_file"
    done
    echo "  Average: accuracy_score=${avg_acc}%, f1_score=${avg_f1}, jaccard_score=${avg_jaccard}, average_precision_score=${avg_auprc}, roc_auc_score=${avg_auroc}" >> "$results_file"
    echo "" >> "$results_file"
  else
    echo "WARNING: No successful runs for $label" >&2
  fi
}

build_model_args() {
  local model="$1"
  MODEL_EXTRA_ARGS=()

  case "$model" in
    tcn)
      MODEL_EXTRA_ARGS=(--tcn_num_channels 64 128 256 --tcn_kernel_size 3 --tcn_dropout 0.2)
      ;;
    cnn_transformer)
      MODEL_EXTRA_ARGS=(
        --cnn_transformer_cnn_channels 64 128 256
        --cnn_transformer_d_model 256
        --cnn_transformer_nhead 8
        --cnn_transformer_num_layers 2
        --cnn_transformer_dim_feedforward 512
        --cnn_transformer_dropout 0.1
      )
      ;;
    mamba|s4)
      MODEL_EXTRA_ARGS=(--mamba_d_model 256 --mamba_n_layers 4 --mamba_d_state 64 --mamba_dropout 0.1)
      ;;
    vit)
      MODEL_EXTRA_ARGS=(
        --vit_patch_size 16
        --vit_d_model 256
        --vit_nhead 8
        --vit_num_layers 6
        --vit_dim_feedforward 1024
        --vit_dropout 0.1
      )
      ;;
    static_hybrid)
      MODEL_EXTRA_ARGS=(
        --static_hybrid_cnn_channels 64 128 256
        --static_hybrid_rnn_hidden 256
        --static_hybrid_rnn_layers 2
        --static_hybrid_rnn_type LSTM
        --static_hybrid_dropout 0.2
      )
      ;;
    lstm|stacked_lstm|bilstm|lstm_attention|lstm_cnn|cnn_lstm|cnn_bilstm_attention|gru|bigru|bigru_meanmax|layernorm_bigru)
      MODEL_EXTRA_ARGS=(
        --lstm_hidden_size "$LSTM_HIDDEN_SIZE"
        --lstm_num_layers "$LSTM_NUM_LAYERS"
        --lstm_dropout "$LSTM_DROPOUT"
      )
      ;;
    inceptiontime)
      MODEL_EXTRA_ARGS=(--inceptiontime_n_filters 32 --inceptiontime_depth 6 --inceptiontime_dropout 0.2)
      ;;
    minirocket)
      MODEL_EXTRA_ARGS=(--minirocket_num_kernels 1000 --minirocket_seed 42 --minirocket_dropout 0.2)
      ;;
    cnn)
      MODEL_EXTRA_ARGS=()
      ;;
    *)
      echo "Unknown or unsupported model in script: $model" >&2
      exit 1
      ;;
  esac
}

for model in "$@"; do
  build_model_args "$model"
  echo ""
  echo "=========================================="
  echo "Training $model for Task 1 ($N_RUNS runs, result averaged)"
  echo "=========================================="
  run_n_times_and_average "$RESULTS_FILE" "$model" python script/train_task1.py \
    --model "$model" \
    "${COMMON_ARGS[@]}" \
    "${MODEL_EXTRA_ARGS[@]}"
done

echo ""
echo "Results written to $RESULTS_FILE"
