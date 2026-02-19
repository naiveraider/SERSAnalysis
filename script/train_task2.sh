#!/bin/bash
# Training script for Task 2
# Trains all models (CNN, TCN, CNN+Transformer, Mamba/S4, ViT, Static Hybrid) on data from datasets/2/
# Each model is run N_RUNS times; final test accuracy is averaged.

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=========================================="
echo "Training Task 2 - All Models"
echo "=========================================="
echo ""

# Default parameters
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-0.001}
VALIDATION_SPLIT=${VALIDATION_SPLIT:-0.2}
N_RUNS=${N_RUNS:-10}

mkdir -p results
RESULTS_FILE="results/task2_summary.txt"
{
  echo "Task 2 - Test Accuracy Summary"
  echo "Generated: $(date)"
  echo "N_RUNS=$N_RUNS"
  echo ""
} > "$RESULTS_FILE"
echo "[Results will be written to $RESULTS_FILE]"
echo ""

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
    local log
    log=$(mktemp)
    if "$@" 2>&1 | tee "$log"; then
      local acc
      acc=$(grep "FINAL_TEST_ACC=" "$log" | tail -1 | cut -d= -f2)
      if [ -n "$acc" ]; then
        sum=$(echo "$sum + $acc" | bc)
        count=$((count + 1))
        accs+=( "$acc" )
      fi
    fi
    rm -f "$log"
  done
  set -e
  if [ "$count" -gt 0 ]; then
    local avg
    avg=$(echo "scale=2; $sum / $count" | bc)
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
echo "Training CNN Model for Task 2 ($N_RUNS runs, result averaged)"
echo "=========================================="
run_n_times_and_average "$RESULTS_FILE" "CNN" python script/train_task2.py \
    --model cnn \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --validation_split $VALIDATION_SPLIT

echo ""
echo "=========================================="
echo "Training TCN Model for Task 2 ($N_RUNS runs, result averaged)"
echo "=========================================="
run_n_times_and_average "$RESULTS_FILE" "TCN" python script/train_task2.py \
    --model tcn \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --validation_split $VALIDATION_SPLIT \
    --tcn_num_channels 64 128 256 \
    --tcn_kernel_size 3 \
    --tcn_dropout 0.2

echo ""
echo "=========================================="
echo "Training CNN+Transformer Model for Task 2 ($N_RUNS runs, result averaged)"
echo "=========================================="
run_n_times_and_average "$RESULTS_FILE" "CNN+Transformer" python script/train_task2.py \
    --model cnn_transformer \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --validation_split $VALIDATION_SPLIT \
    --cnn_transformer_cnn_channels 64 128 256 \
    --cnn_transformer_d_model 256 \
    --cnn_transformer_nhead 8 \
    --cnn_transformer_num_layers 2 \
    --cnn_transformer_dim_feedforward 512 \
    --cnn_transformer_dropout 0.1

echo ""
echo "=========================================="
echo "Training Mamba/S4 Model for Task 2 ($N_RUNS runs, result averaged)"
echo "=========================================="
run_n_times_and_average "$RESULTS_FILE" "Mamba/S4" python script/train_task2.py \
    --model mamba \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --validation_split $VALIDATION_SPLIT \
    --mamba_d_model 256 \
    --mamba_n_layers 4 \
    --mamba_d_state 64 \
    --mamba_dropout 0.1

echo ""
echo "=========================================="
echo "Training ViT Model for Task 2 ($N_RUNS runs, result averaged)"
echo "=========================================="
run_n_times_and_average "$RESULTS_FILE" "ViT" python script/train_task2.py \
    --model vit \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --validation_split $VALIDATION_SPLIT \
    --vit_patch_size 16 \
    --vit_d_model 256 \
    --vit_nhead 8 \
    --vit_num_layers 6 \
    --vit_dim_feedforward 1024 \
    --vit_dropout 0.1

echo ""
echo "=========================================="
echo "Training Static Hybrid Model for Task 2 ($N_RUNS runs, result averaged)"
echo "=========================================="
run_n_times_and_average "$RESULTS_FILE" "Static Hybrid" python script/train_task2.py \
    --model static_hybrid \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --validation_split $VALIDATION_SPLIT \
    --static_hybrid_cnn_channels 64 128 256 \
    --static_hybrid_rnn_hidden 256 \
    --static_hybrid_rnn_layers 2 \
    --static_hybrid_rnn_type LSTM \
    --static_hybrid_dropout 0.2

echo ""
echo "=========================================="
echo "Training LSTM Model for Task 2 ($N_RUNS runs, result averaged)"
echo "=========================================="
run_n_times_and_average "$RESULTS_FILE" "LSTM" python script/train_task2.py \
    --model lstm --epochs $EPOCHS --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE \
    --validation_split $VALIDATION_SPLIT --lstm_hidden_size 128 --lstm_num_layers 2 --lstm_dropout 0.2

echo ""
echo "=========================================="
echo "Training GRU Model for Task 2 ($N_RUNS runs, result averaged)"
echo "=========================================="
run_n_times_and_average "$RESULTS_FILE" "GRU" python script/train_task2.py \
    --model gru --epochs $EPOCHS --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE \
    --validation_split $VALIDATION_SPLIT --lstm_hidden_size 128 --lstm_num_layers 2 --lstm_dropout 0.2

echo ""
echo "=========================================="
echo "Training InceptionTime Model for Task 2 ($N_RUNS runs, result averaged)"
echo "=========================================="
run_n_times_and_average "$RESULTS_FILE" "InceptionTime" python script/train_task2.py \
    --model inceptiontime --epochs $EPOCHS --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE \
    --validation_split $VALIDATION_SPLIT --inceptiontime_n_filters 32 --inceptiontime_depth 6 --inceptiontime_dropout 0.2

echo ""
echo "=========================================="
echo "Training MiniROCKET Model for Task 2 ($N_RUNS runs, result averaged)"
echo "=========================================="
run_n_times_and_average "$RESULTS_FILE" "MiniROCKET" python script/train_task2.py \
    --model minirocket --epochs $EPOCHS --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE \
    --validation_split $VALIDATION_SPLIT --minirocket_num_kernels 1000 --minirocket_seed 42 --minirocket_dropout 0.2

echo ""
echo "=========================================="
echo "Task 2 Training Complete!"
echo "=========================================="

