#!/bin/bash
# Run all ablation pooling models with fixed parameters
# N_RUNS=10 EPOCHS=100 DEVICE=cuda:0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=0.001
VALIDATION_SPLIT=0.2
N_RUNS=10
TASK_ID=1
DEVICE=cuda:0

# All ablation models
BASE_MODELS=(
  "cnn"
  "tcn"
  "vit"
  "inceptiontime"
  "lstm"
  "gru"
  "cnn_lstm"
  "cnn_transformer"
  "mamba"
)

POOLING_METHODS=("mean" "max" "meanmax")

MODELS=()
for base_model in "${BASE_MODELS[@]}"; do
  for pooling_method in "${POOLING_METHODS[@]}"; do
    MODELS+=("${base_model}_${pooling_method}")
  done
done

# Run python script with all ablation models
PYTHONPATH=. python script/ablation_test.py \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --learning_rate "$LEARNING_RATE" \
  --validation_split "$VALIDATION_SPLIT" \
  --n_runs "$N_RUNS" \
  --task_id "$TASK_ID" \
  --device "$DEVICE" \
  "${MODELS[@]}"

