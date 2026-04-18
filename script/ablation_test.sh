#!/bin/bash
# Run all ablation MeanMax models with fixed parameters
# N_RUNS=10 EPOCHS=100 DEVICE=cuda:0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=0.001
VALIDATION_SPLIT=0.2
N_RUNS=10
DEVICE=cuda:0

# All ablation models
MODELS=(
  "cnn_meanmax"
  "tcn_meanmax"
  "vit_meanmax"
  "inceptiontime_meanmax"
  "lstm_meanmax"
  "cnn_lstm_meanmax"
  "cnn_transformer_meanmax"
  "mamba_meanmax"
)

# Run python script with all ablation models
PYTHONPATH=. python script/ablation_test.py \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --learning_rate "$LEARNING_RATE" \
  --validation_split "$VALIDATION_SPLIT" \
  --n_runs "$N_RUNS" \
  --device "$DEVICE" \
  "${MODELS[@]}"

